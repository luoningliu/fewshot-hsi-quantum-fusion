from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.run_hybridsn_small_fewshot import OnTheFlyPatchDataset, _load_dataset_config, _preprocess_full_image
from scripts.run_hybridsn_small_spectral_qnn_gated_metric_fewshot import SpectralQNNGatedMetricFusion
from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.datasets.hsi_dataset import load_hsi_mat
from src.models.classical import HybridSNSmall
from src.utils.seed import set_seed


METRICS = ("OA", "AA", "Kappa", "Macro-F1", "Weighted-F1")
TRAINED_MODELS = ("frozen_linear", "frozen_linear_proto", "spectral_mlp_proto", "spectral_qnn_proto")
GATED_MODELS = ("spectral_mlp_proto", "spectral_qnn_proto")
COMPARISON_PAIRS = (
    ("frozen_linear", "hybridsn_small"),
    ("spectral_mlp_proto", "frozen_linear"),
    ("spectral_qnn_proto", "frozen_linear"),
    ("spectral_qnn_proto", "spectral_mlp_proto"),
)


class FeatureDataset(Dataset):
    def __init__(self, z: np.ndarray, spectra: np.ndarray, labels: np.ndarray, indices: list[int]):
        self.z = z
        self.spectra = spectra
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        return (
            torch.from_numpy(self.z[idx]).float(),
            torch.from_numpy(self.spectra[idx]).float(),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )


class FrozenLinearControl(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes))
        self.feature_norm = nn.LayerNorm(embedding_dim)

    def fused_features(self, z: torch.Tensor, spectrum: torch.Tensor | None = None) -> torch.Tensor:
        return self.feature_norm(z)

    def forward(self, z: torch.Tensor, spectrum: torch.Tensor | None = None, return_aux: bool = False):
        logits = self.head(z)
        if return_aux:
            return logits, {"base_logits": logits}
        return logits


class SpectralMLPGatedMetricFusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        spectral_dim: int,
        num_classes: int,
        mlp_hidden_dim: int = 6,
        mlp_output_dim: int = 6,
        gate_mode: str = "classwise",
    ):
        super().__init__()
        self.base_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes))
        self.spectral_mlp = nn.Sequential(
            nn.LayerNorm(spectral_dim),
            nn.Linear(spectral_dim, int(mlp_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(mlp_hidden_dim), int(mlp_output_dim)),
        )
        self.spectral_head = nn.Linear(int(mlp_output_dim), num_classes)
        gate_dim = 1 if gate_mode == "scalar" else num_classes
        if gate_mode not in {"scalar", "classwise"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        self.gate = nn.Sequential(
            nn.LayerNorm(embedding_dim + spectral_dim),
            nn.Linear(embedding_dim + spectral_dim, gate_dim),
            nn.Sigmoid(),
        )
        self.feature_norm = nn.LayerNorm(embedding_dim + int(mlp_output_dim))

    def spectral_features(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.spectral_mlp(spectrum)

    def fused_features(self, z: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        return self.feature_norm(torch.cat([z, self.spectral_features(spectrum)], dim=1))

    def forward(self, z: torch.Tensor, spectrum: torch.Tensor, return_aux: bool = False):
        base_logits = self.base_head(z)
        spectral_feature = self.spectral_features(spectrum)
        spectral_logits = self.spectral_head(spectral_feature)
        gate = self.gate(torch.cat([z, spectrum], dim=1))
        logits = base_logits + gate * spectral_logits
        if return_aux:
            return logits, {
                "gate": gate,
                "base_logits": base_logits,
                "spectral_logits": spectral_logits,
                "spectral_feature": spectral_feature,
            }
        return logits


def main() -> None:
    args = _build_parser().parse_args()
    out = Path(args.output_dir)
    _prepare_output(out)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    device = _resolve_device(args.device)
    all_rows = _load_hybridsn_rows(args)
    failures: list[dict[str, Any]] = []
    _write_outputs(out, args, all_rows, failures)

    for dataset_name in args.datasets:
        data_cfg = _load_dataset_config(dataset_name, args.data_root)
        raw = load_hsi_mat(data_cfg)
        rows, cols = np.nonzero(raw.gt != raw.background_label)
        labels = raw.gt[rows, cols].astype(np.int64) - 1
        num_classes = int(data_cfg["num_classes"])
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(f"Labels for {dataset_name} do not match config classes.")
        cube_pca, pca_evr_sum = _preprocess_full_image(raw.cube, args.pca_bands, args.seed)
        radius = args.patch_size // 2
        padded_cube = np.pad(cube_pca, ((radius, radius), (radius, radius), (0, 0)), mode="reflect").astype(np.float32)

        for shot in args.shots:
            for seed in args.seeds:
                split_path = Path(args.split_dir) / f"{dataset_name}_seed{int(seed)}_{int(shot)}shot.json"
                for model_name in args.models:
                    try:
                        row = _run_one(
                            args=args,
                            out=out,
                            dataset_name=dataset_name,
                            data_cfg=data_cfg,
                            padded_cube=padded_cube,
                            rows=rows,
                            cols=cols,
                            labels=labels,
                            num_classes=num_classes,
                            shot=int(shot),
                            seed=int(seed),
                            model_name=model_name,
                            split_path=split_path,
                            pca_evr_sum=pca_evr_sum,
                            device=device,
                        )
                        all_rows.append(row)
                    except Exception as exc:
                        failure = {
                            "dataset": dataset_name,
                            "model": model_name,
                            "shot": int(shot),
                            "seed": int(seed),
                            "error": repr(exc),
                        }
                        failures.append(failure)
                        print(f"[WARN] skipped {failure}")
                    _write_outputs(out, args, all_rows, failures)
    print(f"Result directory: {out}")


def _run_one(
    args: argparse.Namespace,
    out: Path,
    dataset_name: str,
    data_cfg: dict[str, Any],
    padded_cube: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    shot: int,
    seed: int,
    model_name: str,
    split_path: Path,
    pca_evr_sum: float,
    device: torch.device,
) -> dict[str, Any]:
    if model_name not in TRAINED_MODELS:
        raise ValueError(f"Unsupported fair-control model: {model_name}")
    set_seed(seed)
    split = _load_split(split_path)
    z, spectra, feature_path = _load_or_extract_features(
        args, out, dataset_name, padded_cube, rows, cols, labels, num_classes, shot, seed, device
    )
    model = _make_model(args, model_name, z.shape[1], spectra.shape[1], num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_loader = _loader(z, spectra, labels, split["train"], args.batch_size, True)
    val_loader = _loader(z, spectra, labels, split["validation"], args.batch_size, False)
    test_loader = _loader(z, spectra, labels, split["test"], args.test_batch_size, False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_proto = model_name.endswith("_proto")
    train_indices = split["train"]
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0
    best_epoch = 0
    best_log: dict[str, Any] | None = None
    stale = 0
    logs: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            z,
            spectra,
            labels,
            train_indices,
            num_classes,
            args.metric_weight if use_proto else 0.0,
            args.temperature,
        )
        val_loss, y_val, pred_val = _evaluate(model, val_loader, criterion, device)
        val_metrics = classification_metrics(y_val, pred_val, labels=list(range(num_classes)))
        log_row = {
            "dataset": dataset_name,
            "model": model_name,
            "shot": shot,
            "seed": seed,
            "epoch": epoch,
            **train_stats,
            "val_loss": val_loss,
            "val_OA": val_metrics["OA"],
            "val_AA": val_metrics["AA"],
            "val_Kappa": val_metrics["Kappa"],
            "val_Macro-F1": val_metrics["Macro-F1"],
            "val_Weighted-F1": val_metrics["Weighted-F1"],
        }
        logs.append(log_row)
        monitor = val_metrics["OA"] if args.monitor == "oa" else val_metrics["Macro-F1"]
        if monitor > best_metric:
            best_metric = monitor
            best_epoch = epoch
            best_log = log_row
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    train_time = time.time() - started
    model.load_state_dict(best_state)
    stem = _stem(dataset_name, model_name, shot, seed)
    ckpt_path = out / "checkpoints" / f"{stem}_checkpoint.pt"
    torch.save(best_state, ckpt_path)
    test_started = time.time()
    test_loss, y_test, pred_test = _evaluate(model, test_loader, criterion, device)
    test_time = time.time() - test_started
    metrics = classification_metrics(y_test, pred_test, labels=list(range(num_classes)))
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class = per_class_metrics(y_test, pred_test, class_names)
    per_class["accuracy"] = per_class["recall"]
    cm = confusion_matrix(y_test, pred_test, labels=list(range(num_classes)))
    per_class.to_csv(out / "metrics" / f"{stem}_per_class_metrics.csv", index=False)
    pd.DataFrame(logs).to_csv(out / "logs" / f"{stem}_training_log.csv", index=False)
    pd.DataFrame(cm).to_csv(out / "confusion_matrices" / f"{stem}_confusion_matrix.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(
        out / "confusion_matrices" / f"{stem}_normalized_confusion_matrix.csv", index=False
    )
    if model_name in GATED_MODELS:
        _write_gate_values(out, stem, dataset_name, model_name, shot, seed, model, test_loader, device)
    payload = {
        "dataset": dataset_name,
        "model": model_name,
        "shot": shot,
        "seed": seed,
        "num_classes": num_classes,
        "pca_fit_scope": "full_image_unsupervised",
        "pca_evr_sum": pca_evr_sum,
        "patch_size": args.patch_size,
        "pca_bands": args.pca_bands,
        "best_epoch": best_epoch,
        "train_loss": None if best_log is None else best_log["train_loss"],
        "val_loss": None if best_log is None else best_log["val_loss"],
        "test_loss": test_loss,
        "train_time_seconds": train_time,
        "test_time_seconds": test_time,
        "trainable_parameters": param_count,
        "feature_path": str(feature_path),
        "checkpoint": str(ckpt_path),
        **metrics,
    }
    write_json(out / "metrics" / f"{stem}.json", payload)
    print(
        f"{dataset_name} {model_name} {shot}-shot seed{seed}: "
        f"OA={metrics['OA'] * 100:.2f} Macro-F1={metrics['Macro-F1'] * 100:.2f} best_epoch={best_epoch}"
    )
    return {
        "dataset": dataset_name,
        "model": model_name,
        "shot": shot,
        "seed": seed,
        **metrics,
        "best_epoch": best_epoch,
        "train_loss": np.nan if best_log is None else best_log["train_loss"],
        "val_loss": np.nan if best_log is None else best_log["val_loss"],
        "trainable_parameters": param_count,
        "train_size": len(split["train"]),
        "validation_size": len(split["validation"]),
        "test_size": len(split["test"]),
        "train_time_seconds": train_time,
        "test_time_seconds": test_time,
    }


def _make_model(args: argparse.Namespace, model_name: str, embedding_dim: int, spectral_dim: int, num_classes: int) -> nn.Module:
    if model_name in {"frozen_linear", "frozen_linear_proto"}:
        return FrozenLinearControl(embedding_dim, num_classes)
    if model_name == "spectral_mlp_proto":
        return SpectralMLPGatedMetricFusion(
            embedding_dim=embedding_dim,
            spectral_dim=spectral_dim,
            num_classes=num_classes,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_output_dim=args.mlp_output_dim,
            gate_mode=args.gate_mode,
        )
    if model_name == "spectral_qnn_proto":
        return SpectralQNNGatedMetricFusion(
            embedding_dim=embedding_dim,
            spectral_dim=spectral_dim,
            num_classes=num_classes,
            gate_mode=args.gate_mode,
            qubits=args.qubits,
            layers=args.qnn_layers,
            entanglement=args.entanglement,
            backend=args.backend,
            diff_method=args.diff_method,
            normalize_input=True,
            angle_scale=args.angle_scale,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    z_np: np.ndarray,
    spectra_np: np.ndarray,
    labels_np: np.ndarray,
    train_indices: list[int],
    num_classes: int,
    metric_weight: float,
    temperature: float,
) -> dict[str, float]:
    model.train()
    train_idx = np.asarray(train_indices, dtype=np.int64)
    support_z = torch.from_numpy(z_np[train_idx]).float().to(device)
    support_s = torch.from_numpy(spectra_np[train_idx]).float().to(device)
    support_y = torch.from_numpy(labels_np[train_idx]).long().to(device)
    loss_sum = ce_sum = metric_sum = 0.0
    correct = count = 0
    for z, spectra, y, _ in loader:
        z = z.to(device)
        spectra = spectra.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(z, spectra)
        ce_loss = criterion(logits, y)
        metric_loss = torch.tensor(0.0, device=device)
        if metric_weight > 0:
            support_features = model.fused_features(support_z, support_s)
            query_features = model.fused_features(z, spectra)
            metric_logits = _prototype_logits(query_features, support_features, support_y, num_classes, temperature)
            metric_loss = criterion(metric_logits, y)
        loss = ce_loss + float(metric_weight) * metric_loss
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * len(y)
        ce_sum += float(ce_loss.item()) * len(y)
        metric_sum += float(metric_loss.item()) * len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        count += len(y)
    return {
        "train_loss": loss_sum / max(count, 1),
        "train_ce_loss": ce_sum / max(count, 1),
        "train_metric_loss": metric_sum / max(count, 1),
        "train_accuracy": correct / max(count, 1),
    }


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    loss_sum = 0.0
    count = 0
    ys, preds = [], []
    with torch.no_grad():
        for z, spectra, y, _ in loader:
            z = z.to(device)
            spectra = spectra.to(device)
            yy = y.to(device)
            logits = model(z, spectra)
            loss_sum += float(criterion(logits, yy).item()) * len(y)
            count += len(y)
            ys.append(y.numpy())
            preds.append(logits.argmax(1).cpu().numpy())
    return loss_sum / max(count, 1), np.concatenate(ys), np.concatenate(preds)


def _prototype_logits(
    query_features: torch.Tensor,
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes: int,
    temperature: float,
) -> torch.Tensor:
    query = F.normalize(query_features, dim=1)
    support = F.normalize(support_features, dim=1)
    prototypes = []
    for class_id in range(num_classes):
        class_mask = support_labels == class_id
        prototypes.append(support[class_mask].mean(dim=0))
    prototypes = F.normalize(torch.stack(prototypes, dim=0), dim=1)
    return -(torch.cdist(query, prototypes, p=2) ** 2) / float(temperature)


def _write_gate_values(
    out: Path,
    stem: str,
    dataset: str,
    model_name: str,
    shot: int,
    seed: int,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    model.eval()
    frames = []
    with torch.no_grad():
        for z, spectra, y, sample_indices in loader:
            logits, aux = model(z.to(device), spectra.to(device), return_aux=True)
            base_logits = aux["base_logits"].detach().cpu()
            spectral_logits = aux["spectral_logits"].detach().cpu()
            logits = logits.detach().cpu()
            pred = logits.argmax(dim=1)
            gate = aux["gate"].detach().cpu()
            if gate.shape[1] == 1:
                gate_by_class = gate.expand(-1, logits.shape[1])
            else:
                gate_by_class = gate
            records = []
            for item in range(len(y)):
                true_class = int(y[item].item())
                pred_class = int(pred[item].item())
                records.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "shot": shot,
                        "seed": seed,
                        "sample_index": int(sample_indices[item].item()),
                        "true_label": true_class,
                        "pred_label": pred_class,
                        "correct": bool(true_class == pred_class),
                        "mean_gate": float(gate_by_class[item].mean().item()),
                        "max_gate": float(gate_by_class[item].max().item()),
                        "gate_for_pred_class": float(gate_by_class[item, pred_class].item()),
                        "gate_for_true_class": float(gate_by_class[item, true_class].item()),
                        "base_margin": _margin(base_logits[item], pred_class),
                        "spectral_margin": _margin(spectral_logits[item], pred_class),
                        "final_margin": _margin(logits[item], pred_class),
                    }
                )
            frames.append(pd.DataFrame(records))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "metrics" / f"{stem}_gate_values.csv", index=False)


def _margin(logits: torch.Tensor, class_id: int) -> float:
    other = torch.cat([logits[:class_id], logits[class_id + 1 :]])
    return float(logits[class_id].item() - other.max().item())


def _load_or_extract_features(
    args: argparse.Namespace,
    out: Path,
    dataset_name: str,
    padded_cube: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    shot: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, Path]:
    feature_path = out / "features" / f"{dataset_name}_shot{shot}_seed{seed}_features.npz"
    if feature_path.exists() and not args.rebuild_features:
        cached = np.load(feature_path)
        if {"z", "spectra"}.issubset(cached.files):
            if not {"y", "rows", "cols"}.issubset(cached.files):
                np.savez_compressed(
                    feature_path,
                    z=cached["z"],
                    spectra=cached["spectra"],
                    y=labels.astype(np.int64),
                    rows=rows.astype(np.int64),
                    cols=cols.astype(np.int64),
                )
            return cached["z"].astype(np.float32), cached["spectra"].astype(np.float32), feature_path
    ckpt_path = Path(args.encoder_checkpoint_dir) / f"{dataset_name}_shot{shot}_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing HybridSN-small checkpoint: {ckpt_path}")
    encoder = HybridSNSmall(
        pca_channels=args.pca_bands,
        num_classes=num_classes,
        patch_size=args.patch_size,
        conv3d_channels=tuple(args.conv3d_channels),
        conv2d_channels=args.conv2d_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    for module in (encoder.conv3d, encoder.conv2d):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    encoder.eval()
    dataset = OnTheFlyPatchDataset(padded_cube, rows, cols, labels, list(range(len(labels))), args.patch_size)
    loader = DataLoader(dataset, batch_size=args.feature_batch_size, shuffle=False, num_workers=0)
    center = args.patch_size // 2
    z_parts, spectra_parts = [], []
    with torch.no_grad():
        for patches, _ in loader:
            spectra_parts.append(patches[:, center, center, :].numpy())
            patches = encoder._prepare_input(patches.to(device))
            encoded = encoder.conv3d(patches)
            batch, channels, spectral, height, width = encoded.shape
            encoded = encoded.reshape(batch, channels * spectral, height, width)
            z_parts.append(encoder.conv2d(encoded).flatten(1).cpu().numpy())
    z = np.concatenate(z_parts, axis=0).astype(np.float32)
    spectra = np.concatenate(spectra_parts, axis=0).astype(np.float32)
    np.savez_compressed(
        feature_path,
        z=z,
        spectra=spectra,
        y=labels.astype(np.int64),
        rows=rows.astype(np.int64),
        cols=cols.astype(np.int64),
    )
    return z, spectra, feature_path


def _load_hybridsn_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    metrics_path = Path(args.hybridsn_result_dir) / "metrics" / "all_runs.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing HybridSN-small baseline metrics: {metrics_path}. Run "
            "`python scripts/run_hybridsn_small_fewshot.py --datasets indian_pines pavia_university salinas "
            "--shots 5 10 --seeds 0 1 2 3 4 --patch_size 19 --pca_bands 30 --epochs 200 --patience 30 "
            "--output_root result --run_dir result/hybridsn_small_fewshot_3datasets` first."
        )
    baseline = pd.read_csv(metrics_path)
    baseline = baseline[
        baseline["dataset"].isin(args.datasets)
        & baseline["shot"].isin([int(shot) for shot in args.shots])
        & baseline["seed"].isin([int(seed) for seed in args.seeds])
    ].copy()
    baseline["model"] = "hybridsn_small"
    for column in ("train_loss", "val_loss"):
        if column not in baseline:
            baseline[column] = np.nan
    columns = [
        "dataset",
        "model",
        "shot",
        "seed",
        *METRICS,
        "best_epoch",
        "train_loss",
        "val_loss",
        "trainable_parameters",
        "train_size",
        "validation_size",
        "test_size",
        "train_time_seconds",
        "test_time_seconds",
    ]
    return baseline.reindex(columns=columns).to_dict("records")


def _write_outputs(out: Path, args: argparse.Namespace, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    write_json(out / "failed_runs.json", {"failed_runs": failures})
    if not rows:
        return
    all_runs = pd.DataFrame(rows)
    all_runs = all_runs.sort_values(["dataset", "shot", "model", "seed"])
    all_runs.to_csv(out / "metrics" / "all_runs.csv", index=False)
    _write_main_summary(out, all_runs)
    _write_baseline_comparison(out, all_runs)
    _write_qnn_mlp_comparison(out, all_runs)
    _write_paired_seed_delta(out, all_runs)
    _write_gate_summaries(out)
    _write_report(out, args, all_runs, failures)


def _write_main_summary(out: Path, all_runs: pd.DataFrame) -> None:
    rows = []
    for (dataset, shot, model), group in all_runs.groupby(["dataset", "shot", "model"]):
        row = {"Dataset": dataset, "Shot": int(shot), "Model": model, "Runs": len(group)}
        for metric in METRICS:
            row[f"{metric} mean±std"] = _mean_std(group[metric])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["Dataset", "Shot", "Model"])
    summary.to_csv(out / "metrics" / "summary_by_dataset_shot_model.csv", index=False)
    (out / "metrics" / "summary_by_dataset_shot_model.md").write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")


def _write_baseline_comparison(out: Path, all_runs: pd.DataFrame) -> None:
    means = all_runs.groupby(["dataset", "shot", "model"], as_index=False)[list(METRICS)].mean()
    rows = []
    for (dataset, shot), group in means.groupby(["dataset", "shot"]):
        baseline = group[group["model"] == "hybridsn_small"]
        if baseline.empty:
            continue
        base_row = baseline.iloc[0]
        for _, row in group[group["model"] != "hybridsn_small"].iterrows():
            out_row = {"Dataset": dataset, "Shot": int(shot), "Model": row["model"]}
            for metric in METRICS:
                out_row[f"Δ{metric}"] = float(row[metric] - base_row[metric])
            rows.append(out_row)
    columns = ["Dataset", "Shot", "Model", "ΔOA", "ΔAA", "ΔKappa", "ΔMacro-F1", "ΔWeighted-F1"]
    comparison = pd.DataFrame(rows, columns=columns)
    comparison.to_csv(out / "metrics" / "comparison_vs_hybridsn_small.csv", index=False)
    (out / "metrics" / "comparison_vs_hybridsn_small.md").write_text(_markdown_or_empty(comparison), encoding="utf-8")


def _write_qnn_mlp_comparison(out: Path, all_runs: pd.DataFrame) -> None:
    means = all_runs.groupby(["dataset", "shot", "model"], as_index=False)[["OA", "Macro-F1"]].mean()
    rows = []
    for (dataset, shot), group in means.groupby(["dataset", "shot"]):
        qnn = group[group["model"] == "spectral_qnn_proto"]
        mlp = group[group["model"] == "spectral_mlp_proto"]
        if qnn.empty or mlp.empty:
            continue
        delta_macro = float(qnn.iloc[0]["Macro-F1"] - mlp.iloc[0]["Macro-F1"])
        rows.append(
            {
                "Dataset": dataset,
                "Shot": int(shot),
                "QNN OA": float(qnn.iloc[0]["OA"]),
                "MLP OA": float(mlp.iloc[0]["OA"]),
                "ΔOA": float(qnn.iloc[0]["OA"] - mlp.iloc[0]["OA"]),
                "QNN Macro-F1": float(qnn.iloc[0]["Macro-F1"]),
                "MLP Macro-F1": float(mlp.iloc[0]["Macro-F1"]),
                "ΔMacro-F1": delta_macro,
                "Winner": "QNN" if delta_macro > 0 else "MLP" if delta_macro < 0 else "Tie",
            }
        )
    columns = ["Dataset", "Shot", "QNN OA", "MLP OA", "ΔOA", "QNN Macro-F1", "MLP Macro-F1", "ΔMacro-F1", "Winner"]
    comparison = pd.DataFrame(rows, columns=columns)
    comparison.to_csv(out / "metrics" / "comparison_qnn_vs_mlp.csv", index=False)
    (out / "metrics" / "comparison_qnn_vs_mlp.md").write_text(_markdown_or_empty(comparison), encoding="utf-8")


def _write_paired_seed_delta(out: Path, all_runs: pd.DataFrame) -> None:
    rows = []
    keyed = all_runs.set_index(["dataset", "shot", "seed", "model"])
    for dataset, shot, seed in all_runs[["dataset", "shot", "seed"]].drop_duplicates().itertuples(index=False):
        for model_a, model_b in COMPARISON_PAIRS:
            key_a = (dataset, shot, seed, model_a)
            key_b = (dataset, shot, seed, model_b)
            if key_a not in keyed.index or key_b not in keyed.index:
                continue
            row_a = keyed.loc[key_a]
            row_b = keyed.loc[key_b]
            for metric in METRICS:
                rows.append(
                    {
                        "Dataset": dataset,
                        "Shot": int(shot),
                        "Seed": int(seed),
                        "Model A": model_a,
                        "Model B": model_b,
                        "Metric": metric,
                        "Value A": float(row_a[metric]),
                        "Value B": float(row_b[metric]),
                        "Delta": float(row_a[metric] - row_b[metric]),
                    }
                )
    columns = ["Dataset", "Shot", "Seed", "Model A", "Model B", "Metric", "Value A", "Value B", "Delta"]
    pd.DataFrame(rows, columns=columns).to_csv(out / "metrics" / "paired_seed_delta.csv", index=False)


def _write_gate_summaries(out: Path) -> None:
    gate_paths = sorted((out / "metrics").glob("*_gate_values.csv"))
    columns = ["Dataset", "Shot", "Model", "Samples", "mean_gate", "std_gate", "mean_gate_correct", "mean_gate_wrong"]
    class_columns = ["Dataset", "Shot", "Model", "true_label", "samples", "mean_gate_by_class", "std_gate_by_class"]
    if not gate_paths:
        pd.DataFrame(columns=columns).to_csv(out / "metrics" / "gate_summary_by_dataset_shot_model.csv", index=False)
        pd.DataFrame(columns=class_columns).to_csv(out / "metrics" / "gate_summary_by_class.csv", index=False)
        return
    gates = pd.concat([pd.read_csv(path) for path in gate_paths], ignore_index=True)
    summary_rows, class_rows = [], []
    for (dataset, shot, model), group in gates.groupby(["dataset", "shot", "model"]):
        correct = group[group["correct"].astype(bool)]
        wrong = group[~group["correct"].astype(bool)]
        summary_rows.append(
            {
                "Dataset": dataset,
                "Shot": int(shot),
                "Model": model,
                "Samples": len(group),
                "mean_gate": float(group["mean_gate"].mean()),
                "std_gate": float(group["mean_gate"].std(ddof=0)),
                "mean_gate_correct": float(correct["mean_gate"].mean()) if not correct.empty else np.nan,
                "mean_gate_wrong": float(wrong["mean_gate"].mean()) if not wrong.empty else np.nan,
            }
        )
        for true_label, class_group in group.groupby("true_label"):
            class_rows.append(
                {
                    "Dataset": dataset,
                    "Shot": int(shot),
                    "Model": model,
                    "true_label": int(true_label),
                    "samples": len(class_group),
                    "mean_gate_by_class": float(class_group["mean_gate"].mean()),
                    "std_gate_by_class": float(class_group["mean_gate"].std(ddof=0)),
                }
            )
    pd.DataFrame(summary_rows, columns=columns).sort_values(["Dataset", "Shot", "Model"]).to_csv(
        out / "metrics" / "gate_summary_by_dataset_shot_model.csv", index=False
    )
    pd.DataFrame(class_rows, columns=class_columns).sort_values(["Dataset", "Shot", "Model", "true_label"]).to_csv(
        out / "metrics" / "gate_summary_by_class.csv", index=False
    )


def _write_report(out: Path, args: argparse.Namespace, all_runs: pd.DataFrame, failures: list[dict[str, Any]]) -> None:
    summary = _read_text(out / "metrics" / "summary_by_dataset_shot_model.md")
    vs_baseline = _read_text(out / "metrics" / "comparison_vs_hybridsn_small.md")
    qnn_vs_mlp = _read_text(out / "metrics" / "comparison_qnn_vs_mlp.md")
    paired_text = _paired_positive_seed_lines(out / "metrics" / "paired_seed_delta.csv")
    gate_text = _gate_report_text(out / "metrics" / "gate_summary_by_dataset_shot_model.csv")
    conclusion = _conclusion_text(out / "metrics" / "comparison_qnn_vs_mlp.csv", out / "metrics" / "comparison_vs_hybridsn_small.csv")
    model_table = pd.DataFrame(
        [
            {
                "模型": "HybridSN-small",
                "构造": "原始 few-shot HybridSN-small 端到端 baseline",
                "作用": "给出经典 spectral-spatial baseline",
            },
            {
                "模型": "Frozen HybridSN + Linear head",
                "构造": "冻结 conv3d/conv2d pooled feature z，仅训练 LayerNorm + Linear",
                "作用": "排除重新训练分类 head 的影响",
            },
            {
                "模型": "Spectral MLP Gated Fusion + Prototype",
                "构造": "z 与中心 PCA spectrum 的 MLP branch 做 gated residual fusion，并加 prototype loss",
                "作用": "经典 center spectral branch 对照",
            },
            {
                "模型": "Spectral QNN Gated Fusion + Prototype",
                "构造": "同一 gated fusion/prototype 设置下将 MLP branch 换成 QNN branch",
                "作用": "检验 QNN branch 的独立贡献",
            },
        ]
    ).to_markdown(index=False)
    lines = [
        "# 公平对照 few-shot HSI 实验报告",
        "",
        "## 1. 实验目的",
        "",
        "本实验固定 HybridSN-small encoder checkpoint、few-shot split、seed 和测试流程，用于拆分 QNN 增益来源：",
        "",
        "`HybridSN-small -> Frozen Linear -> Spectral MLP -> Spectral QNN`",
        "",
        "`frozen_linear_proto` 是额外控制项：它仅在 frozen feature `z` 上加入 prototype loss，用于判断 metric loss 本身的影响。",
        "",
        "## 2. 四个模型说明",
        "",
        model_table,
        "",
        "## 3. 主结果表",
        "",
        summary,
        "",
        "## 4. 相对 HybridSN-small 的提升",
        "",
        vs_baseline,
        "",
        "## 5. QNN vs MLP",
        "",
        qnn_vs_mlp,
        "",
        "## 6. Paired seed 分析",
        "",
        paired_text,
        "",
        "## 7. Gate 分析",
        "",
        gate_text,
        "",
        "## 8. 初步结论",
        "",
        conclusion,
        "",
        "## 运行配置",
        "",
        f"- datasets: {args.datasets}",
        f"- shots: {args.shots}",
        f"- seeds: {args.seeds}",
        f"- models: {args.models}",
        f"- monitor: validation {args.monitor}",
        f"- metric_weight: {args.metric_weight}",
        f"- temperature: {args.temperature}",
        f"- failures: {len(failures)}",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paired_positive_seed_lines(path: Path) -> str:
    if not path.exists():
        return "尚无 paired seed delta。"
    paired = pd.read_csv(path)
    paired = paired[paired["Metric"] == "Macro-F1"]
    if paired.empty:
        return "尚无 Macro-F1 paired seed delta。"
    lines = []
    for (dataset, shot, model_a, model_b), group in paired.groupby(["Dataset", "Shot", "Model A", "Model B"]):
        positive = int((group["Delta"] > 0).sum())
        lines.append(
            f"- {dataset} {int(shot)}-shot: {model_a} vs {model_b} Macro-F1 positive seeds = {positive}/{len(group)}"
        )
    return "\n".join(lines)


def _gate_report_text(path: Path) -> str:
    if not path.exists():
        return "尚无 gate 汇总。"
    gate = pd.read_csv(path)
    if gate.empty:
        return "尚无 gate 汇总。"
    lines = ["`mean_gate` 越高表示 spectral residual logits 在最终 logits 中被保留得越多。"]
    for row in gate.sort_values(["Dataset", "Shot", "Model"]).itertuples(index=False):
        lines.append(
            f"- {row.Dataset} {int(row.Shot)}-shot {row.Model}: mean_gate={row.mean_gate:.4f}, "
            f"correct={row.mean_gate_correct:.4f}, wrong={row.mean_gate_wrong:.4f}"
        )
    return "\n".join(lines)


def _conclusion_text(qnn_path: Path, baseline_path: Path) -> str:
    conclusions = []
    qnn = pd.read_csv(qnn_path) if qnn_path.exists() else pd.DataFrame()
    if qnn.empty:
        conclusions.append("QNN 与 MLP 的 direct comparison 尚未齐全，暂不判断 spectral branch 类型贡献。")
    elif (qnn["ΔMacro-F1"] > 0).all():
        conclusions.append("在已完成的 dataset/shot 上 QNN 的 mean Macro-F1 全部高于 MLP，可将其视为 quantum spectral branch 的独立正向证据。")
    elif (qnn["ΔMacro-F1"] < 0).all():
        conclusions.append("在已完成的 dataset/shot 上 QNN 的 mean Macro-F1 全部低于 MLP，当前 QNN 结构尚未超过经典 spectral branch。")
    elif qnn["ΔMacro-F1"].abs().mean() <= 0.01:
        conclusions.append("QNN 与 MLP 的 mean Macro-F1 差异整体接近，当前结果更支持 spectral branch + prototype 是主要因素，QNN 是可竞争实现。")
    else:
        conclusions.append("QNN 与 MLP 在不同 dataset/shot 上结果不一致，应优先看 paired seed delta，而不是只看 pooled mean。")
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    salinas = baseline[
        (baseline.get("Dataset") == "salinas")
        & (baseline.get("Shot") == 10)
        & (baseline.get("Model").isin(["spectral_mlp_proto", "spectral_qnn_proto"]))
    ] if not baseline.empty else pd.DataFrame()
    if not salinas.empty and (salinas["ΔMacro-F1"] < 0).any():
        conclusions.append("Salinas 10-shot 仍存在相对 baseline 的下降，经典 baseline 接近饱和时，当前 spectral branch 可能产生负迁移。")
    return "\n\n".join(conclusions)


def _prepare_output(out: Path) -> None:
    for subdir in ("features", "checkpoints", "logs", "metrics", "confusion_matrices"):
        (out / subdir).mkdir(parents=True, exist_ok=True)


def _loader(z: np.ndarray, spectra: np.ndarray, labels: np.ndarray, indices: list[int], batch_size: int, shuffle: bool):
    return DataLoader(FeatureDataset(z, spectra, labels, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _load_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline split file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stem(dataset: str, model_name: str, shot: int, seed: int) -> str:
    return f"{dataset}_{model_name}_shot{shot}_seed{seed}"


def _mean_std(values: pd.Series) -> str:
    return f"{values.mean():.4f}±{values.std(ddof=0):.4f}"


def _markdown_or_empty(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) + "\n" if not frame.empty else "No completed comparison rows.\n"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else "No completed rows."


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fair-control frozen HybridSN few-shot model comparisons.")
    parser.add_argument("--datasets", nargs="+", default=["indian_pines"], choices=["indian_pines", "pavia_university", "salinas"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--shots", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=list(TRAINED_MODELS), choices=list(TRAINED_MODELS))
    parser.add_argument("--encoder_checkpoint_dir", default="result/hybridsn_small_fewshot_3datasets/checkpoints")
    parser.add_argument("--split_dir", default="result/hybridsn_small_fewshot_3datasets/split_indices")
    parser.add_argument("--hybridsn_result_dir", default="result/hybridsn_small_fewshot_3datasets")
    parser.add_argument("--output_dir", default="result/fair_control_models_fewshot")
    parser.add_argument("--patch_size", type=int, default=19)
    parser.add_argument("--pca_bands", type=int, default=30)
    parser.add_argument("--conv3d_channels", nargs=3, type=int, default=[8, 16, 16])
    parser.add_argument("--conv2d_channels", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--metric_weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--gate_mode", choices=["scalar", "classwise"], default="classwise")
    parser.add_argument("--mlp_hidden_dim", type=int, default=6)
    parser.add_argument("--mlp_output_dim", type=int, default=6)
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--qnn_layers", type=int, default=1)
    parser.add_argument("--entanglement", default="linear")
    parser.add_argument("--backend", default="lightning.qubit")
    parser.add_argument("--diff_method", default="adjoint")
    parser.add_argument("--angle_scale", type=float, default=float(np.pi))
    parser.add_argument("--monitor", choices=["macro_f1", "oa"], default="macro_f1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Unsupervised full-image PCA seed.")
    parser.add_argument("--rebuild_features", action="store_true")
    return parser


if __name__ == "__main__":
    main()
