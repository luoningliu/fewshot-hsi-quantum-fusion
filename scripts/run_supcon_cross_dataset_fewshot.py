from __future__ import annotations

import argparse
import copy
import json
import shutil
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

from scripts.run_fair_control_models_fewshot import (
    FeatureDataset,
    _load_or_extract_features as _load_or_extract_features_base,
    _load_split,
    _resolve_device,
    _write_gate_values,
)
from scripts.run_hybridsn_small_fewshot import _load_dataset_config, _preprocess_full_image
from scripts.run_hybridsn_small_spectral_qnn_gated_metric_fewshot import SpectralQNNGatedMetricFusion
from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.datasets.hsi_dataset import load_hsi_mat
from src.models.classical import HybridSNSmall
from src.utils.seed import set_seed


METRICS = ("OA", "AA", "Kappa", "Macro-F1", "Weighted-F1")
MODEL_NAME = "spectral_qnn_gated_supcon"
JSON_MODEL_NAME = "hybridsn_small_spectral_qnn_gated_fusion_supcon"


class SupConFeatureDataset(Dataset):
    def __init__(
        self,
        z: np.ndarray,
        spectra: np.ndarray,
        labels: np.ndarray,
        indices: list[int],
        base_logits: np.ndarray | None = None,
    ):
        self.z = z
        self.spectra = spectra
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.base_logits = base_logits

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        values = [
            torch.from_numpy(self.z[idx]).float(),
            torch.from_numpy(self.spectra[idx]).float(),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        ]
        if self.base_logits is not None:
            values.append(torch.from_numpy(self.base_logits[idx]).float())
        return tuple(values)


def main() -> None:
    args = _build_parser().parse_args()
    out = Path(args.output_dir)
    _prepare_output(out)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_training_config(out, args)
    device = _resolve_device(args.device)
    all_rows: list[dict[str, Any]] = _load_existing_rows(out) if args.resume else []
    failures: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        data_cfg = _load_dataset_config(dataset_name, args.data_root)
        raw = load_hsi_mat(data_cfg)
        rows, cols = np.nonzero(raw.gt != raw.background_label)
        labels = raw.gt[rows, cols].astype(np.int64) - 1
        num_classes = int(data_cfg["num_classes"])
        cube_pca, pca_evr_sum = _preprocess_full_image(raw.cube, args.pca_bands, args.seed)
        radius = args.patch_size // 2
        padded_cube = np.pad(cube_pca, ((radius, radius), (radius, radius), (0, 0)), mode="reflect").astype(np.float32)

        for shot in args.shots:
            for seed in args.seeds:
                if args.resume and _completed(out, dataset_name, int(shot), int(seed), args):
                    print(f"[SKIP] completed {dataset_name} SupCon {int(shot)}-shot seed{int(seed)}")
                    continue
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
                        pca_evr_sum=float(pca_evr_sum),
                        device=device,
                    )
                    all_rows.append(row)
                except Exception as exc:
                    failure = {
                        "dataset": dataset_name,
                        "shot": int(shot),
                        "seed": int(seed),
                        "model": _model_name(args),
                        "stage": "supcon_run",
                        "status": "failed",
                        "reason": repr(exc),
                    }
                    failures.append(failure)
                    print(f"[WARN] failed {failure}")
                _write_outputs(out, all_rows, failures)

    _write_outputs(out, all_rows, failures)
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
    pca_evr_sum: float,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    split_path = _resolve_cached_input(
        [args.split_dir, *args.split_dirs],
        f"{dataset_name}_seed{seed}_{shot}shot.json",
        "few-shot split",
    )
    split = _load_split(split_path)
    _copy_cached_features_if_available(args, out, dataset_name, shot, seed)
    z, spectra, base_logits, feature_path = _load_or_extract_features(
        args, out, dataset_name, padded_cube, rows, cols, labels, num_classes, shot, seed, device
    )
    model = SpectralQNNGatedMetricFusion(
        embedding_dim=z.shape[1],
        spectral_dim=spectra.shape[1],
        num_classes=num_classes,
        gate_mode=args.gate_mode,
        qnn_variant=args.qnn_variant,
        residual_scale_mode=args.residual_scale_mode,
        residual_alpha_init=args.residual_alpha_init,
        gate_context_mode=args.gate_context_mode,
        high_confidence_guard_mode=args.high_confidence_guard_mode,
        guard_floor=args.guard_floor,
        guard_tau=args.guard_tau,
        guard_temperature=args.guard_temperature,
        base_logit_mode=args.base_logit_mode,
        qubits=args.qubits,
        layers=args.qnn_layers,
        entanglement=args.entanglement,
        backend=args.backend,
        diff_method=args.diff_method,
        normalize_input=True,
        angle_scale=args.angle_scale,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_loader = _loader(z, spectra, labels, split["train"], args.batch_size, True, base_logits)
    val_loader = _loader(z, spectra, labels, split["validation"], args.batch_size, False, base_logits)
    test_loader = _loader(z, spectra, labels, split["test"], args.test_batch_size, False, base_logits)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0
    best_epoch = 0
    best_log: dict[str, Any] | None = None
    stale = 0
    logs: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        _set_residual_warmup(model, epoch, args.residual_warmup_epochs)
        train_stats = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.metric_weight,
            args.temperature,
            args.gate_confidence_penalty,
        )
        val_loss, y_val, pred_val = _evaluate(model, val_loader, criterion, device)
        val_metrics = classification_metrics(y_val, pred_val, labels=list(range(num_classes)))
        log_row = {
            "dataset": dataset_name,
            "model": _model_name(args),
            "shot": shot,
            "seed": seed,
            "epoch": epoch,
            "residual_scale": float(model.residual_scale().detach().cpu().item()),
            "residual_warmup_factor": float(model.residual_warmup_factor.detach().cpu().item()),
            "high_confidence_guard_mode": args.high_confidence_guard_mode,
            "guard_floor": args.guard_floor,
            "guard_tau": args.guard_tau,
            "guard_temperature": args.guard_temperature,
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
    model_name = _model_name(args)
    json_model_name = _json_model_name(args)
    stem = f"{dataset_name}_{model_name}_shot{shot}_seed{seed}"
    ckpt_path = out / "checkpoints" / f"{stem}.pt"
    torch.save(best_state, ckpt_path)
    test_started = time.time()
    test_loss, y_test, pred_test = _evaluate(model, test_loader, criterion, device)
    test_time = time.time() - test_started
    metrics = classification_metrics(y_test, pred_test, labels=list(range(num_classes)))
    residual_scale = float(model.residual_scale().detach().cpu().item())
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class = per_class_metrics(y_test, pred_test, class_names)
    per_class["accuracy"] = per_class["recall"]
    cm = confusion_matrix(y_test, pred_test, labels=list(range(num_classes)))
    norm_cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    per_class.to_csv(out / "metrics" / f"{stem}_per_class.csv", index=False)
    per_class.to_csv(out / "metrics" / f"{stem}_per_class_metrics.csv", index=False)
    pd.DataFrame(logs).to_csv(out / "logs" / f"{stem}.csv", index=False)
    pd.DataFrame(cm).to_csv(out / "confusion_matrices" / f"{stem}.csv", index=False)
    pd.DataFrame(norm_cm).to_csv(out / "confusion_matrices" / f"{stem}_normalized.csv", index=False)
    _write_gate_values(out, stem, dataset_name, model_name, shot, seed, model, test_loader, device)

    setting_dir = out / "raw" / dataset_name / f"{shot}shot" / f"seed{seed}"
    setting_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(setting_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm).to_csv(setting_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(norm_cm).to_csv(setting_dir / "normalized_confusion_matrix.csv", index=False)
    pd.DataFrame(logs).to_csv(setting_dir / "training_log.csv", index=False)
    _write_training_config(setting_dir, args)

    payload = {
        "dataset": dataset_name,
        "model": json_model_name,
        "model_short": model_name,
        "qnn_variant": args.qnn_variant,
        "residual_scale_mode": args.residual_scale_mode,
        "residual_alpha_init": args.residual_alpha_init,
        "residual_warmup_epochs": args.residual_warmup_epochs,
        "residual_scale_final": residual_scale,
        "gate_context_mode": args.gate_context_mode,
        "gate_confidence_penalty": args.gate_confidence_penalty,
        "base_logit_mode": args.base_logit_mode,
        "high_confidence_guard_mode": args.high_confidence_guard_mode,
        "guard_floor": args.guard_floor,
        "guard_tau": args.guard_tau,
        "guard_temperature": args.guard_temperature,
        "shot": shot,
        "seed": seed,
        "loss_mode": "supcon",
        "metric_weight": args.metric_weight,
        "temperature": args.temperature,
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
    write_json(setting_dir / "metrics.json", payload)
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
        "train_time_seconds": train_time,
        "test_time_seconds": test_time,
        "trainable_parameters": param_count,
        "train_size": len(split["train"]),
        "validation_size": len(split["validation"]),
        "test_size": len(split["test"]),
        "residual_scale_final": residual_scale,
        "mean_train_guard": None if best_log is None else best_log["train_mean_guard"],
    }


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metric_weight: float,
    temperature: float,
    gate_confidence_penalty: float,
) -> dict[str, float]:
    model.train()
    loss_sum = ce_sum = metric_sum = gate_penalty_sum = guard_sum = 0.0
    correct = count = 0
    for batch in loader:
        z, spectra, y, _, base_logits = _unpack_batch(batch)
        z = z.to(device)
        spectra = spectra.to(device)
        y = y.to(device)
        base_logits = None if base_logits is None else base_logits.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(z, spectra, return_aux=True, base_logits=base_logits)
        ce_loss = criterion(logits, y)
        features = model.fused_features(z, spectra)
        metric_loss = _supcon_loss(features, y, temperature)
        gate_penalty = _gate_confidence_loss(aux)
        loss = ce_loss + float(metric_weight) * metric_loss + float(gate_confidence_penalty) * gate_penalty
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * len(y)
        ce_sum += float(ce_loss.item()) * len(y)
        metric_sum += float(metric_loss.item()) * len(y)
        gate_penalty_sum += float(gate_penalty.item()) * len(y)
        guard_sum += float(aux["guard"].mean().item()) * len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        count += len(y)
    return {
        "train_loss": loss_sum / max(count, 1),
        "train_ce_loss": ce_sum / max(count, 1),
        "train_supcon_loss": metric_sum / max(count, 1),
        "train_gate_confidence_penalty": gate_penalty_sum / max(count, 1),
        "train_mean_guard": guard_sum / max(count, 1),
        "train_accuracy": correct / max(count, 1),
    }


def _gate_confidence_loss(aux: dict[str, torch.Tensor]) -> torch.Tensor:
    gate = aux["gate"]
    confidence = aux["base_confidence"].to(gate.device)
    return (gate.mean(dim=1) * confidence).mean()


def _supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    batch = labels.shape[0]
    eye = torch.eye(batch, dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
    logits_mask = ~eye
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return features.new_tensor(0.0)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1)[valid] / positive_count[valid].float()
    return -mean_log_prob_pos.mean()


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    loss_sum = 0.0
    count = 0
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            z, spectra, y, _, base_logits = _unpack_batch(batch)
            z = z.to(device)
            spectra = spectra.to(device)
            yy = y.to(device)
            base_logits = None if base_logits is None else base_logits.to(device)
            logits = model(z, spectra, base_logits=base_logits)
            loss_sum += float(criterion(logits, yy).item()) * len(y)
            count += len(y)
            ys.append(y.numpy())
            preds.append(logits.argmax(1).cpu().numpy())
    return loss_sum / max(count, 1), np.concatenate(ys), np.concatenate(preds)


def _set_residual_warmup(model: nn.Module, epoch: int, warmup_epochs: int) -> None:
    if not hasattr(model, "set_residual_warmup_factor"):
        return
    if warmup_epochs <= 0:
        model.set_residual_warmup_factor(1.0)
        return
    model.set_residual_warmup_factor(min(1.0, float(epoch) / float(warmup_epochs)))


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Path]:
    z, spectra, feature_path = _load_or_extract_features_base(
        args, out, dataset_name, padded_cube, rows, cols, labels, num_classes, shot, seed, device
    )
    if args.base_logit_mode != "pretrained":
        return z, spectra, None, feature_path
    cached = np.load(feature_path)
    if "base_logits" in cached.files and not args.rebuild_features:
        return z, spectra, cached["base_logits"].astype(np.float32), feature_path

    ckpt_path = _resolve_cached_input(
        [args.encoder_checkpoint_dir, *args.encoder_checkpoint_dirs],
        f"{dataset_name}_shot{shot}_seed{seed}.pt",
        "HybridSN-small checkpoint",
    )
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
    encoder.eval()
    logits_parts = []
    with torch.no_grad():
        for start in range(0, len(z), args.feature_batch_size):
            batch_z = torch.from_numpy(z[start : start + args.feature_batch_size]).float().to(device)
            logits_parts.append(encoder.classifier(batch_z).cpu().numpy())
    base_logits = np.concatenate(logits_parts, axis=0).astype(np.float32)
    np.savez_compressed(
        feature_path,
        z=z.astype(np.float32),
        spectra=spectra.astype(np.float32),
        base_logits=base_logits,
        y=labels.astype(np.int64),
        rows=rows.astype(np.int64),
        cols=cols.astype(np.int64),
    )
    return z, spectra, base_logits, feature_path


def _unpack_batch(batch):
    if len(batch) == 5:
        z, spectra, y, sample_indices, base_logits = batch
        return z, spectra, y, sample_indices, base_logits
    z, spectra, y, sample_indices = batch
    return z, spectra, y, sample_indices, None


def _loader(
    z: np.ndarray,
    spectra: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    base_logits: np.ndarray | None = None,
):
    dataset = SupConFeatureDataset(z, spectra, labels, indices, base_logits)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _copy_cached_features_if_available(args: argparse.Namespace, out: Path, dataset: str, shot: int, seed: int) -> None:
    target = out / "features" / f"{dataset}_shot{shot}_seed{seed}_features.npz"
    if target.exists() and not args.rebuild_features:
        return
    for root in args.feature_cache_dirs:
        path = Path(root) / "features" / f"{dataset}_shot{shot}_seed{seed}_features.npz"
        if path.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            return


def _resolve_cached_input(dirs: list[str], filename: str, label: str) -> Path:
    checked = []
    for directory in dirs:
        if not directory:
            continue
        path = Path(directory) / filename
        checked.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {label}: {filename}. Checked: {checked}")


def _write_outputs(out: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    pd.DataFrame(
        failures,
        columns=["dataset", "shot", "seed", "model", "stage", "status", "reason"],
    ).to_csv(out / "failed_runs.csv", index=False)
    write_json(out / "failed_runs.json", {"failed_runs": failures})
    if not rows:
        return
    all_runs = pd.DataFrame(rows).sort_values(["dataset", "shot", "seed"])
    all_runs.to_csv(out / "metrics" / "all_runs_metric_qnn.csv", index=False)
    all_runs.to_csv(out / "seedwise_results.csv", index=False)

    summary_rows = []
    for (dataset, shot, model), group in all_runs.groupby(["dataset", "shot", "model"]):
        row = {"dataset": dataset, "model": model, "shot": int(shot), "runs": len(group)}
        for metric in METRICS:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        row["mean_best_epoch"] = float(group["best_epoch"].mean())
        row["trainable_parameters"] = int(group["trainable_parameters"].iloc[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "shot", "model"])
    summary.to_csv(out / "metrics" / "summary_by_dataset_shot_metric_qnn.csv", index=False)
    summary.to_csv(out / "summary_by_dataset_shot_metric_qnn.csv", index=False)
    _write_task_tables(out, all_runs, summary)
    _write_hybridsn_comparisons(out, all_runs, summary)
    _write_report(out, all_runs, summary, failures)


def _load_existing_rows(out: Path) -> list[dict[str, Any]]:
    path = out / "metrics" / "all_runs_metric_qnn.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _completed(out: Path, dataset: str, shot: int, seed: int, args: argparse.Namespace) -> bool:
    stem = f"{dataset}_{_model_name(args)}_shot{shot}_seed{seed}"
    return (
        (out / "metrics" / f"{stem}.json").exists()
        and (out / "metrics" / f"{stem}_per_class.csv").exists()
        and (out / "confusion_matrices" / f"{stem}.csv").exists()
        and (out / "confusion_matrices" / f"{stem}_normalized.csv").exists()
    )


def _write_task_tables(out: Path, all_runs: pd.DataFrame, summary: pd.DataFrame) -> None:
    rows = []
    for _, row in all_runs.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "shot": int(row["shot"]),
                "model": row["model"],
                "loss_type": "SupCon",
                "seed": int(row["seed"]),
                "oa": float(row["OA"]),
                "aa": float(row["AA"]),
                "kappa": float(row["Kappa"]),
                "macro_f1": float(row["Macro-F1"]),
                "weighted_f1": float(row["Weighted-F1"]),
            }
        )
    pd.DataFrame(rows).to_csv(out / "supcon_cross_dataset_summary.csv", index=False)

    agg_rows = []
    for _, row in summary.iterrows():
        agg_rows.append(
            {
                "dataset": row["dataset"],
                "shot": int(row["shot"]),
                "model": row["model"],
                "loss_type": "SupCon",
                "runs": int(row["runs"]),
                "oa_mean": row["mean_OA"],
                "oa_std": row["std_OA"],
                "aa_mean": row["mean_AA"],
                "aa_std": row["std_AA"],
                "kappa_mean": row["mean_Kappa"],
                "kappa_std": row["std_Kappa"],
                "macro_f1_mean": row["mean_Macro-F1"],
                "macro_f1_std": row["std_Macro-F1"],
                "weighted_f1_mean": row["mean_Weighted-F1"],
                "weighted_f1_std": row["std_Weighted-F1"],
            }
        )
    pd.DataFrame(agg_rows).to_csv(out / "supcon_cross_dataset_aggregate.csv", index=False)


def _write_hybridsn_comparisons(out: Path, all_runs: pd.DataFrame, summary: pd.DataFrame) -> None:
    baseline = _load_hybridsn_baseline_runs()
    if baseline.empty:
        return
    paired = all_runs.merge(
        baseline,
        on=["dataset", "shot", "seed"],
        suffixes=("_qnn", "_hybridsn"),
        how="inner",
    )
    if not paired.empty:
        paired_rows = []
        for _, row in paired.iterrows():
            paired_rows.append(
                {
                    "dataset": row["dataset"],
                    "shot": int(row["shot"]),
                    "seed": int(row["seed"]),
                    "model": row["model"],
                    "hybridsn_oa": float(row["OA_hybridsn"]),
                    "qnn_oa": float(row["OA_qnn"]),
                    "delta_oa": float(row["OA_qnn"] - row["OA_hybridsn"]),
                    "hybridsn_macro_f1": float(row["Macro-F1_hybridsn"]),
                    "qnn_macro_f1": float(row["Macro-F1_qnn"]),
                    "delta_macro_f1": float(row["Macro-F1_qnn"] - row["Macro-F1_hybridsn"]),
                    "hybridsn_weighted_f1": float(row["Weighted-F1_hybridsn"]),
                    "qnn_weighted_f1": float(row["Weighted-F1_qnn"]),
                    "delta_weighted_f1": float(row["Weighted-F1_qnn"] - row["Weighted-F1_hybridsn"]),
                }
            )
        paired_delta = pd.DataFrame(paired_rows).sort_values(["dataset", "shot", "seed"])
        paired_delta.to_csv(out / "paired_seed_delta_vs_hybridsn_small.csv", index=False)
    else:
        paired_delta = pd.DataFrame()

    baseline_summary_rows = []
    for (dataset, shot), group in baseline.groupby(["dataset", "shot"]):
        row = {"dataset": dataset, "shot": int(shot), "baseline_runs": len(group)}
        for metric in METRICS:
            row[f"baseline_{metric}"] = float(group[metric].mean())
        baseline_summary_rows.append(row)
    baseline_summary = pd.DataFrame(baseline_summary_rows)
    comparison = summary.merge(baseline_summary, on=["dataset", "shot"], how="inner")
    if comparison.empty:
        return
    rows = []
    for _, row in comparison.iterrows():
        task_paired = paired_delta[
            (paired_delta["dataset"] == row["dataset"])
            & (paired_delta["shot"] == int(row["shot"]))
            & (paired_delta["model"] == row["model"])
        ]
        positive_oa_seeds = int((task_paired["delta_oa"] > 0).sum()) if not task_paired.empty else 0
        positive_macro_f1_seeds = int((task_paired["delta_macro_f1"] > 0).sum()) if not task_paired.empty else 0
        rows.append(
            {
                "dataset": row["dataset"],
                "shot": int(row["shot"]),
                "model": row["model"],
                "runs": int(row["runs"]),
                "baseline_runs": int(row["baseline_runs"]),
                "baseline_oa": float(row["baseline_OA"]),
                "qnn_oa": float(row["mean_OA"]),
                "delta_oa": float(row["mean_OA"] - row["baseline_OA"]),
                "positive_oa_seeds": positive_oa_seeds,
                "baseline_macro_f1": float(row["baseline_Macro-F1"]),
                "qnn_macro_f1": float(row["mean_Macro-F1"]),
                "delta_macro_f1": float(row["mean_Macro-F1"] - row["baseline_Macro-F1"]),
                "positive_macro_f1_seeds": positive_macro_f1_seeds,
                "passes_oa_macro_f1_rule": bool(
                    row["mean_OA"] > row["baseline_OA"]
                    and row["mean_Macro-F1"] > row["baseline_Macro-F1"]
                    and positive_oa_seeds >= 3
                    and positive_macro_f1_seeds >= 3
                ),
            }
        )
    pd.DataFrame(rows).sort_values(["dataset", "shot", "model"]).to_csv(
        out / "comparison_vs_hybridsn_small.csv", index=False
    )


def _load_hybridsn_baseline_runs() -> pd.DataFrame:
    paths = [
        Path("result/hybridsn_small_fewshot_3datasets/metrics/all_runs.csv"),
        Path("result/hybridsn_small_fewshot_pavia_salinas_5_10shot/metrics/all_runs.csv"),
    ]
    frames = []
    for source_order, path in enumerate(paths):
        if path.exists():
            frame = pd.read_csv(path)
            frame["_source_order"] = source_order
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    baseline = pd.concat(frames, ignore_index=True)
    baseline = baseline.sort_values("_source_order").drop_duplicates(["dataset", "shot", "seed"], keep="last")
    return baseline.drop(columns=["_source_order"])


def _write_report(out: Path, all_runs: pd.DataFrame, summary: pd.DataFrame, failures: list[dict[str, Any]]) -> None:
    display = summary.copy()
    for column in display.columns:
        if column.startswith("mean_") or column.startswith("std_"):
            if column != "mean_best_epoch":
                display[column] = (display[column] * 100).round(2)
            else:
                display[column] = display[column].round(2)
    lines = [
        "# Spectral QNN Gated Fusion + SupCon Cross-dataset Runs",
        "",
        "## Completed Runs",
        "",
        display.to_markdown(index=False) if not display.empty else "No completed runs.",
        "",
        "## Seedwise Results",
        "",
        all_runs.to_markdown(index=False) if len(all_runs) <= 30 else all_runs.head(30).to_markdown(index=False),
        "",
        "## Failures",
        "",
        pd.DataFrame(failures).to_markdown(index=False) if failures else "No failed runs.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_training_config(out: Path, args: argparse.Namespace) -> None:
    lines = []
    for key, value in vars(args).items():
        if isinstance(value, list):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {value}")
    (out / "training_config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _model_name(args: argparse.Namespace) -> str:
    residual = "_residualsafe" if args.residual_scale_mode == "learnable_sigmoid" else ""
    base = "_prelogit" if args.base_logit_mode == "pretrained" else ""
    if args.high_confidence_guard_mode != "none":
        guard = "_confguardb"
    else:
        guard = "_confguard" if args.gate_context_mode != "none" or args.gate_confidence_penalty > 0 else ""
    if args.qnn_variant == "standard":
        return f"spectral_qnn{base}{residual}{guard}_gated_supcon"
    return f"spectral_qnn_{args.qnn_variant}{base}{residual}{guard}_gated_supcon"


def _json_model_name(args: argparse.Namespace) -> str:
    residual = "_residualsafe" if args.residual_scale_mode == "learnable_sigmoid" else ""
    base = "_prelogit" if args.base_logit_mode == "pretrained" else ""
    if args.high_confidence_guard_mode != "none":
        guard = "_confguardb"
    else:
        guard = "_confguard" if args.gate_context_mode != "none" or args.gate_confidence_penalty > 0 else ""
    if args.qnn_variant == "standard":
        return f"hybridsn_small_spectral_qnn{base}{residual}{guard}_gated_fusion_supcon"
    return f"hybridsn_small_spectral_qnn_{args.qnn_variant}{base}{residual}{guard}_gated_fusion_supcon"


def _prepare_output(out: Path) -> None:
    for subdir in ("features", "checkpoints", "logs", "metrics", "confusion_matrices", "raw"):
        (out / subdir).mkdir(parents=True, exist_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled Pavia/Salinas Spectral QNN Gated Fusion + SupCon experiments.")
    parser.add_argument("--datasets", nargs="+", default=["pavia_university", "salinas"], choices=["indian_pines", "pavia_university", "salinas"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--shots", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--encoder_checkpoint_dir", default="result/hybridsn_small_fewshot_pavia_salinas_5_10shot/checkpoints")
    parser.add_argument(
        "--encoder_checkpoint_dirs",
        nargs="*",
        default=["result/hybridsn_small_fewshot_3datasets/checkpoints"],
        help="Additional checkpoint directories searched after --encoder_checkpoint_dir.",
    )
    parser.add_argument("--split_dir", default="result/hybridsn_small_fewshot_pavia_salinas_5_10shot/split_indices")
    parser.add_argument(
        "--split_dirs",
        nargs="*",
        default=["result/hybridsn_small_fewshot_3datasets/split_indices"],
        help="Additional split directories searched after --split_dir.",
    )
    parser.add_argument("--output_dir", default="result/supcon_cross_dataset_pavia_salinas")
    parser.add_argument(
        "--feature_cache_dirs",
        nargs="*",
        default=[
            "result/hybridsn_small_spectral_qnn_gated_proto_pavia_salinas_5_10shot",
            "result/hybridsn_small_spectral_qnn_gated_proto_pavia_seed4_5shot",
            "result/hybridsn_small_spectral_qnn_gated_proto_pavia_10shot_seed0_2",
            "result/hybridsn_small_spectral_qnn_gated_proto_pavia_10shot_seed3",
            "result/hybridsn_small_spectral_qnn_gated_proto_pavia_10shot_seed4",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_partial",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_seed1_2_5shot",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_seed3_4_5shot",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_10shot_partial",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_10shot_seed1_2",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_10shot_seed3_4",
        ],
    )
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
    parser.add_argument("--qnn_variant", choices=["standard", "reupload_multiobs"], default="standard")
    parser.add_argument("--residual_scale_mode", choices=["none", "learnable_sigmoid"], default="none")
    parser.add_argument("--residual_alpha_init", type=float, default=-4.0)
    parser.add_argument("--residual_warmup_epochs", type=int, default=0)
    parser.add_argument("--base_logit_mode", choices=["learned_head", "pretrained"], default="learned_head")
    parser.add_argument("--gate_context_mode", choices=["none", "base_confidence_margin"], default="none")
    parser.add_argument("--gate_confidence_penalty", type=float, default=0.0)
    parser.add_argument("--high_confidence_guard_mode", choices=["none", "margin_suppression"], default="none")
    parser.add_argument("--guard_floor", type=float, default=0.05)
    parser.add_argument("--guard_tau", type=float, default=0.35)
    parser.add_argument("--guard_temperature", type=float, default=0.08)
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
    parser.add_argument("--resume", action="store_true", help="Skip dataset/shot/seed runs with completed metric and confusion files.")
    return parser


if __name__ == "__main__":
    main()
