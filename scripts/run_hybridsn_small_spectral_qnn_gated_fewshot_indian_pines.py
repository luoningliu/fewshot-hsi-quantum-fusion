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
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.run_hybridsn_small_fewshot import OnTheFlyPatchDataset, _load_dataset_config, _preprocess_full_image
from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.datasets.hsi_dataset import load_hsi_mat
from src.models.classical import HybridSNSmall
from src.models.quantum import QNNClassifier
from src.utils.seed import set_seed


class FusionDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, spectra: np.ndarray, labels: np.ndarray, indices: list[int]):
        self.embeddings = embeddings
        self.spectra = spectra
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        return (
            torch.from_numpy(self.embeddings[idx]).float(),
            torch.from_numpy(self.spectra[idx]).float(),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )


class SpectralQNNGatedFusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        spectral_dim: int,
        num_classes: int,
        gate_mode: str = "scalar",
        **qnn_kwargs,
    ):
        super().__init__()
        self.classical_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes))
        self.qnn_head = QNNClassifier(spectral_dim, num_classes, **qnn_kwargs)
        gate_in = embedding_dim + spectral_dim
        if gate_mode == "scalar":
            self.gate = nn.Sequential(nn.LayerNorm(gate_in), nn.Linear(gate_in, 1), nn.Sigmoid())
        elif gate_mode == "classwise":
            self.gate = nn.Sequential(nn.LayerNorm(gate_in), nn.Linear(gate_in, num_classes), nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")

    def forward(self, z: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        base_logits = self.classical_head(z)
        qnn_logits = self.qnn_head(spectrum)
        gate = self.gate(torch.cat([z, spectrum], dim=1))
        return base_logits + gate * qnn_logits


def main() -> None:
    args = _build_parser().parse_args()
    out = Path(args.output_dir)
    for sub in ("features", "metrics", "logs", "checkpoints", "confusion_matrices"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    device = _resolve_device(args.device)

    data_cfg = _load_dataset_config("indian_pines", args.data_root)
    raw = load_hsi_mat(data_cfg)
    rows, cols = np.nonzero(raw.gt != raw.background_label)
    labels = raw.gt[rows, cols].astype(np.int64) - 1
    num_classes = int(data_cfg["num_classes"])
    cube_pca, pca_evr_sum = _preprocess_full_image(raw.cube, args.pca_bands, args.seed)
    radius = args.patch_size // 2
    padded_cube = np.pad(cube_pca, ((radius, radius), (radius, radius), (0, 0)), mode="reflect").astype(np.float32)

    all_rows: list[dict[str, Any]] = []
    for shot in args.shots:
        for seed in args.seeds:
            row = _run_one(args, out, data_cfg, padded_cube, rows, cols, labels, num_classes, shot, seed, pca_evr_sum, device)
            all_rows.append(row)
            _write_summary(out, all_rows)
    _write_comparison_report(out, args)
    print(f"Result directory: {out}")


def _run_one(
    args: argparse.Namespace,
    out: Path,
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
    set_seed(int(seed))
    split = _load_split(Path(args.split_dir) / f"indian_pines_seed{seed}_{shot}shot.json")
    z, spectra = _load_or_extract_features(args, out, padded_cube, rows, cols, labels, num_classes, shot, seed, device)
    model = SpectralQNNGatedFusion(
        embedding_dim=z.shape[1],
        spectral_dim=spectra.shape[1],
        num_classes=num_classes,
        gate_mode=args.gate_mode,
        qubits=args.qubits,
        layers=args.qnn_layers,
        entanglement=args.entanglement,
        backend=args.backend,
        diff_method=args.diff_method,
        normalize_input=True,
        angle_scale=args.angle_scale,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.debug:
        logits = model(torch.zeros(2, z.shape[1], device=device), torch.zeros(2, spectra.shape[1], device=device))
        assert logits.shape == (2, num_classes)

    train_loader = _loader(z, spectra, labels, split["train"], args.batch_size, True)
    val_loader = _loader(z, spectra, labels, split["validation"], args.batch_size, False)
    test_loader = _loader(z, spectra, labels, split["test"], args.test_batch_size, False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0
    best_epoch = 0
    stale = 0
    logs: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_val, pred_val = _evaluate(model, val_loader, criterion, device)
        val_metrics = classification_metrics(y_val, pred_val, labels=list(range(num_classes)))
        logs.append(
            {
                "shot": shot,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_OA": val_metrics["OA"],
                "val_AA": val_metrics["AA"],
                "val_Kappa": val_metrics["Kappa"],
                "val_Macro-F1": val_metrics["Macro-F1"],
                "val_Weighted-F1": val_metrics["Weighted-F1"],
            }
        )
        monitor = val_metrics["OA"] if args.monitor == "oa" else val_metrics["Macro-F1"]
        if monitor > best_metric:
            best_metric = monitor
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    train_time = time.time() - started
    model.load_state_dict(best_state)
    torch.save(best_state, out / "checkpoints" / f"indian_pines_spectral_gated_qnn_shot{shot}_seed{seed}.pt")
    test_started = time.time()
    test_loss, y_test, pred_test = _evaluate(model, test_loader, criterion, device)
    test_time = time.time() - test_started
    metrics = classification_metrics(y_test, pred_test, labels=list(range(num_classes)))
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class_metrics(y_test, pred_test, class_names).to_csv(
        out / "metrics" / f"indian_pines_spectral_gated_qnn_shot{shot}_seed{seed}_per_class.csv",
        index=False,
    )
    cm = confusion_matrix(y_test, pred_test, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(out / "confusion_matrices" / f"indian_pines_spectral_gated_qnn_shot{shot}_seed{seed}.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(
        out / "confusion_matrices" / f"indian_pines_spectral_gated_qnn_shot{shot}_seed{seed}_normalized.csv",
        index=False,
    )
    pd.DataFrame(logs).to_csv(out / "logs" / f"indian_pines_spectral_gated_qnn_shot{shot}_seed{seed}.csv", index=False)
    payload = {
        "dataset": "indian_pines",
        "model": "hybridsn_small_spectral_qnn_gated_fusion",
        "shot": shot,
        "seed": seed,
        "pca_fit_scope": "full_image_unsupervised",
        "pca_evr_sum": pca_evr_sum,
        "patch_size": args.patch_size,
        "pca_bands": args.pca_bands,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "train_time_seconds": train_time,
        "test_time_seconds": test_time,
        "trainable_parameters": param_count,
        **metrics,
    }
    write_json(out / "metrics" / f"indian_pines_spectral_gated_qnn_shot{shot}_seed{seed}.json", payload)
    row = {
        "dataset": "indian_pines",
        "model": "hybridsn_small_spectral_qnn_gated_fusion",
        "shot": shot,
        "seed": seed,
        "OA": metrics["OA"],
        "AA": metrics["AA"],
        "Kappa": metrics["Kappa"],
        "Macro-F1": metrics["Macro-F1"],
        "Weighted-F1": metrics["Weighted-F1"],
        "best_epoch": best_epoch,
        "train_time_seconds": train_time,
        "test_time_seconds": test_time,
        "trainable_parameters": param_count,
        "encoder_feature_dim": z.shape[1],
        "spectral_dim": spectra.shape[1],
        "train_size": len(split["train"]),
        "validation_size": len(split["validation"]),
        "test_size": len(split["test"]),
    }
    print(
        f"Spectral Gated QNN Indian Pines {shot}-shot seed{seed}: "
        f"OA={metrics['OA']*100:.2f} Macro-F1={metrics['Macro-F1']*100:.2f} best_epoch={best_epoch}"
    )
    return row


def _load_or_extract_features(
    args: argparse.Namespace,
    out: Path,
    padded_cube: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    shot: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    feature_path = out / "features" / f"indian_pines_shot{shot}_seed{seed}_features.npz"
    if feature_path.exists() and not args.rebuild_features:
        data = np.load(feature_path)
        return data["z"].astype(np.float32), data["spectra"].astype(np.float32)
    ckpt = Path(args.encoder_checkpoint_dir) / f"indian_pines_shot{shot}_seed{seed}.pt"
    model = HybridSNSmall(
        pca_channels=args.pca_bands,
        num_classes=num_classes,
        patch_size=args.patch_size,
        conv3d_channels=tuple(args.conv3d_channels),
        conv2d_channels=args.conv2d_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    ds = OnTheFlyPatchDataset(padded_cube, rows, cols, labels, list(range(len(labels))), args.patch_size)
    loader = DataLoader(ds, batch_size=args.feature_batch_size, shuffle=False, num_workers=0)
    z_parts, spectrum_parts = [], []
    center = args.patch_size // 2
    with torch.no_grad():
        for x, _ in loader:
            spectrum_parts.append(x[:, center, center, :].numpy())
            x = x.to(device)
            x = model._prepare_input(x)
            x = model.conv3d(x)
            batch, channels, spectral, height, width = x.shape
            x = x.reshape(batch, channels * spectral, height, width)
            x = model.conv2d(x).flatten(1)
            z_parts.append(x.cpu().numpy())
    z = np.concatenate(z_parts, axis=0).astype(np.float32)
    spectra = np.concatenate(spectrum_parts, axis=0).astype(np.float32)
    np.savez_compressed(feature_path, z=z, spectra=spectra, y=labels)
    return z, spectra


def _train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for z, s, y in loader:
        z = z.to(device)
        s = s.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(z, s)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        count += len(y)
    return loss_sum / max(count, 1), correct / max(count, 1)


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    loss_sum = 0.0
    count = 0
    ys, ps = [], []
    with torch.no_grad():
        for z, s, y in loader:
            z = z.to(device)
            s = s.to(device)
            yy = y.to(device)
            logits = model(z, s)
            loss = criterion(logits, yy)
            loss_sum += float(loss.item()) * len(y)
            count += len(y)
            ys.append(y.numpy())
            ps.append(logits.argmax(1).cpu().numpy())
    return loss_sum / max(count, 1), np.concatenate(ys), np.concatenate(ps)


def _loader(z: np.ndarray, spectra: np.ndarray, labels: np.ndarray, indices: list[int], batch_size: int, shuffle: bool):
    return DataLoader(FusionDataset(z, spectra, labels, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _load_split(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_summary(out: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics" / "all_runs_spectral_gated_qnn.csv", index=False)
    summary_rows = []
    for shot, group in df.groupby("shot"):
        row = {"dataset": "indian_pines", "model": "hybridsn_small_spectral_qnn_gated_fusion", "shot": int(shot), "runs": len(group)}
        for metric in ("OA", "AA", "Kappa", "Macro-F1", "Weighted-F1"):
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        row["mean_best_epoch"] = float(group["best_epoch"].mean())
        row["trainable_parameters"] = int(group["trainable_parameters"].iloc[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("shot")
    summary.to_csv(out / "metrics" / "summary_by_shot_spectral_gated_qnn.csv", index=False)
    display = summary.copy()
    for col in display.columns:
        if col.startswith("mean_") or col.startswith("std_"):
            if col != "mean_best_epoch":
                display[col] = (display[col] * 100).round(2)
            else:
                display[col] = display[col].round(2)
    (out / "metrics" / "summary_by_shot_spectral_gated_qnn.md").write_text(display.to_markdown(index=False) + "\n", encoding="utf-8")


def _write_comparison_report(out: Path, args: argparse.Namespace) -> None:
    qnn = pd.read_csv(out / "metrics" / "summary_by_shot_spectral_gated_qnn.csv")
    baseline = pd.read_csv(args.baseline_summary)
    rows = []
    for _, b in baseline.iterrows():
        shot = int(b["shot"])
        q = qnn[qnn["shot"] == shot]
        if q.empty:
            continue
        q = q.iloc[0]
        rows.append(
            {
                "shot": shot,
                "HybridSN-small OA": b["mean_OA"] * 100,
                "Spectral Gated QNN OA": q["mean_OA"] * 100,
                "Delta OA": (q["mean_OA"] - b["mean_OA"]) * 100,
                "HybridSN-small Macro-F1": b["mean_Macro-F1"] * 100,
                "Spectral Gated QNN Macro-F1": q["mean_Macro-F1"] * 100,
                "Delta Macro-F1": (q["mean_Macro-F1"] - b["mean_Macro-F1"]) * 100,
                "runs": int(q["runs"]),
            }
        )
    comp = pd.DataFrame(rows).sort_values("shot")
    comp.to_csv(out / "metrics" / "comparison_vs_hybridsn_small.csv", index=False)
    display = comp.copy()
    for col in display.columns:
        if col not in ("shot", "runs"):
            display[col] = display[col].round(2)
    lines = [
        "# HybridSN-small + Spectral QNN Gated Fusion: Indian Pines Few-shot",
        "",
        "This experiment freezes the HybridSN-small encoder for each shot/seed, feeds the center PCA spectrum to a QNN branch, and fuses it with the classical encoder logits using a learned scalar gate.",
        "",
        "## Comparison",
        "",
        display.to_markdown(index=False) if not display.empty else "No completed runs.",
        "",
        "## Formula",
        "",
        "`logits = Linear(z_classical) + gate([z_classical, x_spectral]) * QNN(x_spectral)`",
        "",
        "The splits and encoder checkpoints are identical to the HybridSN-small baseline.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HybridSN-small + Spectral QNN Gated Fusion few-shot on Indian Pines.")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
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
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--qnn_layers", type=int, default=1)
    parser.add_argument("--entanglement", default="linear")
    parser.add_argument("--gate_mode", choices=["scalar", "classwise"], default="scalar")
    parser.add_argument("--backend", default="lightning.qubit")
    parser.add_argument("--diff_method", default="adjoint")
    parser.add_argument("--angle_scale", type=float, default=float(np.pi))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--monitor", choices=["macro_f1", "oa"], default="macro_f1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_dir", default="result/hybridsn_small_fewshot_3datasets/split_indices")
    parser.add_argument("--encoder_checkpoint_dir", default="result/hybridsn_small_fewshot_3datasets/checkpoints")
    parser.add_argument("--baseline_summary", default="result/hybridsn_small_fewshot_indian_pines_only/metrics/summary_by_shot.csv")
    parser.add_argument("--output_dir", default="result/hybridsn_small_spectral_qnn_gated_fewshot_indian_pines")
    parser.add_argument("--rebuild_features", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    main()
