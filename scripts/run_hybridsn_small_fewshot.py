from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.datasets.fewshot_sampler import make_fewshot_split, save_split
from src.datasets.hsi_dataset import load_hsi_mat
from src.models.classical import HybridSNSmall
from src.utils.config import load_yaml
from src.utils.seed import set_seed


DATASET_CONFIGS = {
    "indian_pines": "configs/data/indian_pines.yaml",
    "pavia_university": "configs/data/pavia_university.yaml",
    "salinas": "configs/data/salinas.yaml",
}


class OnTheFlyPatchDataset(Dataset):
    def __init__(
        self,
        padded_cube: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        labels: np.ndarray,
        indices: list[int],
        patch_size: int,
    ):
        self.padded_cube = padded_cube
        self.rows = rows
        self.cols = cols
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.patch_size = int(patch_size)
        self.radius = self.patch_size // 2

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        row = int(self.rows[idx]) + self.radius
        col = int(self.cols[idx]) + self.radius
        patch = self.padded_cube[
            row - self.radius : row + self.radius + 1,
            col - self.radius : col + self.radius + 1,
            :,
        ]
        return torch.from_numpy(patch).float(), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.datasets is None:
        args.datasets = [args.dataset] if args.dataset else ["indian_pines"]
    run_root = _make_output_root(args)
    _write_config(run_root / "config.yaml", vars(args))
    device = _resolve_device(args.device)
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        data_cfg = _load_dataset_config(dataset_name, args.data_root)
        raw = load_hsi_mat(data_cfg)
        rows, cols = np.nonzero(raw.gt != raw.background_label)
        labels = raw.gt[rows, cols].astype(np.int64) - 1
        num_classes = int(data_cfg["num_classes"])
        assert labels.min() >= 0
        assert labels.max() < num_classes

        cube_pca, pca_evr_sum = _preprocess_full_image(raw.cube, args.pca_bands, args.seed)
        padded_cube = np.pad(
            cube_pca,
            ((args.patch_size // 2, args.patch_size // 2), (args.patch_size // 2, args.patch_size // 2), (0, 0)),
            mode="reflect",
        ).astype(np.float32)
        for shot in args.shots:
            for seed in args.seeds:
                try:
                    row = _run_one(
                        args=args,
                        run_root=run_root,
                        dataset_name=dataset_name,
                        display_name=raw.display_name,
                        data_cfg=data_cfg,
                        padded_cube=padded_cube,
                        rows=rows,
                        cols=cols,
                        labels=labels,
                        num_classes=num_classes,
                        shot=int(shot),
                        seed=int(seed),
                        pca_evr_sum=pca_evr_sum,
                        device=device,
                    )
                    all_rows.append(row)
                except Exception as exc:
                    failure = {"dataset": dataset_name, "shot": int(shot), "seed": int(seed), "error": repr(exc)}
                    failures.append(failure)
                    print(f"[WARN] skipped {failure}")
                _write_summary(run_root, all_rows)
                write_json(run_root / "failed_runs.json", {"failed_runs": failures})
    _write_report(run_root, args, all_rows, failures)
    print(f"Result directory: {run_root}")


def _run_one(
    args: argparse.Namespace,
    run_root: Path,
    dataset_name: str,
    display_name: str,
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
    split = make_fewshot_split(labels, shot=shot, seed=seed, num_classes=num_classes)
    if split["skipped"]:
        raise ValueError(f"classes with insufficient samples: {split['skipped']}")
    for cls, counts in split["per_class"].items():
        if counts["train"] != shot:
            raise AssertionError(f"class {cls} train count {counts['train']} != shot {shot}")
    split_path = run_root / "split_indices" / f"{dataset_name}_seed{seed}_{shot}shot.json"
    save_split(split_path, split)

    model = HybridSNSmall(
        pca_channels=args.pca_bands,
        num_classes=num_classes,
        patch_size=args.patch_size,
        conv3d_channels=tuple(args.conv3d_channels),
        conv2d_channels=args.conv2d_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.debug:
        dummy = torch.zeros(2, args.patch_size, args.patch_size, args.pca_bands, device=device)
        logits = model(dummy)
        assert logits.shape == (2, num_classes)

    train_ds = OnTheFlyPatchDataset(padded_cube, rows, cols, labels, split["train"], args.patch_size)
    val_ds = OnTheFlyPatchDataset(padded_cube, rows, cols, labels, split["validation"], args.patch_size)
    test_ds = OnTheFlyPatchDataset(padded_cube, rows, cols, labels, split["test"], args.patch_size)
    train_loader = _make_train_loader(train_ds, labels, split["train"], args.batch_size, args.class_balanced_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = _make_criterion(labels, split["train"], num_classes, args.class_weight, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0
    best_epoch = 0
    stale = 0
    logs = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_val, pred_val = _evaluate(model, val_loader, criterion, device)
        val_metrics = classification_metrics(y_val, pred_val, labels=list(range(num_classes)))
        log_row = {
            "dataset": dataset_name,
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
        logs.append(log_row)
        monitor = val_metrics["Macro-F1"] if args.monitor == "macro_f1" else val_metrics["OA"]
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
    ckpt_path = run_root / "checkpoints" / f"{dataset_name}_shot{shot}_seed{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_path)
    test_started = time.time()
    test_loss, y_test, pred_test = _evaluate(model, test_loader, criterion, device)
    test_time = time.time() - test_started
    metrics = classification_metrics(y_test, pred_test, labels=list(range(num_classes)))
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class = per_class_metrics(y_test, pred_test, class_names)
    per_class["accuracy"] = per_class["recall"]
    cm = confusion_matrix(y_test, pred_test, labels=list(range(num_classes)))

    logs_dir = run_root / "logs"
    metrics_dir = run_root / "metrics"
    cm_dir = run_root / "confusion_matrices"
    for path in (logs_dir, metrics_dir, cm_dir):
        path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(logs_dir / f"{dataset_name}_shot{shot}_seed{seed}.csv", index=False)
    per_class.to_csv(metrics_dir / f"{dataset_name}_shot{shot}_seed{seed}_per_class.csv", index=False)
    pd.DataFrame(cm).to_csv(cm_dir / f"{dataset_name}_shot{shot}_seed{seed}.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(
        cm_dir / f"{dataset_name}_shot{shot}_seed{seed}_normalized.csv",
        index=False,
    )
    payload = {
        "dataset": dataset_name,
        "display_name": display_name,
        "shot": shot,
        "seed": seed,
        "num_classes": num_classes,
        "class_order": list(range(num_classes)),
        "class_names": class_names,
        "pca_fit_scope": "full_image_unsupervised",
        "pca_evr_sum": pca_evr_sum,
        "patch_size": args.patch_size,
        "pca_bands": args.pca_bands,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "train_time_seconds": train_time,
        "test_time_seconds": test_time,
        "trainable_parameters": param_count,
        "checkpoint": str(ckpt_path),
        **metrics,
    }
    write_json(metrics_dir / f"{dataset_name}_shot{shot}_seed{seed}.json", payload)
    row = {
        "dataset": dataset_name,
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
        "train_size": len(split["train"]),
        "validation_size": len(split["validation"]),
        "test_size": len(split["test"]),
    }
    print(
        f"{dataset_name} {shot}-shot seed{seed}: "
        f"OA={metrics['OA']*100:.2f} Macro-F1={metrics['Macro-F1']*100:.2f} best_epoch={best_epoch}"
    )
    return row


def _preprocess_full_image(cube: np.ndarray, pca_bands: int, seed: int) -> tuple[np.ndarray, float]:
    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0, dtype=np.float64)
    std = flat.std(axis=0, dtype=np.float64) + 1e-6
    norm = ((cube - mean) / std).astype(np.float32)
    pca = PCA(n_components=int(pca_bands), whiten=False, random_state=int(seed))
    reduced = pca.fit_transform(norm.reshape(-1, norm.shape[-1])).astype(np.float32)
    return reduced.reshape(cube.shape[0], cube.shape[1], int(pca_bands)), float(pca.explained_variance_ratio_.sum())


def _train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
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
        for x, y in loader:
            x = x.to(device)
            yy = y.to(device)
            logits = model(x)
            loss = criterion(logits, yy)
            loss_sum += float(loss.item()) * len(y)
            count += len(y)
            ys.append(y.numpy())
            ps.append(logits.argmax(1).cpu().numpy())
    return loss_sum / max(count, 1), np.concatenate(ys), np.concatenate(ps)


def _make_train_loader(ds: Dataset, labels: np.ndarray, indices: list[int], batch_size: int, enabled: bool) -> DataLoader:
    if not enabled:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    y = labels[np.asarray(indices, dtype=np.int64)]
    counts = np.bincount(y, minlength=int(y.max()) + 1)
    weights = 1.0 / np.maximum(counts[y], 1)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)


def _make_criterion(labels: np.ndarray, train_indices: list[int], num_classes: int, mode: str, device: torch.device) -> nn.Module:
    if mode == "none":
        return nn.CrossEntropyLoss()
    if mode != "balanced":
        raise ValueError(f"Unsupported class_weight: {mode}")
    y = labels[np.asarray(train_indices, dtype=np.int64)]
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = len(y) / (num_classes * np.maximum(counts, 1.0))
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def _write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "all_runs.csv", index=False)
    summary_rows = []
    for (dataset, shot), group in df.groupby(["dataset", "shot"]):
        row = {"dataset": dataset, "shot": shot, "runs": len(group)}
        for metric in ("OA", "AA", "Kappa", "Macro-F1", "Weighted-F1"):
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        row["mean_best_epoch"] = float(group["best_epoch"].mean())
        row["trainable_parameters"] = int(group["trainable_parameters"].iloc[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "shot"])
    summary.to_csv(metrics_dir / "summary_by_dataset_shot.csv", index=False)
    display = summary.copy()
    for col in display.columns:
        if col.startswith("mean_") or col.startswith("std_"):
            if col != "mean_best_epoch":
                display[col] = (display[col] * 100).round(2)
    (metrics_dir / "summary_by_dataset_shot.md").write_text(display.to_markdown(index=False) + "\n", encoding="utf-8")


def _write_report(root: Path, args: argparse.Namespace, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    summary_path = root / "metrics" / "summary_by_dataset_shot.md"
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else "No completed runs."
    lines = [
        "# HybridSN-small Few-shot HSI Classification",
        "",
        "## Purpose",
        "",
        "Evaluate a lightweight HybridSN-style 3D-2D CNN as a classical few-shot baseline for hyperspectral image classification.",
        "",
        "## Architecture",
        "",
        "HybridSN-small keeps the HybridSN hierarchy: three 3D convolution blocks for spectral-spatial features, reshape spectral depth into channels, one 2D convolution block for spatial abstraction, global average pooling, and a small MLP classifier.",
        "",
        "The large original Flatten-Dense classifier is replaced by Global Average Pooling plus Linear(32 -> 64 -> classes), sharply reducing dense parameters.",
        "",
        "## Few-shot Protocol",
        "",
        "For each dataset, shot, and seed, the support set contains exactly K labeled samples per class. Validation uses min(10 samples per class, 20% of remaining samples per class), and all remaining labeled pixels are used for testing. Splits are stratified and saved under `split_indices/`.",
        "",
        "PCA is fitted on the full image without labels because strict 1-shot support sets cannot fit 20/30 PCA components. This is an unsupervised preprocessing step and is documented as `pca_fit_scope=full_image_unsupervised` in each metrics JSON.",
        "",
        "## Hyperparameters",
        "",
        f"- datasets: {args.datasets}",
        f"- shots: {args.shots}",
        f"- seeds: {args.seeds}",
        f"- patch_size: {args.patch_size}",
        f"- pca_bands: {args.pca_bands}",
        f"- dropout: {args.dropout}",
        f"- lr: {args.lr}",
        f"- weight_decay: {args.weight_decay}",
        f"- epochs: {args.epochs}",
        f"- patience: {args.patience}",
        "",
        "## Aggregated Results",
        "",
        summary_text,
        "",
        "## Failed or Skipped Runs",
        "",
        pd.DataFrame(failures).to_markdown(index=False) if failures else "None.",
        "",
        "## Initial Interpretation",
        "",
        "HybridSN-small preserves the key HybridSN design: 3D spectral-spatial feature extraction followed by 2D spatial feature learning. Compared with the original HybridSN, this version replaces the large Flatten-Dense classifier with Global Average Pooling and a small classifier, making it more suitable for strict few-shot settings.",
        "",
        "The 1-shot results should be interpreted with caution because the variance can be high. The 5-shot and 10-shot settings are more informative for evaluating whether a lightweight spectral-spatial CNN can learn stable representations under limited labels.",
        "",
        "This experiment should serve as the classical lightweight baseline for later comparison with HybridSN-small + classical bottleneck, HybridSN-small + quantum bottleneck, and quantum prototype network.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_dataset_config(dataset_name: str, data_root: str) -> dict[str, Any]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset {dataset_name}. Available: {sorted(DATASET_CONFIGS)}")
    cfg = load_yaml(DATASET_CONFIGS[dataset_name])
    if data_root:
        root = Path(data_root)
        for key in ("data_path", "gt_path"):
            original = Path(cfg[key])
            cfg[key] = str(root / original.relative_to("data")) if original.parts and original.parts[0] == "data" else str(original)
    return cfg


def _make_output_root(args: argparse.Namespace) -> Path:
    if args.run_dir:
        root = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(args.output_root) / f"{stamp}_hybridsn_small_fewshot"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    serializable = {key: value for key, value in payload.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HybridSN-small few-shot HSI classification.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), help="Single dataset to run.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_CONFIGS), help="Datasets to run.")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--patch_size", type=int, default=19)
    parser.add_argument("--pca_bands", type=int, default=30)
    parser.add_argument("--conv3d_channels", nargs=3, type=int, default=[8, 16, 16])
    parser.add_argument("--conv2d_channels", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--class_balanced_sampler", action="store_true")
    parser.add_argument("--monitor", choices=["macro_f1", "oa"], default="macro_f1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_root", default="result")
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--seed", type=int, default=42, help="Preprocessing PCA seed.")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    main()
