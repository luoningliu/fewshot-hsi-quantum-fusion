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
from src.datasets.fewshot_sampler import make_fewshot_split, save_split
from src.datasets.hsi_dataset import load_hsi_mat
from src.models.classical import CNN1D, CNN2D, CNN3D
from src.models.traditional import build_knn, build_random_forest, build_svm_rbf
from src.utils.seed import set_seed


TRADITIONAL_BUILDERS = {
    "svm_rbf": build_svm_rbf,
    "random_forest": build_random_forest,
    "knn": build_knn,
}
DEEP_BUILDERS = {
    "cnn1d": lambda bands, classes: CNN1D(bands, classes),
    "cnn2d": lambda bands, classes: CNN2D(bands, classes),
    "cnn3d": lambda bands, classes: CNN3D(bands, classes),
}


class SpectralVectorDataset(Dataset):
    def __init__(self, cube: np.ndarray, rows: np.ndarray, cols: np.ndarray, labels: np.ndarray, indices: list[int]):
        self.cube = cube
        self.rows = rows
        self.cols = cols
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        return (
            torch.from_numpy(self.cube[int(self.rows[idx]), int(self.cols[idx])]).float(),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )


def main() -> None:
    args = _build_parser().parse_args()
    out = Path(args.output_dir)
    for sub in ("metrics", "logs", "checkpoints", "confusion_matrices", "split_indices"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
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
        radius = args.patch_size // 2
        padded = np.pad(cube_pca, ((radius, radius), (radius, radius), (0, 0)), mode="reflect").astype(np.float32)
        vectors = cube_pca[rows, cols].astype(np.float32)

        for shot in args.shots:
            for seed in args.seeds:
                split = make_fewshot_split(labels, int(shot), int(seed), num_classes)
                if split["skipped"]:
                    failures.append({"dataset": dataset_name, "shot": int(shot), "seed": int(seed), "error": f"skipped {split['skipped']}"})
                    continue
                save_split(out / "split_indices" / f"{dataset_name}_seed{seed}_{shot}shot.json", split)
                for model_name in args.models:
                    try:
                        if model_name in TRADITIONAL_BUILDERS:
                            result = _run_traditional(args, out, model_name, data_cfg, vectors, labels, split, dataset_name, shot, seed, pca_evr_sum)
                        else:
                            result = _run_deep(args, out, model_name, data_cfg, cube_pca, padded, rows, cols, labels, split, dataset_name, shot, seed, pca_evr_sum, num_classes, device)
                        all_rows.append(result)
                        _write_summary(out, all_rows)
                    except Exception as exc:
                        failure = {"dataset": dataset_name, "shot": int(shot), "seed": int(seed), "model": model_name, "error": repr(exc)}
                        failures.append(failure)
                        print(f"[WARN] skipped {failure}")
                    write_json(out / "failed_runs.json", {"failed_runs": failures})
    _write_report(out, all_rows, failures)
    print(f"Result directory: {out}")


def _run_traditional(args, out, model_name, data_cfg, vectors, labels, split, dataset_name, shot, seed, pca_evr_sum):
    started = time.time()
    model = TRADITIONAL_BUILDERS[model_name]()
    train_idx = np.asarray(split["train"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)
    model.fit(vectors[train_idx], labels[train_idx])
    train_time = time.time() - started
    test_started = time.time()
    pred = model.predict(vectors[test_idx])
    test_time = time.time() - test_started
    y_test = labels[test_idx]
    metrics = classification_metrics(y_test, pred, labels=list(range(int(data_cfg["num_classes"]))))
    _save_predictions(out, data_cfg, dataset_name, model_name, shot, seed, y_test, pred)
    row = _row(dataset_name, model_name, shot, seed, metrics, 0, train_time, test_time, None, split)
    write_json(out / "metrics" / f"{dataset_name}_{model_name}_shot{shot}_seed{seed}.json", {
        **row,
        "pca_evr_sum": pca_evr_sum,
        "pca_fit_scope": "full_image_unsupervised",
        "input": "center_pca_spectral_vector",
    })
    print(f"{dataset_name} {model_name} {shot}-shot seed{seed}: OA={metrics['OA']*100:.2f} Macro-F1={metrics['Macro-F1']*100:.2f}")
    return row


def _run_deep(args, out, model_name, data_cfg, cube_pca, padded, rows, cols, labels, split, dataset_name, shot, seed, pca_evr_sum, num_classes, device):
    set_seed(int(seed))
    model = DEEP_BUILDERS[model_name](args.pca_bands, num_classes).to(device)
    train_ds = _deep_dataset(model_name, cube_pca, padded, rows, cols, labels, split["train"], args.patch_size)
    val_ds = _deep_dataset(model_name, cube_pca, padded, rows, cols, labels, split["validation"], args.patch_size)
    test_ds = _deep_dataset(model_name, cube_pca, padded, rows, cols, labels, split["test"], args.patch_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0
    best_epoch = stale = 0
    logs = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_val, p_val = _evaluate(model, val_loader, criterion, device)
        val = classification_metrics(y_val, p_val, labels=list(range(num_classes)))
        logs.append({
            "dataset": dataset_name, "model": model_name, "shot": shot, "seed": seed, "epoch": epoch,
            "train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss,
            "val_OA": val["OA"], "val_AA": val["AA"], "val_Kappa": val["Kappa"],
            "val_Macro-F1": val["Macro-F1"], "val_Weighted-F1": val["Weighted-F1"],
        })
        monitor = val["OA"] if args.monitor == "oa" else val["Macro-F1"]
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
    torch.save(best_state, out / "checkpoints" / f"{dataset_name}_{model_name}_shot{shot}_seed{seed}.pt")
    pd.DataFrame(logs).to_csv(out / "logs" / f"{dataset_name}_{model_name}_shot{shot}_seed{seed}.csv", index=False)
    test_started = time.time()
    _, y_test, pred = _evaluate(model, test_loader, criterion, device)
    test_time = time.time() - test_started
    metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
    _save_predictions(out, data_cfg, dataset_name, model_name, shot, seed, y_test, pred)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    row = _row(dataset_name, model_name, shot, seed, metrics, best_epoch, train_time, test_time, params, split)
    write_json(out / "metrics" / f"{dataset_name}_{model_name}_shot{shot}_seed{seed}.json", {
        **row,
        "pca_evr_sum": pca_evr_sum,
        "pca_fit_scope": "full_image_unsupervised",
        "patch_size": args.patch_size if model_name != "cnn1d" else None,
        "pca_bands": args.pca_bands,
    })
    print(f"{dataset_name} {model_name} {shot}-shot seed{seed}: OA={metrics['OA']*100:.2f} Macro-F1={metrics['Macro-F1']*100:.2f} best_epoch={best_epoch}")
    return row


def _deep_dataset(model_name, cube_pca, padded, rows, cols, labels, indices, patch_size):
    if model_name == "cnn1d":
        return SpectralVectorDataset(cube_pca, rows, cols, labels, indices)
    return OnTheFlyPatchDataset(padded, rows, cols, labels, indices, patch_size)


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = correct = count = 0
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


def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = count = 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            yy = y.to(device)
            logits = model(x)
            loss_sum += float(criterion(logits, yy).item()) * len(y)
            count += len(y)
            ys.append(y.numpy())
            ps.append(logits.argmax(1).cpu().numpy())
    return loss_sum / max(count, 1), np.concatenate(ys), np.concatenate(ps)


def _save_predictions(out, data_cfg, dataset_name, model_name, shot, seed, y_true, y_pred):
    num_classes = int(data_cfg["num_classes"])
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    stem = f"{dataset_name}_{model_name}_shot{shot}_seed{seed}"
    per_class_metrics(y_true, y_pred, class_names).to_csv(out / "metrics" / f"{stem}_per_class.csv", index=False)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(out / "confusion_matrices" / f"{stem}.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(out / "confusion_matrices" / f"{stem}_normalized.csv", index=False)


def _row(dataset, model, shot, seed, metrics, best_epoch, train_time, test_time, params, split):
    return {
        "dataset": dataset,
        "model": model,
        "shot": int(shot),
        "seed": int(seed),
        **metrics,
        "best_epoch": int(best_epoch),
        "train_time_seconds": float(train_time),
        "test_time_seconds": float(test_time),
        "trainable_parameters": params,
        "train_size": len(split["train"]),
        "validation_size": len(split["validation"]),
        "test_size": len(split["test"]),
    }


def _write_summary(out: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics" / "all_runs_other_baselines.csv", index=False)
    summary_rows = []
    for (dataset, model, shot), group in df.groupby(["dataset", "model", "shot"]):
        row = {"dataset": dataset, "model": model, "shot": int(shot), "runs": len(group)}
        for metric in ("OA", "AA", "Kappa", "Macro-F1", "Weighted-F1"):
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        row["mean_best_epoch"] = float(group["best_epoch"].mean())
        row["trainable_parameters"] = group["trainable_parameters"].dropna().iloc[0] if group["trainable_parameters"].notna().any() else ""
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "shot", "model"])
    summary.to_csv(out / "metrics" / "summary_by_dataset_model_shot.csv", index=False)
    display = summary.copy()
    for col in display.columns:
        if col.startswith("mean_") or col.startswith("std_"):
            if col != "mean_best_epoch":
                display[col] = (display[col] * 100).round(2)
            else:
                display[col] = display[col].round(2)
    (out / "metrics" / "summary_by_dataset_model_shot.md").write_text(display.to_markdown(index=False) + "\n", encoding="utf-8")


def _write_report(out: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]]):
    summary = out / "metrics" / "summary_by_dataset_model_shot.md"
    text = summary.read_text(encoding="utf-8") if summary.exists() else "No completed runs."
    lines = [
        "# Other Baselines Few-shot HSI Classification",
        "",
        "All models use the same all-way few-shot sampler as HybridSN-small.",
        "",
        "- Traditional baselines use center PCA spectral vectors.",
        "- CNN1D uses center PCA spectral vectors.",
        "- CNN2D and CNN3D use PCA spatial-spectral patches.",
        "- PCA is fitted on the full image without labels to match the current HybridSN-small few-shot protocol.",
        "",
        "## Summary",
        "",
        text,
        "",
        "## Failed Runs",
        "",
        pd.DataFrame(failures).to_markdown(index=False) if failures else "None.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_parser():
    parser = argparse.ArgumentParser(description="Run non-HybridSN baselines under the all-way few-shot HSI protocol.")
    parser.add_argument("--datasets", nargs="+", choices=["indian_pines", "pavia_university", "salinas"], default=["indian_pines", "pavia_university", "salinas"])
    parser.add_argument("--models", nargs="+", choices=sorted([*TRADITIONAL_BUILDERS, *DEEP_BUILDERS]), default=["svm_rbf", "random_forest", "knn", "cnn1d", "cnn2d", "cnn3d"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--patch_size", type=int, default=19)
    parser.add_argument("--pca_bands", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--monitor", choices=["macro_f1", "oa"], default="macro_f1")
    parser.add_argument("--seed", type=int, default=42, help="PCA preprocessing seed.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="result/other_baselines_fewshot_3datasets")
    return parser


if __name__ == "__main__":
    main()
