from __future__ import annotations

import argparse
import copy
import json
import math
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

from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.models.quantum import GatedResidualQNNClassifier, ResidualQNNClassifier
from src.utils.config import load_yaml
from src.utils.seed import set_seed


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, indices: list[int]):
        self.embeddings = embeddings
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        return torch.from_numpy(self.embeddings[idx]).float(), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-label head regularization on frozen HybridSN embeddings.")
    parser.add_argument("--config", default="configs/experiments/low_label_hybridsn_head_regularization_indian_pines.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output"]["root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(_to_yaml(cfg), encoding="utf-8")

    data_cfg = load_yaml(cfg["dataset"]["config"])
    split = _load_split(data_cfg["output"]["split_json"])
    data = np.load(cfg["embedding_path"])
    embeddings = data["z"].astype(np.float32)
    labels = data["y"].astype(np.int64)
    num_classes = int(data_cfg["num_classes"])
    assert labels.min() >= 0
    assert labels.max() < num_classes
    assert num_classes == 16

    device = _resolve_device(str(cfg["training"].get("device", "auto")))
    debug_lines = [
        f"device: {device}",
        f"embedding_path: {cfg['embedding_path']}",
        f"embedding shape: {embeddings.shape}",
        f"base train size: {len(split['train'])}",
    ]

    all_runs: list[dict[str, Any]] = []
    all_logs: list[dict[str, Any]] = []
    best_state = None
    best_key = None
    best_spec = None
    best_run = None
    best_fraction = None

    for fraction in cfg["training"]["train_fractions"]:
        for seed in cfg["training"]["seeds"]:
            subset = _stratified_low_label_subset(labels, split["train"], float(fraction), seed=int(seed))
            debug_lines.append(f"fraction={fraction} seed={seed} train_subset={len(subset)}")
            for spec in cfg["heads"]:
                run_id = f"{spec['model']}_f{fraction}_seed{seed}"
                set_seed(int(seed))
                model = _make_head(spec, embeddings.shape[1], num_classes).to(device)
                row, logs, state = _fit_one(
                    run_id,
                    model,
                    embeddings,
                    labels,
                    subset,
                    split["validation"],
                    num_classes,
                    cfg,
                    spec,
                    float(fraction),
                    int(seed),
                    device,
                    debug_lines,
                )
                all_runs.append(row)
                all_logs.extend(logs)
                pd.DataFrame(all_runs).to_csv(out_dir / "all_runs.csv", index=False)
                pd.DataFrame(all_logs).to_csv(out_dir / "training_log.csv", index=False)
                score = (row["best_val_macro_f1"], row["best_val_aa"])
                if best_key is None or score > best_key:
                    best_key = score
                    best_state = state
                    best_spec = spec
                    best_run = row
                    best_fraction = float(fraction)

    summary = _summarize(all_runs)
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)
    model = _make_head(best_spec, embeddings.shape[1], num_classes).to(device)
    model.load_state_dict(best_state)
    y_test, pred = _predict(model, embeddings, labels, split["test"], int(cfg["training"]["batch_size"]), device)
    metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
    best_metrics = {
        "dataset_id": data_cfg["dataset_id"],
        "model": best_spec["model"],
        "train_fraction": best_fraction,
        "selected_by": "validation_Macro-F1",
        "source_encoder": "frozen tuned HybridSN encoder",
        **metrics,
        "best_run": best_run,
    }
    write_json(out_dir / "best_metrics.json", best_metrics)
    torch.save(best_state, out_dir / "best_head_state.pt")
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class_metrics(y_test, pred, class_names).to_csv(out_dir / "per_class_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(
        out_dir / "normalized_confusion_matrix.csv",
        index=False,
    )
    _write_model_summary(out_dir / "model_summary.txt", model)
    (out_dir / "debug_shapes.txt").write_text("\n".join(debug_lines) + "\n", encoding="utf-8")
    _write_report(out_dir / "low_label_report.md", summary, best_metrics)


def _stratified_low_label_subset(labels: np.ndarray, train_indices: list[int], fraction: float, seed: int) -> list[int]:
    rng = np.random.default_rng(seed + int(fraction * 10000))
    train = np.asarray(train_indices, dtype=np.int64)
    target = max(16, int(round(len(train) * fraction)))
    selected: list[int] = []
    classes = np.unique(labels[train])
    # Include at least one available sample per class, then fill by stratified quota.
    for label in classes:
        cls = train[labels[train] == label]
        selected.extend(rng.choice(cls, size=1, replace=False).astype(int).tolist())
    remaining = max(0, target - len(selected))
    if remaining > 0:
        weights = np.bincount(labels[train], minlength=int(labels.max()) + 1).astype(np.float64)
        weights = weights / weights.sum()
        extra_counts = np.floor(weights * remaining).astype(int)
        while extra_counts.sum() < remaining:
            extra_counts[int(rng.choice(np.arange(len(extra_counts)), p=weights))] += 1
        for label, count in enumerate(extra_counts):
            if count <= 0:
                continue
            cls = np.asarray(sorted(set(train[labels[train] == label]) - set(selected)), dtype=np.int64)
            if len(cls) == 0:
                continue
            selected.extend(rng.choice(cls, size=min(count, len(cls)), replace=False).astype(int).tolist())
    rng.shuffle(selected)
    return selected[:target]


def _make_head(spec: dict[str, Any], input_dim: int, num_classes: int) -> nn.Module:
    if spec["model"] == "linear_probe":
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, num_classes))
    if spec["model"] == "mlp_probe":
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(spec["hidden_dim"])),
            nn.ReLU(inplace=True),
            nn.Dropout(float(spec["dropout"])),
            nn.Linear(int(spec["hidden_dim"]), num_classes),
        )
    qnn_kwargs = {
        "qubits": int(spec["qubits"]),
        "layers": int(spec["layers"]),
        "entanglement": str(spec["entanglement"]),
        "backend": "lightning.qubit",
        "diff_method": "adjoint",
        "normalize_input": True,
        "angle_scale": float(spec.get("angle_scale", math.pi)),
    }
    if spec["model"] == "residual_qnn":
        return ResidualQNNClassifier(input_dim=input_dim, num_classes=num_classes, **qnn_kwargs)
    if spec["model"] == "gated_residual_qnn":
        return GatedResidualQNNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            gate_mode=str(spec.get("gate_mode", "scalar")),
            **qnn_kwargs,
        )
    raise ValueError(f"Unsupported head: {spec['model']}")


def _fit_one(
    run_id: str,
    model: nn.Module,
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: list[int],
    val_indices: list[int],
    num_classes: int,
    cfg: dict[str, Any],
    spec: dict[str, Any],
    fraction: float,
    seed: int,
    device: torch.device,
    debug_lines: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor]]:
    train_loader = _loader(embeddings, labels, train_indices, int(cfg["training"]["batch_size"]), True)
    val_loader = _loader(embeddings, labels, val_indices, int(cfg["training"]["batch_size"]), False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    logs: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if epoch == 1 and batch_idx == 0:
                grad_norm = _grad_norm(model)
                debug_lines.append(f"{run_id}: x={tuple(x.shape)} logits={tuple(logits.shape)} grad={grad_norm:.6f}")
                assert grad_norm > 0 and not math.isnan(grad_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            count += len(y)
        val_loss, y_val, pred_val = _evaluate(model, val_loader, criterion, device)
        metrics = classification_metrics(y_val, pred_val, labels=list(range(num_classes)))
        log = {
            "run_id": run_id,
            "model": spec["model"],
            "train_fraction": fraction,
            "seed": seed,
            "epoch": epoch,
            "train_loss": total_loss / max(count, 1),
            "train_accuracy": correct / max(count, 1),
            "validation_loss": val_loss,
            "validation_OA": metrics["OA"],
            "validation_AA": metrics["AA"],
            "validation_Kappa": metrics["Kappa"],
            "validation_Macro-F1": metrics["Macro-F1"],
            "validation_Weighted-F1": metrics["Weighted-F1"],
        }
        logs.append(log)
        if metrics["Macro-F1"] > best_metric:
            best_metric = metrics["Macro-F1"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(cfg["training"]["patience"]):
            break
    best_log = max(logs, key=lambda x: x["validation_Macro-F1"])
    return {
        "run_id": run_id,
        "model": spec["model"],
        "train_fraction": fraction,
        "train_size": len(train_indices),
        "seed": seed,
        "best_val_macro_f1": best_log["validation_Macro-F1"],
        "best_val_oa": best_log["validation_OA"],
        "best_val_aa": best_log["validation_AA"],
        "epochs_ran": len(logs),
        "training_time_seconds": time.time() - started,
    }, logs, best_state


def _summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for (fraction, model), group in df.groupby(["train_fraction", "model"]):
        item = {"train_fraction": fraction, "model": model, "runs": len(group), "train_size_mean": float(group["train_size"].mean())}
        for col in ("best_val_oa", "best_val_aa", "best_val_macro_f1"):
            item[f"{col}_mean"] = float(group[col].mean())
            item[f"{col}_std"] = float(group[col].std(ddof=0))
        out.append(item)
    return pd.DataFrame(out).sort_values(["train_fraction", "best_val_macro_f1_mean"], ascending=[True, False])


def _predict(model: nn.Module, embeddings: np.ndarray, labels: np.ndarray, indices: list[int], batch_size: int, device: torch.device):
    loader = _loader(embeddings, labels, indices, batch_size, False)
    _, y_true, y_pred = _evaluate(model, loader, nn.CrossEntropyLoss(), device)
    return y_true, y_pred


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total = 0.0
    count = 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            yy = y.to(device)
            logits = model(x)
            total += float(criterion(logits, yy).item()) * len(y)
            count += len(y)
            ys.append(y.numpy())
            ps.append(logits.argmax(1).cpu().numpy())
    return total / max(count, 1), np.concatenate(ys), np.concatenate(ps)


def _loader(embeddings: np.ndarray, labels: np.ndarray, indices: list[int], batch_size: int, shuffle: bool):
    return DataLoader(EmbeddingDataset(embeddings, labels, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _load_split(path: str | Path) -> dict[str, list[int]]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {key: raw[key] for key in ("train", "validation", "test")}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _grad_norm(model: nn.Module) -> float:
    return sum(float(p.grad.detach().norm().item()) for p in model.parameters() if p.grad is not None)


def _write_model_summary(path: Path, model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    path.write_text(f"{model}\n\nTotal parameters: {total}\nTrainable parameters: {trainable}\n", encoding="utf-8")


def _write_report(path: Path, summary: pd.DataFrame, best_metrics: dict[str, Any]) -> None:
    display = summary.copy()
    for col in display.columns:
        if (col.endswith("_mean") or col.endswith("_std")) and col != "train_size_mean":
            display[col] = (display[col] * 100).round(2)
    lines = [
        "# Low-Label HybridSN Head Regularization: Indian Pines",
        "",
        "Frozen tuned HybridSN embeddings are reused. Only a stratified fraction of the original training split is used for head training; validation/test splits are unchanged.",
        "",
        "## Validation Mean ± Std",
        "",
        display.to_markdown(index=False),
        "",
        "## Best Selected Test Result",
        "",
        "| Model | Fraction | OA | AA | Kappa | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {best_metrics['model']} | {best_metrics['train_fraction']:.2f} | {best_metrics['OA']*100:.2f} | {best_metrics['AA']*100:.2f} | {best_metrics['Kappa']*100:.2f} | {best_metrics['Macro-F1']*100:.2f} | {best_metrics['Weighted-F1']*100:.2f} |",
        "",
        "## Interpretation",
        "",
        "Use this table to test whether QNN heads act as useful regularizers when very few labeled samples are available for the classifier head.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_yaml(payload: dict[str, Any]) -> str:
    lines = []
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
