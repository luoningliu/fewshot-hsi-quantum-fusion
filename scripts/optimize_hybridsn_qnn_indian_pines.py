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

from scripts.tune_hybridsn_indian_pines import _build_inputs
from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.datasets.hsi_dataset import load_hsi_mat
from src.models.classical import HybridSN
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
    parser = argparse.ArgumentParser(description="Optimize QNN heads on tuned HybridSN embeddings.")
    parser.add_argument("--config", default="configs/experiments/hybridsn_qnn_optimized_indian_pines.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    output_dir = Path(cfg["output"]["root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(_to_yaml(cfg), encoding="utf-8")

    data_cfg = load_yaml(cfg["dataset"]["config"])
    raw = load_hsi_mat(data_cfg)
    split = _load_split(data_cfg["output"]["split_json"])
    rows, cols = np.nonzero(raw.gt != raw.background_label)
    labels = raw.gt[rows, cols].astype(np.int64) - 1
    num_classes = int(data_cfg["num_classes"])
    assert labels.min() >= 0
    assert labels.max() < num_classes
    assert num_classes == 16

    set_seed(int(cfg["seed"]))
    device = _resolve_device(str(cfg["training"].get("device", "auto")))
    best_hybrid = json.load(open(cfg["hybridsn"]["metrics"], "r", encoding="utf-8"))
    hcfg = best_hybrid["best_config"]
    inputs = _build_inputs(
        cube=raw.cube,
        rows=rows,
        cols=cols,
        split=split,
        patch_size=int(hcfg["patch_size"]),
        pca_components=int(hcfg["pca_components"]),
        seed=int(hcfg["seed"]),
    )
    embeddings = _extract_embeddings(cfg, hcfg, inputs["patches"], labels, device)
    np.savez_compressed(output_dir / "hybridsn_embeddings_tuned.npz", z=embeddings, y=labels)

    debug_lines = [
        f"device: {device}",
        f"HybridSN checkpoint: {cfg['hybridsn']['checkpoint']}",
        f"patch shape: {inputs['patches'].shape}",
        f"embedding shape: {embeddings.shape}",
        f"label min/max: {labels.min()}/{labels.max()}",
    ]

    all_runs = []
    all_logs = []
    best_state = None
    best_row = None
    best_model_spec = None
    for i, spec in enumerate(cfg["qnn_grid"]):
        run_id = f"qnn_{i:02d}_{spec['model']}"
        model = _make_head(spec, embeddings.shape[1], num_classes).to(device)
        row, logs, state = _fit_head(run_id, model, embeddings, labels, split, num_classes, cfg, spec, device, debug_lines)
        all_runs.append(row)
        all_logs.extend(logs)
        pd.DataFrame(all_runs).to_csv(output_dir / "all_runs.csv", index=False)
        pd.DataFrame(all_logs).to_csv(output_dir / "training_log.csv", index=False)
        if best_row is None or row["best_val_macro_f1"] > best_row["best_val_macro_f1"]:
            best_row = row
            best_state = state
            best_model_spec = spec

    model = _make_head(best_model_spec, embeddings.shape[1], num_classes).to(device)
    model.load_state_dict(best_state)
    y_test, pred = _predict(model, embeddings, labels, split["test"], int(cfg["training"]["batch_size"]), device)
    metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
    best_metrics = {
        "dataset_id": data_cfg["dataset_id"],
        "model": best_model_spec["model"],
        "selected_by": "validation_Macro-F1",
        "source_encoder": "HybridSN tuned frozen encoder",
        **metrics,
        "best_run": best_row,
        "hybridsn_tuned_reference": {
            "OA": best_hybrid["OA"],
            "AA": best_hybrid["AA"],
            "Kappa": best_hybrid["Kappa"],
            "Macro-F1": best_hybrid["Macro-F1"],
            "Weighted-F1": best_hybrid["Weighted-F1"],
        },
    }
    write_json(output_dir / "best_metrics.json", best_metrics)
    torch.save(best_state, output_dir / "best_head_state.pt")
    _write_model_summary(output_dir / "model_summary.txt", model)
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class_metrics(y_test, pred, class_names).to_csv(output_dir / "per_class_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(output_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(
        output_dir / "normalized_confusion_matrix.csv", index=False
    )
    (output_dir / "debug_shapes.txt").write_text("\n".join(debug_lines) + "\n", encoding="utf-8")
    _write_report(output_dir / "qnn_optimization_report.md", best_metrics, all_runs)


def _extract_embeddings(cfg: dict[str, Any], hcfg: dict[str, Any], patches: np.ndarray, labels: np.ndarray, device: torch.device) -> np.ndarray:
    model = HybridSN(
        pca_channels=int(hcfg["pca_components"]),
        num_classes=16,
        embedding_dim=int(hcfg["embedding_dim"]),
        patch_size=int(hcfg["patch_size"]),
        architecture=str(hcfg["architecture"]),
        dropout=float(hcfg["dropout"]),
    ).to(device)
    model.load_state_dict(torch.load(cfg["hybridsn"]["checkpoint"], map_location=device))
    model.eval()
    loader = DataLoader(EmbeddingDataset(patches, labels, list(range(len(labels)))), batch_size=64, shuffle=False)
    chunks = []
    with torch.no_grad():
        for x, _ in loader:
            chunks.append(model.encoder(x.to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _make_head(spec: dict[str, Any], input_dim: int, num_classes: int) -> nn.Module:
    if spec["model"] == "linear_probe":
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, num_classes))
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


def _fit_head(
    run_id: str,
    model: nn.Module,
    embeddings: np.ndarray,
    labels: np.ndarray,
    split: dict[str, list[int]],
    num_classes: int,
    cfg: dict[str, Any],
    spec: dict[str, Any],
    device: torch.device,
    debug_lines: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor]]:
    train_loader = _loader(embeddings, labels, split["train"], int(cfg["training"]["batch_size"]), True)
    val_loader = _loader(embeddings, labels, split["validation"], int(cfg["training"]["batch_size"]), False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec.get("learning_rate", cfg["training"]["learning_rate"])),
        weight_decay=float(spec.get("weight_decay", cfg["training"]["weight_decay"])),
    )
    criterion = nn.CrossEntropyLoss()
    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    logs = []
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
                debug_lines.append(f"{run_id} x/logits/y: {tuple(x.shape)} {tuple(logits.shape)} {tuple(y.shape)} grad={grad_norm:.6f}")
                assert grad_norm > 0 and not math.isnan(grad_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            count += len(y)
        val_loss, y_val, pred_val = _evaluate(model, val_loader, criterion, device)
        metrics = classification_metrics(y_val, pred_val, labels=list(range(num_classes)))
        row = {
            "run_id": run_id,
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
        logs.append(row)
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
        **spec,
        "best_val_macro_f1": best_log["validation_Macro-F1"],
        "best_val_oa": best_log["validation_OA"],
        "best_val_aa": best_log["validation_AA"],
        "epochs_ran": len(logs),
        "training_time_seconds": time.time() - started,
    }, logs, best_state


def _predict(model: nn.Module, embeddings: np.ndarray, labels: np.ndarray, indices: list[int], batch_size: int, device: torch.device):
    loader = _loader(embeddings, labels, indices, batch_size, False)
    _, y_true, y_pred = _evaluate(model, loader, nn.CrossEntropyLoss(), device)
    return y_true, y_pred


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    losses, count = 0.0, 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            yy = y.to(device)
            logits = model(x)
            loss = criterion(logits, yy)
            losses += float(loss.item()) * len(y)
            count += len(y)
            ys.append(y.numpy())
            ps.append(logits.argmax(1).cpu().numpy())
    return losses / max(count, 1), np.concatenate(ys), np.concatenate(ps)


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
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm().item())
    return total


def _write_model_summary(path: Path, model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    path.write_text(f"{model}\n\nTotal parameters: {total}\nTrainable parameters: {trainable}\n", encoding="utf-8")


def _write_report(path: Path, best_metrics: dict[str, Any], all_runs: list[dict[str, Any]]) -> None:
    lines = [
        "# HybridSN-Based QNN Optimization Report",
        "",
        "Frozen tuned HybridSN encoder is used to extract 128-d embeddings. Only classifier heads are trained.",
        "",
        "## Best Test Metrics",
        "",
        "| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {best_metrics['model']} | {best_metrics['OA']*100:.2f} | {best_metrics['AA']*100:.2f} | {best_metrics['Kappa']*100:.2f} | {best_metrics['Macro-F1']*100:.2f} | {best_metrics['Weighted-F1']*100:.2f} |",
        "",
        "## Runs",
        "",
        pd.DataFrame(all_runs).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "This evaluates whether a QNN head can exploit the tuned HybridSN representation without changing the encoder. Compare against the tuned full HybridSN classifier before claiming an end-to-end improvement.",
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
