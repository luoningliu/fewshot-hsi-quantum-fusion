from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.spatial_split_hybridsn_indian_pines import _make_spatial_split
from scripts.tune_hybridsn_indian_pines import _build_inputs
from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.datasets.hsi_dataset import load_hsi_mat
from src.models.classical import HybridSN
from src.models.quantum import GatedResidualQNNClassifier, ResidualQNNClassifier
from src.utils.config import load_yaml
from src.utils.seed import set_seed


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: list[int]):
        self.x = x
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        return torch.from_numpy(self.x[idx]).float(), torch.tensor(int(self.y[idx]), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/spatial_split_qnn_heads_indian_pines.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["output"]["root"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    data_cfg = load_yaml(cfg["dataset"]["config"])
    raw = load_hsi_mat(data_cfg)
    rows, cols = np.nonzero(raw.gt != raw.background_label)
    labels = raw.gt[rows, cols].astype(np.int64) - 1
    num_classes = int(data_cfg["num_classes"])
    assert labels.min() >= 0 and labels.max() < num_classes and num_classes == 16
    split = _make_spatial_split(
        rows,
        cols,
        labels,
        raw.gt.shape,
        int(cfg["spatial_split"]["grid_rows"]),
        int(cfg["spatial_split"]["grid_cols"]),
        seed=int(cfg["seed"]),
        train_block_fraction=float(cfg["spatial_split"]["train_block_fraction"]),
        validation_block_fraction=float(cfg["spatial_split"]["validation_block_fraction"]),
    )
    device = _resolve_device(str(cfg["head_training"].get("device", "auto")))
    inputs = _build_inputs(
        raw.cube,
        rows,
        cols,
        split,
        int(cfg["model"]["patch_size"]),
        int(cfg["model"]["pca_components"]),
        int(cfg["seed"]),
    )
    set_seed(int(cfg["seed"]))
    encoder_model = HybridSN(
        pca_channels=int(cfg["model"]["pca_components"]),
        num_classes=num_classes,
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        patch_size=int(cfg["model"]["patch_size"]),
        architecture=str(cfg["model"]["architecture"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    encoder_logs, encoder_state = _train_encoder(encoder_model, inputs["patches"], labels, split, cfg, device, num_classes)
    encoder_model.load_state_dict(encoder_state)
    embeddings = _extract_embeddings(encoder_model, inputs["patches"], labels, device)
    np.savez_compressed(out / "spatial_embeddings.npz", z=embeddings, y=labels)

    all_runs, all_logs = [], []
    best_state = None
    best_spec = None
    best_row = None
    for spec in cfg["heads"]:
        model = _make_head(spec, embeddings.shape[1], num_classes).to(device)
        row, logs, state = _train_head(_head_name(spec), model, embeddings, labels, split, cfg, device, num_classes)
        all_runs.append(row)
        all_logs.extend(logs)
        pd.DataFrame(all_runs).to_csv(out / "all_runs.csv", index=False)
        pd.DataFrame(all_logs).to_csv(out / "head_training_log.csv", index=False)
        if best_row is None or row["best_val_macro_f1"] > best_row["best_val_macro_f1"]:
            best_row, best_state, best_spec = row, state, spec

    best_model = _make_head(best_spec, embeddings.shape[1], num_classes).to(device)
    best_model.load_state_dict(best_state)
    y_test, pred = _predict(best_model, embeddings, labels, split["test"], int(cfg["head_training"]["batch_size"]), device)
    metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
    best_metrics = {"dataset_id": data_cfg["dataset_id"], "model": _head_name(best_spec), **metrics, "best_run": best_row}
    write_json(out / "best_metrics.json", best_metrics)
    pd.DataFrame(encoder_logs).to_csv(out / "encoder_training_log.csv", index=False)
    torch.save(encoder_state, out / "spatial_encoder_state.pt")
    torch.save(best_state, out / "best_head_state.pt")
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class_metrics(y_test, pred, class_names).to_csv(out / "per_class_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(out / "confusion_matrix.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(out / "normalized_confusion_matrix.csv", index=False)
    _write_report(out / "spatial_qnn_head_report.md", cfg, split, all_runs, best_metrics)


def _train_encoder(model, patches, labels, split, cfg, device, num_classes):
    train_loader = _loader(patches, labels, split["train"], int(cfg["encoder_training"]["batch_size"]), True)
    val_loader = _loader(patches, labels, split["validation"], int(cfg["encoder_training"]["batch_size"]), False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["encoder_training"]["learning_rate"]), weight_decay=float(cfg["encoder_training"]["weight_decay"]))
    counts = np.bincount(labels[np.asarray(split["train"])], minlength=num_classes).astype(np.float32)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.mean()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale, logs = 0, []
    for epoch in range(1, int(cfg["encoder_training"]["epochs"]) + 1):
        model.train()
        total = correct = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            count += len(y)
        val_loss, yt, yp = _eval(model, val_loader, ce, device)
        m = classification_metrics(yt, yp, labels=list(range(num_classes)))
        logs.append({"epoch": epoch, "train_loss": total / max(count, 1), "train_accuracy": correct / max(count, 1), "validation_loss": val_loss, "validation_OA": m["OA"], "validation_AA": m["AA"], "validation_Macro-F1": m["Macro-F1"]})
        if m["Macro-F1"] > best_metric:
            best_metric, best_state, stale = m["Macro-F1"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= int(cfg["encoder_training"]["patience"]):
            break
    return logs, best_state


def _extract_embeddings(model, patches, labels, device):
    loader = _loader(patches, labels, list(range(len(labels))), 64, False)
    model.eval()
    chunks = []
    with torch.no_grad():
        for x, _ in loader:
            chunks.append(model.encoder(x.to(device)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def _make_head(spec, input_dim, num_classes):
    if spec["model"] == "linear_probe":
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, num_classes))
    if spec["model"] == "mlp_probe":
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, int(spec["hidden_dim"])), nn.ReLU(), nn.Dropout(float(spec["dropout"])), nn.Linear(int(spec["hidden_dim"]), num_classes))
    kwargs = {"qubits": int(spec["qubits"]), "layers": int(spec["layers"]), "entanglement": str(spec["entanglement"]), "backend": "lightning.qubit", "diff_method": "adjoint", "normalize_input": True, "angle_scale": float(spec.get("angle_scale", math.pi))}
    if spec["model"] == "residual_qnn":
        return ResidualQNNClassifier(input_dim, num_classes, **kwargs)
    if spec["model"] == "gated_residual_qnn":
        return GatedResidualQNNClassifier(input_dim, num_classes, gate_mode=str(spec.get("gate_mode", "scalar")), **kwargs)
    raise ValueError(spec["model"])


def _train_head(run_id, model, z, labels, split, cfg, device, num_classes):
    train_loader = _loader(z, labels, split["train"], int(cfg["head_training"]["batch_size"]), True)
    val_loader = _loader(z, labels, split["validation"], int(cfg["head_training"]["batch_size"]), False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["head_training"]["learning_rate"]), weight_decay=float(cfg["head_training"]["weight_decay"]))
    ce = nn.CrossEntropyLoss()
    best_metric, stale = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    logs = []
    start = time.time()
    for epoch in range(1, int(cfg["head_training"]["epochs"]) + 1):
        model.train()
        total = correct = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            count += len(y)
        val_loss, yt, yp = _eval(model, val_loader, ce, device)
        m = classification_metrics(yt, yp, labels=list(range(num_classes)))
        logs.append({"run_id": run_id, "epoch": epoch, "train_loss": total / max(count, 1), "train_accuracy": correct / max(count, 1), "validation_loss": val_loss, "validation_OA": m["OA"], "validation_AA": m["AA"], "validation_Macro-F1": m["Macro-F1"]})
        if m["Macro-F1"] > best_metric:
            best_metric, best_state, stale = m["Macro-F1"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= int(cfg["head_training"]["patience"]):
            break
    best = max(logs, key=lambda r: r["validation_Macro-F1"])
    return {"run_id": run_id, "best_val_macro_f1": best["validation_Macro-F1"], "best_val_oa": best["validation_OA"], "best_val_aa": best["validation_AA"], "epochs_ran": len(logs), "training_time_seconds": time.time() - start}, logs, best_state


def _predict(model, z, labels, indices, batch_size, device):
    loader = _loader(z, labels, indices, batch_size, False)
    _, yt, yp = _eval(model, loader, nn.CrossEntropyLoss(), device)
    return yt, yp


def _eval(model, loader, ce, device):
    model.eval()
    total = count = 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x, yy = x.to(device), y.to(device)
            logits = model(x)
            total += float(ce(logits, yy).item()) * len(y)
            count += len(y)
            ys.append(y.numpy())
            ps.append(logits.argmax(1).cpu().numpy())
    return total / max(count, 1), np.concatenate(ys), np.concatenate(ps)


def _loader(x, y, indices, batch_size, shuffle):
    return DataLoader(ArrayDataset(x, y, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _head_name(spec):
    if spec["model"] == "mlp_probe":
        return f"mlp_h{spec['hidden_dim']}"
    if spec["model"] in {"residual_qnn", "gated_residual_qnn"}:
        return f"{spec['model']}_q{spec['qubits']}_l{spec['layers']}_{spec['entanglement']}"
    return spec["model"]


def _resolve_device(device):
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))


def _write_report(path, cfg, split, rows, best):
    df = pd.DataFrame(rows)
    disp = df.copy()
    for c in ["best_val_macro_f1", "best_val_oa", "best_val_aa"]:
        disp[c] = (disp[c] * 100).round(2)
    lines = ["# Spatial Split QNN Head Pilot", "", f"Split sizes: train={len(split['train'])}, validation={len(split['validation'])}, test={len(split['test'])}", "", disp.to_markdown(index=False), "", "## Best Test", "", f"{best['model']}: OA={best['OA']*100:.2f}, AA={best['AA']*100:.2f}, Macro-F1={best['Macro-F1']*100:.2f}, Weighted-F1={best['Weighted-F1']*100:.2f}", "", "The encoder is trained only on spatial train blocks, then frozen before head comparison."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
