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
from src.models.classical import BottleneckClassifier
from src.models.quantum import GatedResidualQNNClassifier, ResidualQNNClassifier
from src.utils.config import load_yaml
from src.utils.seed import set_seed


class EmbeddingDataset(Dataset):
    def __init__(self, z: np.ndarray, y: np.ndarray, indices: list[int]):
        self.z = z
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i])
        return torch.from_numpy(self.z[idx]).float(), torch.tensor(int(self.y[idx]), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/parameter_efficiency_hybridsn_heads_indian_pines.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["output"]["root"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(_to_yaml(cfg), encoding="utf-8")
    data_cfg = load_yaml(cfg["dataset"]["config"])
    split = _load_split(data_cfg["output"]["split_json"])
    data = np.load(cfg["embedding_path"])
    z = data["z"].astype(np.float32)
    y = data["y"].astype(np.int64)
    num_classes = int(data_cfg["num_classes"])
    assert y.min() >= 0 and y.max() < num_classes and num_classes == 16
    device = _resolve_device(str(cfg["training"].get("device", "auto")))

    rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    best_state = None
    best_spec = None
    best_row = None
    for seed in cfg["training"]["seeds"]:
        for spec in cfg["heads"]:
            set_seed(int(seed))
            run_name = _head_name(spec)
            run_id = f"{run_name}_seed{seed}"
            model = _make_head(spec, z.shape[1], num_classes).to(device)
            row, run_logs, state = _fit(run_id, model, z, y, split, cfg, spec, int(seed), device)
            row["params"] = _count_trainable(model)
            rows.append(row)
            logs.extend(run_logs)
            pd.DataFrame(rows).to_csv(out / "all_runs.csv", index=False)
            pd.DataFrame(logs).to_csv(out / "training_log.csv", index=False)
            if best_row is None or (row["best_val_macro_f1"], row["best_val_aa"]) > (
                best_row["best_val_macro_f1"],
                best_row["best_val_aa"],
            ):
                best_row = row
                best_state = state
                best_spec = spec

    summary = _summarize(rows)
    summary.to_csv(out / "summary_mean_std.csv", index=False)
    model = _make_head(best_spec, z.shape[1], num_classes).to(device)
    model.load_state_dict(best_state)
    y_test, pred = _predict(model, z, y, split["test"], int(cfg["training"]["batch_size"]), device)
    metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
    best = {"dataset_id": data_cfg["dataset_id"], "model": _head_name(best_spec), **metrics, "best_run": best_row}
    write_json(out / "best_metrics.json", best)
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class_metrics(y_test, pred, class_names).to_csv(out / "per_class_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(out / "confusion_matrix.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(out / "normalized_confusion_matrix.csv", index=False)
    _write_report(out / "parameter_efficiency_report.md", summary, best)


def _make_head(spec: dict[str, Any], input_dim: int, num_classes: int) -> nn.Module:
    if spec["model"] == "linear_probe":
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, num_classes))
    if spec["model"] == "mlp_probe":
        h = int(spec["hidden_dim"])
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(float(spec["dropout"])), nn.Linear(h, num_classes))
    if spec["model"] == "bottleneck":
        return nn.Sequential(nn.LayerNorm(input_dim), BottleneckClassifier(input_dim, num_classes, int(spec["bottleneck_dim"])))
    kwargs = {
        "qubits": int(spec["qubits"]),
        "layers": int(spec["layers"]),
        "entanglement": str(spec["entanglement"]),
        "backend": "lightning.qubit",
        "diff_method": "adjoint",
        "normalize_input": True,
        "angle_scale": float(spec.get("angle_scale", math.pi)),
    }
    if spec["model"] == "residual_qnn":
        return ResidualQNNClassifier(input_dim, num_classes, **kwargs)
    if spec["model"] == "gated_residual_qnn":
        return GatedResidualQNNClassifier(input_dim, num_classes, gate_mode=str(spec.get("gate_mode", "scalar")), **kwargs)
    raise ValueError(spec["model"])


def _fit(run_id, model, z, y, split, cfg, spec, seed, device):
    train_loader = _loader(z, y, split["train"], int(cfg["training"]["batch_size"]), True)
    val_loader = _loader(z, y, split["validation"], int(cfg["training"]["batch_size"]), False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    ce = nn.CrossEntropyLoss()
    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    logs = []
    start = time.time()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        total = correct = count = 0
        for x, yy in train_loader:
            x, yy = x.to(device), yy.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, yy)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yy)
            correct += int((logits.argmax(1) == yy).sum().item())
            count += len(yy)
        val_loss, yt, yp = _eval(model, val_loader, ce, device)
        m = classification_metrics(yt, yp, labels=list(range(16)))
        log = {"run_id": run_id, "model": _head_name(spec), "seed": seed, "epoch": epoch, "train_loss": total / max(count, 1), "train_accuracy": correct / max(count, 1), "validation_loss": val_loss, "validation_OA": m["OA"], "validation_AA": m["AA"], "validation_Macro-F1": m["Macro-F1"], "validation_Weighted-F1": m["Weighted-F1"]}
        logs.append(log)
        if m["Macro-F1"] > best_metric:
            best_metric = m["Macro-F1"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(cfg["training"]["patience"]):
            break
    best_log = max(logs, key=lambda r: r["validation_Macro-F1"])
    return {"run_id": run_id, "model": _head_name(spec), "seed": seed, "best_val_macro_f1": best_log["validation_Macro-F1"], "best_val_oa": best_log["validation_OA"], "best_val_aa": best_log["validation_AA"], "epochs_ran": len(logs), "training_time_seconds": time.time() - start}, logs, best_state


def _summarize(rows):
    df = pd.DataFrame(rows)
    out = []
    for model, g in df.groupby("model"):
        item = {"model": model, "runs": len(g), "params_mean": float(g["params"].mean())}
        for col in ["best_val_oa", "best_val_aa", "best_val_macro_f1", "training_time_seconds"]:
            item[f"{col}_mean"] = float(g[col].mean())
            item[f"{col}_std"] = float(g[col].std(ddof=0))
        item["params_per_macro_f1"] = item["params_mean"] / max(item["best_val_macro_f1_mean"], 1e-9)
        out.append(item)
    return pd.DataFrame(out).sort_values("best_val_macro_f1_mean", ascending=False)


def _predict(model, z, y, indices, batch_size, device):
    loader = _loader(z, y, indices, batch_size, False)
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


def _loader(z, y, indices, batch_size, shuffle):
    return DataLoader(EmbeddingDataset(z, y, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _head_name(spec):
    if spec["model"] == "mlp_probe":
        return f"mlp_h{spec['hidden_dim']}"
    if spec["model"] == "bottleneck":
        return f"bottleneck_b{spec['bottleneck_dim']}"
    if spec["model"] in {"residual_qnn", "gated_residual_qnn"}:
        return f"{spec['model']}_q{spec['qubits']}_l{spec['layers']}_{spec['entanglement']}"
    return spec["model"]


def _count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_split(path):
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: raw[k] for k in ("train", "validation", "test")}


def _resolve_device(device):
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))


def _write_report(path, summary, best):
    disp = summary.copy()
    for col in disp.columns:
        if ("val_" in col and (col.endswith("_mean") or col.endswith("_std"))):
            disp[col] = (disp[col] * 100).round(2)
        elif "time" in col:
            disp[col] = disp[col].round(2)
        elif col in {"params_mean", "params_per_macro_f1"}:
            disp[col] = disp[col].round(1)
    lines = ["# Parameter Efficiency: Frozen HybridSN Heads", "", disp.to_markdown(index=False), "", "## Best Selected Test", "", f"{best['model']}: OA={best['OA']*100:.2f}, AA={best['AA']*100:.2f}, Macro-F1={best['Macro-F1']*100:.2f}, Weighted-F1={best['Weighted-F1']*100:.2f}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_yaml(payload):
    return json.dumps(payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
