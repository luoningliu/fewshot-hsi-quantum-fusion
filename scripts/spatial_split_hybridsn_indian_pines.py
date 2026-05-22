from __future__ import annotations

import argparse
import copy
import json
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

from scripts.tune_hybridsn_indian_pines import _build_inputs
from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.datasets.hsi_dataset import load_hsi_mat
from src.models.classical import HybridSN
from src.utils.config import load_yaml
from src.utils.seed import set_seed


class PatchDataset(Dataset):
    def __init__(self, patches, labels, indices):
        self.patches = patches
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return torch.from_numpy(self.patches[idx]).float(), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/spatial_split_hybridsn_indian_pines.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["output"]["root"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    data_cfg = load_yaml(cfg["dataset"]["config"])
    raw = load_hsi_mat(data_cfg)
    rows, cols = np.nonzero(raw.gt != raw.background_label)
    labels = raw.gt[rows, cols].astype(np.int64) - 1
    num_classes = int(data_cfg["num_classes"])
    device = _resolve_device(str(cfg["training"].get("device", "auto")))
    all_rows, all_logs = [], []
    best_state = None
    best_row = None
    best_payload = None

    for seed in cfg["spatial_split"]["seeds"]:
        split = _make_spatial_split(
            rows,
            cols,
            labels,
            raw.gt.shape,
            int(cfg["spatial_split"]["grid_rows"]),
            int(cfg["spatial_split"]["grid_cols"]),
            seed=int(seed),
            train_block_fraction=float(cfg["spatial_split"]["train_block_fraction"]),
            validation_block_fraction=float(cfg["spatial_split"]["validation_block_fraction"]),
        )
        inputs = _build_inputs(
            cube=raw.cube,
            rows=rows,
            cols=cols,
            split=split,
            patch_size=int(cfg["model"]["patch_size"]),
            pca_components=int(cfg["model"]["pca_components"]),
            seed=int(seed),
        )
        set_seed(int(seed))
        model = HybridSN(
            pca_channels=int(cfg["model"]["pca_components"]),
            num_classes=num_classes,
            embedding_dim=int(cfg["model"]["embedding_dim"]),
            patch_size=int(cfg["model"]["patch_size"]),
            architecture=str(cfg["model"]["architecture"]),
            dropout=float(cfg["model"]["dropout"]),
        ).to(device)
        row, logs, state = _fit(seed, model, inputs["patches"], labels, split, num_classes, cfg, device)
        row.update({f"{k}_size": len(v) for k, v in split.items()})
        row["pca_evr_sum"] = inputs["pca_evr_sum"]
        all_rows.append(row)
        all_logs.extend(logs)
        pd.DataFrame(all_rows).to_csv(out / "all_runs.csv", index=False)
        pd.DataFrame(all_logs).to_csv(out / "training_log.csv", index=False)
        if best_row is None or row["best_val_macro_f1"] > best_row["best_val_macro_f1"]:
            best_row = row
            best_state = state
            best_payload = (inputs, split)

    inputs, split = best_payload
    model.load_state_dict(best_state)
    y_test, pred = _predict(model, inputs["patches"], labels, split["test"], int(cfg["training"]["batch_size"]), device)
    metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
    best = {"dataset_id": data_cfg["dataset_id"], "model": "hybridsn_spatial_split", **metrics, "best_run": best_row}
    write_json(out / "best_metrics.json", best)
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class_metrics(y_test, pred, class_names).to_csv(out / "per_class_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(out / "confusion_matrix.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(out / "normalized_confusion_matrix.csv", index=False)
    _write_report(out / "spatial_split_report.md", pd.DataFrame(all_rows), best)


def _make_spatial_split(rows, cols, labels, shape, grid_rows, grid_cols, seed, train_block_fraction, validation_block_fraction):
    rng = np.random.default_rng(seed)
    h, w = shape
    br = np.minimum(rows * grid_rows // h, grid_rows - 1)
    bc = np.minimum(cols * grid_cols // w, grid_cols - 1)
    block_ids = br * grid_cols + bc
    blocks = np.unique(block_ids)
    rng.shuffle(blocks)
    n_blocks = len(blocks)
    n_train = max(1, int(round(n_blocks * train_block_fraction)))
    n_val = max(1, int(round(n_blocks * validation_block_fraction)))
    train_blocks = [int(b) for b in blocks[:n_train]]
    val_blocks = [int(b) for b in blocks[n_train : n_train + n_val]]
    test_blocks = [int(b) for b in blocks[n_train + n_val :]]
    return {
        "train": np.where(np.isin(block_ids, train_blocks))[0].astype(int).tolist(),
        "validation": np.where(np.isin(block_ids, val_blocks))[0].astype(int).tolist(),
        "test": np.where(np.isin(block_ids, test_blocks))[0].astype(int).tolist(),
    }


def _fit(seed, model, patches, labels, split, num_classes, cfg, device):
    train_loader = _loader(patches, labels, split["train"], int(cfg["training"]["batch_size"]), True)
    val_loader = _loader(patches, labels, split["validation"], int(cfg["training"]["batch_size"]), False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    counts = np.bincount(labels[np.asarray(split["train"])], minlength=num_classes).astype(np.float32)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.mean()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    best_metric = -1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    logs = []
    start = time.time()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
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
        logs.append({"seed": seed, "epoch": epoch, "train_loss": total / max(count, 1), "train_accuracy": correct / max(count, 1), "validation_loss": val_loss, "validation_OA": m["OA"], "validation_AA": m["AA"], "validation_Macro-F1": m["Macro-F1"], "validation_Weighted-F1": m["Weighted-F1"]})
        if m["Macro-F1"] > best_metric:
            best_metric = m["Macro-F1"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(cfg["training"]["patience"]):
            break
    best_log = max(logs, key=lambda r: r["validation_Macro-F1"])
    return {"seed": seed, "best_val_macro_f1": best_log["validation_Macro-F1"], "best_val_oa": best_log["validation_OA"], "best_val_aa": best_log["validation_AA"], "epochs_ran": len(logs), "training_time_seconds": time.time() - start}, logs, best_state


def _predict(model, patches, labels, indices, batch_size, device):
    loader = _loader(patches, labels, indices, batch_size, False)
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


def _loader(patches, labels, indices, batch_size, shuffle):
    return DataLoader(PatchDataset(patches, labels, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _resolve_device(device):
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))


def _write_report(path, rows, best):
    disp = rows.copy()
    for c in ["best_val_macro_f1", "best_val_oa", "best_val_aa"]:
        disp[c] = (disp[c] * 100).round(2)
    lines = ["# Spatially Disjoint HybridSN Pilot", "", "Grid-block spatial split. Blocks are mutually exclusive across train/validation/test. Ratios are adjusted to keep train class coverage.", "", disp.to_markdown(index=False), "", "## Best Test", "", f"OA={best['OA']*100:.2f}, AA={best['AA']*100:.2f}, Macro-F1={best['Macro-F1']*100:.2f}, Weighted-F1={best['Weighted-F1']*100:.2f}", "", "This should be compared against random pixel split HybridSN OA=98.80 / Macro-F1=97.19."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
