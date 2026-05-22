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
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.datasets.hsi_dataset import load_hsi_mat
from src.datasets.patch_extraction import extract_center_patches
from src.models.classical import HybridSN
from src.utils.config import load_yaml
from src.utils.seed import set_seed


FIXED_BASELINES = {
    "Fixed 1D-CNN": {"OA": 59.28, "AA": 46.99, "Kappa": 52.97, "Macro-F1": 46.33, "Weighted-F1": 56.63},
    "Fixed 3D-CNN": {"OA": 63.68, "AA": 49.08, "Kappa": 57.56, "Macro-F1": 49.00, "Weighted-F1": 59.92},
    "HybridSN initial": {"OA": 81.00, "AA": 68.69, "Kappa": 78.27, "Macro-F1": 69.97, "Weighted-F1": 80.48},
}


class PatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, indices: list[int]):
        self.patches = patches
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        return torch.from_numpy(self.patches[idx]).float(), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune HybridSN on Indian Pines.")
    parser.add_argument("--config", default="configs/experiments/hybridsn_tuning_indian_pines.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    output_dir = Path(cfg["output"]["root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(_to_yaml(cfg), encoding="utf-8")

    data_cfg = load_yaml(cfg["dataset"]["config"])
    raw = load_hsi_mat(data_cfg)
    split = _load_split(data_cfg["output"]["split_json"])
    num_classes = int(data_cfg["num_classes"])
    assert num_classes == 16
    rows, cols = np.nonzero(raw.gt != raw.background_label)
    labels = raw.gt[rows, cols].astype(np.int64) - 1
    assert labels.min() >= 0
    assert labels.max() < num_classes
    _assert_split(split, labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache: dict[tuple[int, int], dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    all_logs: list[dict[str, Any]] = []
    debug_lines: list[str] = [f"device: {device}", f"split counts: { {k: len(split[k]) for k in split} }"]

    def get_inputs(patch_size: int, pca_components: int) -> dict[str, Any]:
        key = (patch_size, pca_components)
        if key not in cache:
            cache[key] = _build_inputs(
                cube=raw.cube,
                rows=rows,
                cols=cols,
                split=split,
                patch_size=patch_size,
                pca_components=pca_components,
                seed=int(cfg["seed"]),
            )
        return cache[key]

    stage_a = _stage_a_configs(cfg)
    if int(cfg["search"].get("stage_a_limit", 0)) > 0:
        stage_a = stage_a[: int(cfg["search"]["stage_a_limit"])]

    for run_cfg in stage_a:
        row, logs = _run_one(
            run_id=f"A{len(all_rows):03d}",
            run_cfg=run_cfg,
            inputs=get_inputs(run_cfg["patch_size"], run_cfg["pca_components"]),
            labels=labels,
            split=split,
            num_classes=num_classes,
            device=device,
            cfg=cfg,
            debug_lines=debug_lines,
        )
        all_rows.append(row)
        all_logs.extend(logs)
        _write_progress(output_dir, all_rows, all_logs)

    top_a = sorted(all_rows, key=lambda r: (r["best_val_macro_f1"], r["best_val_aa"]), reverse=True)[
        : int(cfg["search"]["stage_b_top_k"])
    ]
    stage_b = _stage_b_configs(cfg, top_a)
    if int(cfg["search"].get("stage_b_limit", 0)) > 0:
        stage_b = stage_b[: int(cfg["search"]["stage_b_limit"])]

    for run_cfg in stage_b:
        row, logs = _run_one(
            run_id=f"B{len(all_rows):03d}",
            run_cfg=run_cfg,
            inputs=get_inputs(run_cfg["patch_size"], run_cfg["pca_components"]),
            labels=labels,
            split=split,
            num_classes=num_classes,
            device=device,
            cfg=cfg,
            debug_lines=debug_lines,
        )
        all_rows.append(row)
        all_logs.extend(logs)
        _write_progress(output_dir, all_rows, all_logs)

    best_row = sorted(all_rows, key=lambda r: (r["best_val_macro_f1"], r["best_val_aa"]), reverse=True)[0]
    best_cfg = {key: best_row[key] for key in _config_keys()}
    seed_rows = []
    best_seed_state = None
    best_seed_metric = -float("inf")
    final_logs: list[dict[str, Any]] = []
    for seed in cfg["search"]["confirmation_seeds"]:
        seed_cfg = {**best_cfg, "seed": int(seed)}
        row, logs, state = _run_one(
            run_id=f"C_seed{seed}",
            run_cfg=seed_cfg,
            inputs=get_inputs(seed_cfg["patch_size"], seed_cfg["pca_components"]),
            labels=labels,
            split=split,
            num_classes=num_classes,
            device=device,
            cfg=cfg,
            debug_lines=debug_lines,
            return_state=True,
        )
        seed_rows.append(row)
        final_logs.extend(logs)
        _write_progress(output_dir, all_rows + seed_rows, all_logs + final_logs)
        if row["best_val_macro_f1"] > best_seed_metric:
            best_seed_metric = row["best_val_macro_f1"]
            best_seed_state = state

    best_confirm = sorted(seed_rows, key=lambda r: (r["best_val_macro_f1"], r["best_val_aa"]), reverse=True)[0]
    best_cfg = {key: best_confirm[key] for key in _config_keys()}
    best_inputs = get_inputs(best_cfg["patch_size"], best_cfg["pca_components"])
    model = _make_model(best_cfg, num_classes).to(device)
    model.load_state_dict(best_seed_state["best_val_macro_f1"])
    y_test, pred = _predict(model, best_inputs["patches"], labels, split["test"], best_confirm["batch_size"], device)
    test_metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
    best_metrics = {
        "dataset_id": data_cfg["dataset_id"],
        "model": "HybridSN tuned",
        **test_metrics,
        "selected_by": "validation_Macro-F1",
        "best_config": best_cfg,
        "confirmation_mean_std": _mean_std(seed_rows),
    }

    all_rows.extend(seed_rows)
    all_logs.extend(final_logs)
    pd.DataFrame(all_rows).to_csv(output_dir / "all_runs.csv", index=False)
    pd.DataFrame(all_logs).to_csv(output_dir / "training_log.csv", index=False)
    (output_dir / "best_config.yaml").write_text(_to_yaml(best_cfg), encoding="utf-8")
    write_json(output_dir / "best_metrics.json", best_metrics)

    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class_metrics(y_test, pred, class_names).to_csv(output_dir / "per_class_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(output_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(
        output_dir / "normalized_confusion_matrix.csv", index=False
    )
    (output_dir / "model_summary.txt").write_text(_model_summary(model), encoding="utf-8")
    _save_classification_map(output_dir / "classification_map.png", raw.gt, rows, cols, pred, split["test"])
    torch.save(best_seed_state["best_val_macro_f1"], output_dir / "best_val_macro_f1.pt")
    torch.save(best_seed_state["best_val_oa"], output_dir / "best_val_oa.pt")
    torch.save(best_seed_state["best_val_aa"], output_dir / "best_val_aa.pt")
    (output_dir / "debug_shapes.txt").write_text("\n".join(debug_lines) + "\n", encoding="utf-8")
    _write_report(output_dir / "tuning_report.md", cfg, all_rows, seed_rows, best_metrics)


def _run_one(
    run_id: str,
    run_cfg: dict[str, Any],
    inputs: dict[str, Any],
    labels: np.ndarray,
    split: dict[str, list[int]],
    num_classes: int,
    device: torch.device,
    cfg: dict[str, Any],
    debug_lines: list[str],
    return_state: bool = False,
):
    set_seed(int(run_cfg.get("seed", cfg["seed"])))
    model = _make_model(run_cfg, num_classes).to(device)
    criterion = _criterion(run_cfg, labels, split["train"], num_classes, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_cfg["learning_rate"]),
        weight_decay=float(run_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, int(cfg["training"]["scheduler_patience"])),
    )
    train_loader = _loader(inputs["patches"], labels, split["train"], int(run_cfg["batch_size"]), shuffle=True)
    val_loader = _loader(inputs["patches"], labels, split["validation"], int(run_cfg["batch_size"]), shuffle=False)
    best = {"best_val_macro_f1": -1.0, "best_val_oa": -1.0, "best_val_aa": -1.0}
    best_state = {key: copy.deepcopy(model.state_dict()) for key in best}
    stale = 0
    logs: list[dict[str, Any]] = []
    started = time.time()

    if run_id == "A000":
        _shape_and_sanity_checks(run_cfg, inputs["patches"], labels, split, num_classes, device, debug_lines)

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for batch_idx, (x_raw, y) in enumerate(train_loader):
            x_raw = x_raw.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_raw)
            loss = criterion(logits, y)
            loss.backward()
            if epoch == 1 and batch_idx == 0:
                grad_norm = _grad_norm(model)
                assert grad_norm > 0 and not math.isnan(grad_norm) and not math.isinf(grad_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_count += len(y)

        val_loss, y_val, pred_val = _evaluate(model, val_loader, criterion, device)
        metrics = classification_metrics(y_val, pred_val, labels=list(range(num_classes)))
        scheduler.step(metrics["Macro-F1"])
        row = {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "train_accuracy": total_correct / max(total_count, 1),
            "validation_loss": val_loss,
            "validation_OA": metrics["OA"],
            "validation_AA": metrics["AA"],
            "validation_Kappa": metrics["Kappa"],
            "validation_Macro-F1": metrics["Macro-F1"],
            "validation_Weighted-F1": metrics["Weighted-F1"],
        }
        logs.append(row)
        improved = False
        metric_map = {
            "best_val_macro_f1": metrics["Macro-F1"],
            "best_val_oa": metrics["OA"],
            "best_val_aa": metrics["AA"],
        }
        for key, value in metric_map.items():
            if value > best[key]:
                best[key] = float(value)
                best_state[key] = copy.deepcopy(model.state_dict())
                improved = True
        stale = 0 if improved else stale + 1
        if stale >= int(cfg["training"]["early_stopping_patience"]):
            break

    result = {
        "run_id": run_id,
        **{key: run_cfg[key] for key in _config_keys()},
        **best,
        "epochs_ran": len(logs),
        "pca_explained_variance_ratio_sum": inputs["pca_evr_sum"],
        "training_time_seconds": time.time() - started,
    }
    if return_state:
        return result, logs, best_state
    return result, logs


def _shape_and_sanity_checks(
    run_cfg: dict[str, Any],
    patches: np.ndarray,
    labels: np.ndarray,
    split: dict[str, list[int]],
    num_classes: int,
    device: torch.device,
    debug_lines: list[str],
) -> None:
    model = _make_model(run_cfg, num_classes).to(device)
    loader = _loader(patches, labels, split["train"], int(run_cfg["batch_size"]), shuffle=False)
    x_raw, y = next(iter(loader))
    x_model = x_raw.permute(0, 3, 1, 2).unsqueeze(1)
    logits = model(x_raw.to(device))
    debug_lines.extend(
        [
            f"x raw shape: {tuple(x_raw.shape)}",
            f"x model shape: {tuple(x_model.shape)}",
            f"y shape: {tuple(y.shape)}",
            f"y min/max: {int(y.min().item())}/{int(y.max().item())}",
            f"logits shape: {tuple(logits.shape)}",
        ]
    )
    assert tuple(logits.shape) == (len(y), num_classes)

    tiny = _tiny_indices(labels, split["train"], 32, seed=123)
    tiny_loader = _loader(patches, labels, tiny, min(32, int(run_cfg["batch_size"])), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    first_loss = None
    final_acc = 0.0
    final_loss = 0.0
    for _ in range(150):
        correct = 0
        count = 0
        total = 0.0
        model.train()
        for x, yy in tiny_loader:
            x = x.to(device)
            yy = yy.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, yy)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(yy)
            correct += int((out.argmax(dim=1) == yy).sum().item())
            count += len(yy)
        final_loss = total / max(count, 1)
        final_acc = correct / max(count, 1)
        if first_loss is None:
            first_loss = final_loss
        if final_acc >= 0.95:
            break
    debug_lines.append(f"tiny subset overfit accuracy: {final_acc:.4f}, loss {first_loss:.6f}->{final_loss:.6f}")
    assert final_acc >= 0.95


def _make_model(run_cfg: dict[str, Any], num_classes: int) -> HybridSN:
    return HybridSN(
        pca_channels=int(run_cfg["pca_components"]),
        num_classes=num_classes,
        embedding_dim=int(run_cfg.get("embedding_dim", 128)),
        patch_size=int(run_cfg["patch_size"]),
        architecture=str(run_cfg["architecture"]),
        dropout=float(run_cfg["dropout"]),
    )


def _criterion(
    run_cfg: dict[str, Any],
    labels: np.ndarray,
    train_indices: list[int],
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    if run_cfg["loss_name"] == "cross_entropy" or run_cfg["class_weight_mode"] == "none":
        return nn.CrossEntropyLoss()
    counts = np.bincount(labels[np.asarray(train_indices, dtype=np.int64)], minlength=num_classes).astype(np.float32)
    if run_cfg["class_weight_mode"] == "inverse_freq":
        weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    elif run_cfg["class_weight_mode"] == "inverse_sqrt":
        weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        weights = weights / weights.mean()
    else:
        raise ValueError(f"Unsupported class_weight_mode: {run_cfg['class_weight_mode']}")
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def _build_inputs(
    cube: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    split: dict[str, list[int]],
    patch_size: int,
    pca_components: int,
    seed: int,
) -> dict[str, Any]:
    train_idx = np.asarray(split["train"], dtype=np.int64)
    train_pixels = cube[rows[train_idx], cols[train_idx], :].astype(np.float32)
    mean = train_pixels.mean(axis=0, dtype=np.float64)
    std = train_pixels.std(axis=0, dtype=np.float64) + 1e-6
    cube_norm = ((cube - mean) / std).astype(np.float32)
    pca = PCA(n_components=pca_components, whiten=False, random_state=seed)
    pca.fit(cube_norm[rows[train_idx], cols[train_idx], :])
    reduced = pca.transform(cube_norm.reshape(-1, cube_norm.shape[-1])).astype(np.float32)
    cube_pca = reduced.reshape(cube.shape[0], cube.shape[1], pca_components)
    return {
        "patches": extract_center_patches(cube_pca, rows, cols, patch_size),
        "pca_evr_sum": float(pca.explained_variance_ratio_.sum()),
    }


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            yy = y.to(device)
            logits = model(x)
            loss = criterion(logits, yy)
            total_loss += float(loss.item()) * len(y)
            total_count += len(y)
            y_true.append(y.numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return total_loss / max(total_count, 1), np.concatenate(y_true), np.concatenate(y_pred)


def _predict(model: nn.Module, patches: np.ndarray, labels: np.ndarray, indices: list[int], batch_size: int, device: torch.device):
    loader = _loader(patches, labels, indices, batch_size, shuffle=False)
    _, y_true, y_pred = _evaluate(model, loader, nn.CrossEntropyLoss(), device)
    return y_true, y_pred


def _loader(patches: np.ndarray, labels: np.ndarray, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(PatchDataset(patches, labels, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _stage_a_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base = cfg["search"]["stage_a_fixed"]
    out = []
    priority = [
        (9, 30),
        (7, 30),
        (11, 30),
        (9, 15),
        (9, 50),
        (7, 15),
        (11, 50),
        (7, 50),
        (11, 15),
    ]
    allowed = {(int(p), int(d)) for p in cfg["search"]["patch_size"] for d in cfg["search"]["pca_components"]}
    for patch_size, pca_components in priority:
        if (patch_size, pca_components) in allowed:
            for architecture in cfg["search"]["architectures"]:
                out.append({**base, "architecture": architecture, "patch_size": patch_size, "pca_components": pca_components})
    return out


def _stage_b_configs(cfg: dict[str, Any], top_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in top_rows:
        for lr in cfg["search"]["learning_rate"]:
            for batch_size in cfg["search"]["batch_size"]:
                for weight_decay in cfg["search"]["weight_decay"]:
                    for dropout in cfg["search"]["dropout"]:
                        for loss in cfg["search"]["losses"]:
                            out.append(
                                {
                                    **{key: row[key] for key in ("architecture", "patch_size", "pca_components")},
                                    "learning_rate": lr,
                                    "batch_size": batch_size,
                                    "weight_decay": weight_decay,
                                    "dropout": dropout,
                                    "loss_name": loss["name"],
                                    "class_weight_mode": loss["class_weight_mode"],
                                    "embedding_dim": row["embedding_dim"],
                                    "seed": cfg["seed"],
                                }
                            )
    return out


def _config_keys() -> list[str]:
    return [
        "architecture",
        "patch_size",
        "pca_components",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "dropout",
        "loss_name",
        "class_weight_mode",
        "embedding_dim",
        "seed",
    ]


def _write_progress(output_dir: Path, rows: list[dict[str, Any]], logs: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "all_runs.csv", index=False)
    if logs:
        pd.DataFrame(logs).to_csv(output_dir / "training_log.csv", index=False)


def _tiny_indices(labels: np.ndarray, train_indices: list[int], total: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    train = np.asarray(train_indices, dtype=np.int64)
    selected = []
    for label in np.unique(labels[train]):
        candidates = train[labels[train] == label]
        selected.extend(rng.choice(candidates, size=min(1, len(candidates)), replace=False).astype(int).tolist())
    remaining = total - len(selected)
    if remaining > 0:
        pool = np.asarray(sorted(set(train_indices) - set(selected)), dtype=np.int64)
        selected.extend(rng.choice(pool, size=remaining, replace=False).astype(int).tolist())
    rng.shuffle(selected)
    return selected[:total]


def _assert_split(split: dict[str, list[int]], labels: np.ndarray) -> None:
    train, val, test = set(split["train"]), set(split["validation"]), set(split["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(len(labels)))


def _load_split(path: str | Path) -> dict[str, list[int]]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {key: raw[key] for key in ("train", "validation", "test")}


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm().item())
    return total


def _mean_std(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out = {}
    for key in ("best_val_oa", "best_val_aa", "best_val_macro_f1"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return out


def _model_summary(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{model}\n\nTotal parameters: {total}\nTrainable parameters: {trainable}\n"


def _save_classification_map(path: Path, gt: np.ndarray, rows: np.ndarray, cols: np.ndarray, pred: np.ndarray, test_indices: list[int]) -> None:
    palette = np.asarray(
        [
            [0, 0, 0],
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
        ],
        dtype=np.uint8,
    )
    label_map = np.zeros_like(gt, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    label_map[rows[test], cols[test]] = pred.astype(np.int64) + 1
    Image.fromarray(palette[label_map]).save(path)


def _write_report(path: Path, cfg: dict[str, Any], all_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]], best_metrics: dict[str, Any]) -> None:
    best = best_metrics["best_config"]
    lines = [
        "# HybridSN Tuning Report: Indian Pines",
        "",
        "## Architecture",
        "",
        "HybridSN uses 3D convolution blocks for joint spectral-spatial features, reshapes `[B, C, D, H, W]` into `[B, C*D, H, W]`, then applies a 2D convolution block and an MLP classifier.",
        "",
        "## Input Shape",
        "",
        "- Dataset patch: `[B, patch_size, patch_size, pca_components]`.",
        "- Model input: `[B, 1, pca_components, patch_size, patch_size]`.",
        "- Background label 0 is removed; labels are remapped to `0..15`.",
        "",
        "## Search Space Executed",
        "",
        f"- Stage A attempted configs: {sum(str(r['run_id']).startswith('A') for r in all_rows)}.",
        f"- Stage B attempted configs: {sum(str(r['run_id']).startswith('B') for r in all_rows)}.",
        f"- Stage C confirmation seeds: {[r['seed'] for r in seed_rows]}.",
        "- Selection metric: validation Macro-F1, with validation AA as tie-break context.",
        "",
        "## Best Config",
        "",
        "```yaml",
        _to_yaml(best).strip(),
        "```",
        "",
        "## Test Metrics",
        "",
        "| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in FIXED_BASELINES.items():
        lines.append(
            f"| {name} | {row['OA']:.2f} | {row['AA']:.2f} | {row['Kappa']:.2f} | {row['Macro-F1']:.2f} | {row['Weighted-F1']:.2f} | {'already fixed' if 'Fixed' in name else 'before tuning'} |"
        )
    lines.append(
        f"| HybridSN tuned | {best_metrics['OA'] * 100:.2f} | {best_metrics['AA'] * 100:.2f} | {best_metrics['Kappa'] * 100:.2f} | {best_metrics['Macro-F1'] * 100:.2f} | {best_metrics['Weighted-F1'] * 100:.2f} | best config |"
    )
    improves = best_metrics["OA"] * 100 > FIXED_BASELINES["Fixed 3D-CNN"]["OA"]
    good = best_metrics["OA"] >= 0.80 and best_metrics["Macro-F1"] >= 0.70 and best_metrics["AA"] >= 0.70
    lines.extend(
        [
            "",
            "## Assessment",
            "",
            f"- Clear improvement over fixed 3D-CNN: {'yes' if improves else 'no'}.",
            f"- Meets good target (`OA >= 80`, `Macro-F1 >= 70`, `AA >= 70`): {'yes' if good else 'no'}.",
            "- The tuned HybridSN is suitable as the main classical encoder if final multi-seed test metrics remain stable and no QNN-specific tuning is mixed into this baseline.",
            "",
            "## Remaining Issues",
            "",
            "- This runner supports a limited staged search budget to keep CPU runs tractable; increase `stage_a_limit`, `stage_b_limit`, and `confirmation_seeds` for paper-grade tuning.",
            "- Classification map is generated on the test split only; full-image inference for every non-background pixel can be added if needed.",
        ]
    )
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
