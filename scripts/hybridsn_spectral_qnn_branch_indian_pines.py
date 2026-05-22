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
from src.models.quantum import QNNClassifier
from src.utils.config import load_yaml
from src.utils.seed import set_seed


class FusionDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, spectral: np.ndarray, labels: np.ndarray, indices: list[int]):
        self.embeddings = embeddings
        self.spectral = spectral
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        return (
            torch.from_numpy(self.embeddings[idx]).float(),
            torch.from_numpy(self.spectral[idx]).float(),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )


class EmbeddingLinear(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes))

    def forward(self, z: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class EmbeddingMLP(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class SpectralMLPFusion(nn.Module):
    def __init__(self, embedding_dim: int, spectral_dim: int, num_classes: int, spectral_hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes))
        self.spectral_head = nn.Sequential(
            nn.LayerNorm(spectral_dim),
            nn.Linear(spectral_dim, spectral_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(spectral_hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        return self.embedding_head(z) + self.spectral_head(spectral)


class SpectralQNNFusion(nn.Module):
    def __init__(self, embedding_dim: int, spectral_dim: int, num_classes: int, **qnn_kwargs):
        super().__init__()
        self.embedding_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes))
        self.spectral_qnn = QNNClassifier(spectral_dim, num_classes, **qnn_kwargs)

    def forward(self, z: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        return self.embedding_head(z) + self.spectral_qnn(spectral)


class SpectralGatedQNNFusion(nn.Module):
    def __init__(self, embedding_dim: int, spectral_dim: int, num_classes: int, gate_mode: str = "scalar", **qnn_kwargs):
        super().__init__()
        self.embedding_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes))
        self.spectral_qnn = QNNClassifier(spectral_dim, num_classes, **qnn_kwargs)
        gate_in = embedding_dim + spectral_dim
        if gate_mode == "scalar":
            self.gate = nn.Sequential(nn.LayerNorm(gate_in), nn.Linear(gate_in, 1), nn.Sigmoid())
        elif gate_mode == "classwise":
            self.gate = nn.Sequential(nn.LayerNorm(gate_in), nn.Linear(gate_in, num_classes), nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")

    def forward(self, z: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        base_logits = self.embedding_head(z)
        qnn_logits = self.spectral_qnn(spectral)
        gate = self.gate(torch.cat([z, spectral], dim=1))
        return base_logits + gate * qnn_logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HybridSN embedding plus spectral QNN branch on Indian Pines.")
    parser.add_argument("--config", default="configs/experiments/hybridsn_spectral_qnn_branch_indian_pines.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["output"]["root"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(_to_yaml(cfg), encoding="utf-8")

    data_cfg = load_yaml(cfg["dataset"]["config"])
    raw = load_hsi_mat(data_cfg)
    split = _load_split(data_cfg["output"]["split_json"])
    rows, cols = np.nonzero(raw.gt != raw.background_label)
    labels = raw.gt[rows, cols].astype(np.int64) - 1
    num_classes = int(data_cfg["num_classes"])
    assert num_classes == 16
    assert labels.min() >= 0
    assert labels.max() < num_classes
    _assert_split(split)

    emb_data = np.load(cfg["embedding_path"])
    embeddings = emb_data["z"].astype(np.float32)
    emb_labels = emb_data["y"].astype(np.int64)
    assert embeddings.shape[0] == labels.shape[0]
    assert np.array_equal(emb_labels, labels), "Cached embeddings and rebuilt labels are not aligned."

    inputs = _build_inputs(
        cube=raw.cube,
        rows=rows,
        cols=cols,
        split=split,
        patch_size=int(cfg["input"]["patch_size"]),
        pca_components=int(cfg["input"]["pca_components"]),
        seed=int(cfg["seed"]),
    )
    patches = inputs["patches"].astype(np.float32)
    center = int(cfg["input"]["patch_size"]) // 2
    spectral = patches[:, center, center, :].astype(np.float32)
    assert spectral.shape == (labels.shape[0], int(cfg["input"]["pca_components"]))

    device = _resolve_device(str(cfg["training"].get("device", "auto")))
    debug_lines = [
        f"device: {device}",
        f"embedding shape: {embeddings.shape}",
        f"spectral shape: {spectral.shape}",
        f"patch shape: {patches.shape}",
        f"pca_evr_sum: {inputs['pca_evr_sum']:.6f}",
        f"label min/max: {labels.min()}/{labels.max()}",
        f"split counts: train={len(split['train'])}, validation={len(split['validation'])}, test={len(split['test'])}",
    ]

    all_runs: list[dict[str, Any]] = []
    all_logs: list[dict[str, Any]] = []
    best_key = None
    best_spec = None
    best_run = None
    best_state = None
    best_by_model: dict[str, dict[str, Any]] = {}
    for seed in cfg["training"]["seeds"]:
        for spec in cfg["heads"]:
            run_id = f"{spec['model']}_seed{seed}"
            set_seed(int(seed))
            model = _make_model(spec, embeddings.shape[1], spectral.shape[1], num_classes).to(device)
            row, logs, state = _fit_one(run_id, model, embeddings, spectral, labels, split, cfg, spec, int(seed), device, debug_lines)
            all_runs.append(row)
            all_logs.extend(logs)
            pd.DataFrame(all_runs).to_csv(out / "all_runs.csv", index=False)
            pd.DataFrame(all_logs).to_csv(out / "training_log.csv", index=False)
            score = (row["best_val_macro_f1"], row["best_val_aa"])
            if best_key is None or score > best_key:
                best_key = score
                best_spec = spec
                best_run = row
                best_state = state
            current = best_by_model.get(spec["model"])
            if current is None or score > current["score"]:
                best_by_model[spec["model"]] = {
                    "score": score,
                    "spec": copy.deepcopy(spec),
                    "run": copy.deepcopy(row),
                    "state": copy.deepcopy(state),
                }

    summary = _summarize(all_runs)
    summary.to_csv(out / "summary_mean_std.csv", index=False)
    model = _make_model(best_spec, embeddings.shape[1], spectral.shape[1], num_classes).to(device)
    model.load_state_dict(best_state)
    y_test, pred = _predict(model, embeddings, spectral, labels, split["test"], int(cfg["training"]["batch_size"]), device)
    metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
    best_metrics = {
        "dataset_id": data_cfg["dataset_id"],
        "model": best_spec["model"],
        "selected_by": "validation_Macro-F1",
        "source_encoder": "frozen tuned HybridSN embedding plus center-pixel PCA spectral branch",
        "pca_evr_sum": inputs["pca_evr_sum"],
        **metrics,
        "best_run": best_run,
    }
    write_json(out / "best_metrics.json", best_metrics)
    per_model_test_rows = []
    for model_name, item in best_by_model.items():
        candidate = _make_model(item["spec"], embeddings.shape[1], spectral.shape[1], num_classes).to(device)
        candidate.load_state_dict(item["state"])
        y_candidate, pred_candidate = _predict(
            candidate,
            embeddings,
            spectral,
            labels,
            split["test"],
            int(cfg["training"]["batch_size"]),
            device,
        )
        candidate_metrics = classification_metrics(y_candidate, pred_candidate, labels=list(range(num_classes)))
        per_model_test_rows.append(
            {
                "model": model_name,
                "selected_by": "validation_Macro-F1 within model",
                "best_seed": item["run"]["seed"],
                "best_val_macro_f1": item["run"]["best_val_macro_f1"],
                "best_val_oa": item["run"]["best_val_oa"],
                "best_val_aa": item["run"]["best_val_aa"],
                **candidate_metrics,
            }
        )
    per_model_test = pd.DataFrame(per_model_test_rows).sort_values("Macro-F1", ascending=False)
    per_model_test.to_csv(out / "per_model_test_metrics.csv", index=False)
    torch.save(best_state, out / "best_head_state.pt")
    _write_model_summary(out / "model_summary.txt", model)
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class_metrics(y_test, pred, class_names).to_csv(out / "per_class_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(out / "confusion_matrix.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(out / "normalized_confusion_matrix.csv", index=False)
    (out / "debug_shapes.txt").write_text("\n".join(debug_lines) + "\n", encoding="utf-8")
    _write_report(out / "spectral_qnn_branch_report.md", cfg, summary, best_metrics, per_model_test)


def _make_model(spec: dict[str, Any], embedding_dim: int, spectral_dim: int, num_classes: int) -> nn.Module:
    model = spec["model"]
    if model == "embedding_linear":
        return EmbeddingLinear(embedding_dim, num_classes)
    if model == "embedding_mlp":
        return EmbeddingMLP(embedding_dim, num_classes, int(spec["hidden_dim"]), float(spec["dropout"]))
    if model == "spectral_mlp_fusion":
        return SpectralMLPFusion(
            embedding_dim,
            spectral_dim,
            num_classes,
            int(spec["spectral_hidden_dim"]),
            float(spec["dropout"]),
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
    if model == "spectral_qnn_fusion":
        return SpectralQNNFusion(embedding_dim, spectral_dim, num_classes, **qnn_kwargs)
    if model == "spectral_gated_qnn_fusion":
        return SpectralGatedQNNFusion(
            embedding_dim,
            spectral_dim,
            num_classes,
            gate_mode=str(spec.get("gate_mode", "scalar")),
            **qnn_kwargs,
        )
    raise ValueError(f"Unsupported model: {model}")


def _fit_one(
    run_id: str,
    model: nn.Module,
    embeddings: np.ndarray,
    spectral: np.ndarray,
    labels: np.ndarray,
    split: dict[str, list[int]],
    cfg: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
    device: torch.device,
    debug_lines: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor]]:
    train_loader = _loader(embeddings, spectral, labels, split["train"], int(cfg["training"]["batch_size"]), True)
    val_loader = _loader(embeddings, spectral, labels, split["validation"], int(cfg["training"]["batch_size"]), False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0
    stale = 0
    logs: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for batch_idx, (z, s, y) in enumerate(train_loader):
            z = z.to(device)
            s = s.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(z, s)
            loss = criterion(logits, y)
            loss.backward()
            if epoch == 1 and batch_idx == 0:
                grad = _grad_norm(model)
                debug_lines.append(
                    f"{run_id}: z={tuple(z.shape)} spectral={tuple(s.shape)} logits={tuple(logits.shape)} y={tuple(y.shape)} grad={grad:.6f}"
                )
                assert logits.shape == (y.shape[0], 16)
                assert grad > 0 and not math.isnan(grad) and not math.isinf(grad)
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            count += len(y)
        val_loss, y_val, pred_val = _evaluate(model, val_loader, criterion, device)
        metrics = classification_metrics(y_val, pred_val, labels=list(range(16)))
        row = {
            "run_id": run_id,
            "model": spec["model"],
            "seed": seed,
            "epoch": epoch,
            "train_loss": loss_sum / max(count, 1),
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
    best_log = max(logs, key=lambda item: item["validation_Macro-F1"])
    return {
        "run_id": run_id,
        "model": spec["model"],
        "seed": seed,
        "best_val_macro_f1": best_log["validation_Macro-F1"],
        "best_val_oa": best_log["validation_OA"],
        "best_val_aa": best_log["validation_AA"],
        "epochs_ran": len(logs),
        "training_time_seconds": time.time() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }, logs, best_state


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


def _predict(
    model: nn.Module,
    embeddings: np.ndarray,
    spectral: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    batch_size: int,
    device: torch.device,
):
    loader = _loader(embeddings, spectral, labels, indices, batch_size, False)
    _, y_true, y_pred = _evaluate(model, loader, nn.CrossEntropyLoss(), device)
    return y_true, y_pred


def _loader(embeddings: np.ndarray, spectral: np.ndarray, labels: np.ndarray, indices: list[int], batch_size: int, shuffle: bool):
    return DataLoader(FusionDataset(embeddings, spectral, labels, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for model, group in df.groupby("model"):
        item = {
            "model": model,
            "runs": len(group),
            "parameters_mean": float(group["parameters"].mean()),
            "time_seconds_mean": float(group["training_time_seconds"].mean()),
        }
        for col in ("best_val_oa", "best_val_aa", "best_val_macro_f1"):
            item[f"{col}_mean"] = float(group[col].mean())
            item[f"{col}_std"] = float(group[col].std(ddof=0))
        out.append(item)
    return pd.DataFrame(out).sort_values("best_val_macro_f1_mean", ascending=False)


def _load_split(path: str | Path) -> dict[str, list[int]]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {key: raw[key] for key in ("train", "validation", "test")}


def _assert_split(split: dict[str, list[int]]) -> None:
    train = set(split["train"])
    val = set(split["validation"])
    test = set(split["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


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


def _write_report(
    path: Path,
    cfg: dict[str, Any],
    summary: pd.DataFrame,
    best_metrics: dict[str, Any],
    per_model_test: pd.DataFrame,
) -> None:
    reference = json.load(open(cfg["hybridsn_reference"], "r", encoding="utf-8"))
    display = summary.copy()
    for col in display.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            display[col] = (display[col] * 100).round(2) if "time" not in col and "parameters" not in col else display[col].round(2)
    lines = [
        "# HybridSN + Spectral Branch: Indian Pines",
        "",
        "This experiment freezes the tuned HybridSN encoder embedding and adds a center-pixel PCA spectral branch. The QNN branch is therefore moved earlier than the previous final-head-only QNN.",
        "",
        "## Mean Validation Results",
        "",
        display.to_markdown(index=False),
        "",
        "## Best Selected Test Result",
        "",
        "| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {best_metrics['model']} | {best_metrics['OA']*100:.2f} | {best_metrics['AA']*100:.2f} | {best_metrics['Kappa']*100:.2f} | {best_metrics['Macro-F1']*100:.2f} | {best_metrics['Weighted-F1']*100:.2f} |",
        f"| Tuned full HybridSN reference | {reference['OA']*100:.2f} | {reference['AA']*100:.2f} | {reference['Kappa']*100:.2f} | {reference['Macro-F1']*100:.2f} | {reference['Weighted-F1']*100:.2f} |",
        "",
        "## Per-Model Test Results",
        "",
        _format_percent_table(per_model_test),
        "",
        "## Interpretation",
        "",
        "A spectral QNN branch is useful only if it beats the embedding-only probes or at least improves validation Macro-F1 at a comparable parameter budget. Otherwise the current evidence still favors classical heads on frozen HybridSN features.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_percent_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in ("best_val_macro_f1", "best_val_oa", "best_val_aa", "OA", "AA", "Kappa", "Macro-F1", "Weighted-F1"):
        display[col] = (display[col] * 100).round(2)
    return display.to_markdown(index=False)


def _to_yaml(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
