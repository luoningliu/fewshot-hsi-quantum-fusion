from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from torch import nn

from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.models.classical import MLPClassifier
from src.training.train import _fit_model, _load_json, _predict, _write_model_summary
from src.utils.config import load_yaml
from src.utils.seed import set_seed


class TunableBottleneckClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, bottleneck_dim: int, activation: str):
        super().__init__()
        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "relu":
            act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.net = nn.Sequential(nn.Linear(input_dim, bottleneck_dim), act, nn.Linear(bottleneck_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run(config_path: str | Path) -> None:
    config = load_yaml(config_path)
    dataset_config = load_yaml(config["dataset"])
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["training"]["seed"]))
    device = _resolve_device(str(config["training"].get("device", "auto")))

    embedding_data = np.load(config["embedding_path"], allow_pickle=True)
    embeddings = embedding_data["z"].astype(np.float32)
    labels = embedding_data["y"].astype(np.int64)
    processed = np.load(dataset_config["output"]["processed_npz"], allow_pickle=True)
    spectral = processed["spectral"].astype(np.float32)
    split = _load_json(dataset_config["output"]["split_json"])
    num_classes = int(dataset_config["num_classes"])

    rows: list[dict[str, Any]] = []
    rows.extend(_run_linear_grid(config, dataset_config, embeddings, labels, split, output_root, device))
    rows.extend(_run_mlp_grid(config, dataset_config, embeddings, labels, split, output_root, device))
    rows.extend(_run_bottleneck_grid(config, dataset_config, embeddings, labels, split, output_root, device))
    rows.extend(_run_svm_grid(config, dataset_config, spectral, labels, split, output_root))

    summary = pd.DataFrame(rows).sort_values(["Macro-F1", "OA"], ascending=False)
    summary.to_csv(output_root / "tuned_baseline_results.csv", index=False)
    _write_report(output_root / "report.md", summary)

    best = summary.groupby("family", as_index=False).head(1).sort_values(["Macro-F1", "OA"], ascending=False)
    best.to_csv(output_root / "best_by_family.csv", index=False)


def _run_linear_grid(config, dataset_config, embeddings, labels, split, output_root, device):
    rows = []
    for lr in config["linear_grid"]["learning_rate"]:
        model = nn.Linear(embeddings.shape[1], int(dataset_config["num_classes"]))
        row = _train_head(
            family="linear",
            name=f"linear_lr{lr}",
            model=model,
            lr=float(lr),
            config=config,
            dataset_config=dataset_config,
            embeddings=embeddings,
            labels=labels,
            split=split,
            output_root=output_root,
            device=device,
        )
        rows.append(row)
    return rows


def _run_mlp_grid(config, dataset_config, embeddings, labels, split, output_root, device):
    rows = []
    for hidden_dim, dropout, lr in itertools.product(
        config["mlp_grid"]["hidden_dim"],
        config["mlp_grid"]["dropout"],
        config["mlp_grid"]["learning_rate"],
    ):
        model = MLPClassifier(
            input_dim=embeddings.shape[1],
            num_classes=int(dataset_config["num_classes"]),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        row = _train_head(
            family="mlp",
            name=f"mlp_h{hidden_dim}_d{dropout}_lr{lr}",
            model=model,
            lr=float(lr),
            config=config,
            dataset_config=dataset_config,
            embeddings=embeddings,
            labels=labels,
            split=split,
            output_root=output_root,
            device=device,
        )
        row.update({"hidden_dim": int(hidden_dim), "dropout": float(dropout)})
        rows.append(row)
    return rows


def _run_bottleneck_grid(config, dataset_config, embeddings, labels, split, output_root, device):
    rows = []
    for bottleneck_dim, activation, lr in itertools.product(
        config["bottleneck_grid"]["bottleneck_dim"],
        config["bottleneck_grid"]["activation"],
        config["bottleneck_grid"]["learning_rate"],
    ):
        model = TunableBottleneckClassifier(
            input_dim=embeddings.shape[1],
            num_classes=int(dataset_config["num_classes"]),
            bottleneck_dim=int(bottleneck_dim),
            activation=str(activation),
        )
        row = _train_head(
            family="bottleneck",
            name=f"bottleneck_b{bottleneck_dim}_{activation}_lr{lr}",
            model=model,
            lr=float(lr),
            config=config,
            dataset_config=dataset_config,
            embeddings=embeddings,
            labels=labels,
            split=split,
            output_root=output_root,
            device=device,
        )
        row.update({"bottleneck_dim": int(bottleneck_dim), "activation": str(activation)})
        rows.append(row)
    return rows


def _train_head(
    family: str,
    name: str,
    model: nn.Module,
    lr: float,
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    embeddings: np.ndarray,
    labels: np.ndarray,
    split: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    result_dir = output_root / family / name
    result_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = model.to(device)
    training = {**config["training"], "learning_rate": lr}
    train_log, best_state = _fit_model(
        model=model,
        inputs=embeddings,
        labels=labels,
        split=split,
        num_classes=int(dataset_config["num_classes"]),
        training_config=training,
        device=device,
    )
    model.load_state_dict(best_state)
    y_test, pred = _predict(
        model=model,
        inputs=embeddings,
        labels=labels,
        indices=split["test"],
        batch_size=int(config["training"]["batch_size"]),
        device=device,
    )
    metrics = classification_metrics(y_test, pred, labels=list(range(int(dataset_config["num_classes"]))))
    row = {
        "dataset_id": dataset_config["dataset_id"],
        "family": family,
        "model": name,
        "learning_rate": lr,
        **metrics,
        "training_time_seconds": time.time() - started,
        "device": str(device),
    }
    pd.DataFrame(train_log).to_csv(result_dir / "training_log.csv", index=False)
    write_json(result_dir / "metrics.json", row)
    _write_model_summary(result_dir / "model_summary.txt", model)
    torch.save(model.state_dict(), result_dir / "model_state.pt")
    _write_classification_outputs(result_dir, dataset_config, y_test, pred)
    _append_partial(output_root, row)
    return row


def _run_svm_grid(config, dataset_config, spectral, labels, split, output_root):
    rows = []
    x_train = spectral[np.asarray(split["train"], dtype=np.int64)]
    y_train = labels[np.asarray(split["train"], dtype=np.int64)]
    x_test = spectral[np.asarray(split["test"], dtype=np.int64)]
    y_test = labels[np.asarray(split["test"], dtype=np.int64)]
    x_val = spectral[np.asarray(split["validation"], dtype=np.int64)]
    y_val = labels[np.asarray(split["validation"], dtype=np.int64)]
    for c_value, gamma in itertools.product(config["svm_grid"]["C"], config["svm_grid"]["gamma"]):
        name = f"svm_C{c_value}_gamma{gamma}"
        result_dir = output_root / "svm_rbf" / name
        result_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        model = SVC(kernel="rbf", C=float(c_value), gamma=gamma)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_metrics = classification_metrics(y_val, val_pred, labels=list(range(int(dataset_config["num_classes"]))))
        test_pred = model.predict(x_test)
        metrics = classification_metrics(y_test, test_pred, labels=list(range(int(dataset_config["num_classes"]))))
        row = {
            "dataset_id": dataset_config["dataset_id"],
            "family": "svm_rbf",
            "model": name,
            "C": float(c_value),
            "gamma": gamma,
            "validation_Macro-F1": val_metrics["Macro-F1"],
            **metrics,
            "training_time_seconds": time.time() - started,
            "device": "cpu",
        }
        write_json(result_dir / "metrics.json", row)
        _write_classification_outputs(result_dir, dataset_config, y_test, test_pred)
        rows.append(row)
        _append_partial(output_root, row)
    return rows


def _write_classification_outputs(result_dir: Path, dataset_config: dict[str, Any], y_test: np.ndarray, pred: np.ndarray) -> None:
    num_classes = int(dataset_config["num_classes"])
    per_class_metrics(
        y_test,
        pred,
        {i: dataset_config["class_names"][i + 1] for i in range(num_classes)},
    ).to_csv(result_dir / "per_class_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(result_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(
        result_dir / "normalized_confusion_matrix.csv",
        index=False,
    )


def _append_partial(output_root: Path, row: dict[str, Any]) -> None:
    path = output_root / "partial_results.csv"
    frame = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.sort_values(["Macro-F1", "OA"], ascending=False).to_csv(path, index=False)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    display = summary.copy()
    for col in ["OA", "AA", "Kappa", "Macro-F1", "Weighted-F1", "validation_Macro-F1"]:
        if col in display:
            display[col] = (display[col] * 100).round(2)
    display["training_time_seconds"] = display["training_time_seconds"].round(2)
    path.write_text("# Indian Pines Tuned Baselines\n\n" + display.head(30).to_markdown(index=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Indian Pines classical baselines.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

