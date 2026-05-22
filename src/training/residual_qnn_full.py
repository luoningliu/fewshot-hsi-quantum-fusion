from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.models.quantum.residual_qnn import ResidualQNNClassifier
from src.training.train import _fit_model, _load_json, _predict, _write_model_summary
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def run(config_path: str | Path) -> None:
    config = load_yaml(config_path)
    dataset_config = load_yaml(config["dataset"])
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    embedding_data = np.load(config["embedding_path"], allow_pickle=True)
    embeddings = embedding_data["z"].astype(np.float32)
    labels = embedding_data["y"].astype(np.int64)
    split = _load_json(dataset_config["output"]["split_json"])
    training = config["training"]
    qnn = config["qnn"]
    set_seed(int(training["seed"]))
    device = _resolve_device(str(training.get("device", "auto")))

    started = time.time()
    model = ResidualQNNClassifier(
        input_dim=embeddings.shape[1],
        num_classes=int(dataset_config["num_classes"]),
        qubits=int(qnn["qubits"]),
        layers=int(qnn["layers"]),
        entanglement=str(qnn["entanglement"]),
        backend=str(qnn["backend"]),
        diff_method=str(qnn["diff_method"]),
        normalize_input=bool(qnn["normalize_input"]),
        angle_scale=float(qnn["angle_scale"]),
    ).to(device)
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
        batch_size=int(training["batch_size"]),
        device=device,
    )
    metrics = classification_metrics(y_test, pred, labels=list(range(int(dataset_config["num_classes"]))))
    row = {
        "dataset_id": dataset_config["dataset_id"],
        "model": "residual_qnn_full",
        "qubits": int(qnn["qubits"]),
        "layers": int(qnn["layers"]),
        "learning_rate": float(training["learning_rate"]),
        "entanglement": str(qnn["entanglement"]),
        **metrics,
        "training_time_seconds": time.time() - started,
        "train_size": len(split["train"]),
        "validation_size": len(split["validation"]),
        "test_size": len(split["test"]),
        "device": str(device),
    }
    pd.DataFrame([row]).to_csv(output_root / "summary_table.csv", index=False)
    pd.DataFrame(train_log).to_csv(output_root / "training_log.csv", index=False)
    write_json(output_root / "metrics.json", row)
    _write_model_summary(output_root / "model_summary.txt", model)
    torch.save(model.state_dict(), output_root / "model_state.pt")
    per_class_metrics(
        y_test,
        pred,
        {i: dataset_config["class_names"][i + 1] for i in range(int(dataset_config["num_classes"]))},
    ).to_csv(output_root / "per_class_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=list(range(int(dataset_config["num_classes"]))))
    pd.DataFrame(cm).to_csv(output_root / "confusion_matrix.csv", index=False)
    pd.DataFrame(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).to_csv(
        output_root / "normalized_confusion_matrix.csv",
        index=False,
    )
    _write_report(output_root / "report.md", row, train_log)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _write_report(path: Path, row: dict, train_log: list[dict]) -> None:
    display = pd.DataFrame([row])
    for col in ["OA", "AA", "Kappa", "Macro-F1", "Weighted-F1"]:
        display[col] = (display[col] * 100).round(2)
    lines = [
        "# Indian Pines Full-Test Residual QNN",
        "",
        display.to_markdown(index=False),
        "",
        "## Training Log",
        "",
        pd.DataFrame(train_log).to_markdown(index=False),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Indian Pines residual QNN evaluation.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

