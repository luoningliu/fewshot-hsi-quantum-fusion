from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.analysis.metrics import classification_metrics, write_json
from src.models.quantum import QNNClassifier
from src.models.quantum.residual_qnn import ResidualQNNClassifier
from src.training.train import _fit_model, _load_json, _make_stratified_subset_split, _predict
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def run_qnn_sweep(config_path: str | Path) -> None:
    config = load_yaml(config_path)
    dataset_config = load_yaml(config["dataset"])
    split = _load_json(dataset_config["output"]["split_json"])
    embedding_data = np.load(config["embedding_path"], allow_pickle=True)
    embeddings = embedding_data["z"].astype(np.float32)
    labels = embedding_data["y"].astype(np.int64)

    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    training_config = config["training"]
    set_seed(int(training_config["seed"]))
    device = _resolve_device(str(training_config.get("device", "auto")))
    subset_split = _make_stratified_subset_split(
        labels,
        split,
        training_config["subset"],
        seed=int(training_config["seed"]),
    )

    grid = config["qnn_grid"]
    rows = []
    head_types = grid.get("head_type", ["qnn"])
    for head_type, qubits, layers, lr, entanglement in itertools.product(
        head_types,
        grid["qubits"],
        grid["layers"],
        grid["learning_rate"],
        grid["entanglement"],
    ):
        name = f"{head_type}_q{qubits}_l{layers}_lr{lr}_ent{entanglement}"
        result_dir = output_root / name
        result_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        model_cls = ResidualQNNClassifier if head_type == "residual_qnn" else QNNClassifier
        model = model_cls(
            input_dim=embeddings.shape[1],
            num_classes=int(dataset_config["num_classes"]),
            qubits=int(qubits),
            layers=int(layers),
            entanglement=str(entanglement),
            backend=str(grid.get("backend", "lightning.qubit")),
            diff_method=str(grid.get("diff_method", "adjoint")),
            normalize_input=bool(grid.get("normalize_input", True)),
            angle_scale=float(grid.get("angle_scale", np.pi)),
        ).to(device)
        train_log, best_state = _fit_model(
            model=model,
            inputs=embeddings,
            labels=labels,
            split=subset_split,
            num_classes=int(dataset_config["num_classes"]),
            training_config={
                **training_config,
                "learning_rate": float(lr),
                "patience": int(training_config["epochs"]) + 1,
            },
            device=device,
        )
        model.load_state_dict(best_state)
        y_test, pred = _predict(
            model=model,
            inputs=embeddings,
            labels=labels,
            indices=subset_split["test"],
            batch_size=int(training_config["batch_size"]),
            device=device,
        )
        metrics = classification_metrics(y_test, pred, labels=list(range(int(dataset_config["num_classes"]))))
        row = {
            "dataset_id": dataset_config["dataset_id"],
            "model": name,
            "head_type": str(head_type),
            "qubits": int(qubits),
            "layers": int(layers),
            "learning_rate": float(lr),
            "entanglement": str(entanglement),
            **metrics,
            "training_time_seconds": time.time() - started,
            "subset_train": len(subset_split["train"]),
            "subset_validation": len(subset_split["validation"]),
            "subset_test": len(subset_split["test"]),
            "device": str(device),
        }
        rows.append(row)
        pd.DataFrame(train_log).to_csv(result_dir / "training_log.csv", index=False)
        write_json(result_dir / "metrics.json", row)
        pd.DataFrame(rows).sort_values(["Macro-F1", "OA"], ascending=False).to_csv(
            output_root / "qnn_sweep_results.csv",
            index=False,
        )

    summary = pd.DataFrame(rows).sort_values(["Macro-F1", "OA"], ascending=False)
    summary.to_csv(output_root / "qnn_sweep_results.csv", index=False)
    _write_report(output_root / "report.md", summary)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    display = summary.copy()
    for col in ["OA", "AA", "Kappa", "Macro-F1", "Weighted-F1"]:
        display[col] = (display[col] * 100).round(2)
    display["training_time_seconds"] = display["training_time_seconds"].round(2)
    path.write_text("# Indian Pines QNN Sweep\n\n" + display.head(20).to_markdown(index=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Indian Pines QNN sweep on cached HybridSN embeddings.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_qnn_sweep(args.config)


if __name__ == "__main__":
    main()
