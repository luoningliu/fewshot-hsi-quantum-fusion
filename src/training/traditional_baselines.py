from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.models.traditional import build_knn, build_random_forest, build_svm_rbf
from src.utils.config import load_yaml


BUILDERS = {
    "svm_rbf": build_svm_rbf,
    "random_forest": build_random_forest,
    "knn": build_knn,
}


def run_stage2(config_path: str | Path) -> None:
    config = load_yaml(config_path)
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_config_path in config["datasets"]:
        dataset_config = load_yaml(dataset_config_path)
        data = np.load(dataset_config["output"]["processed_npz"], allow_pickle=True)
        split = load_yaml_json(dataset_config["output"]["split_json"])
        x = data["spectral"]
        y = data["y"]
        num_classes = int(dataset_config["num_classes"])
        for model_name in config["models"]:
            model = BUILDERS[model_name]()
            model.fit(x[np.asarray(split["train"])], y[np.asarray(split["train"])])
            pred = model.predict(x[np.asarray(split["test"])])
            y_test = y[np.asarray(split["test"])]
            metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
            row = {"dataset_id": dataset_config["dataset_id"], "model": model_name, **metrics}
            rows.append(row)
            model_dir = output_root / dataset_config["dataset_id"] / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            write_json(model_dir / "metrics.json", row)
            per_class_metrics(
                y_test,
                pred,
                {i: dataset_config["class_names"][i + 1] for i in range(num_classes)},
            ).to_csv(model_dir / "per_class_metrics.csv", index=False)
            pd.DataFrame(confusion_matrix(y_test, pred, labels=list(range(num_classes)))).to_csv(
                model_dir / "confusion_matrix.csv",
                index=False,
            )
    pd.DataFrame(rows).to_csv(output_root / "summary_table.csv", index=False)


def load_yaml_json(path: str | Path) -> dict:
    import json

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 traditional baselines.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_stage2(args.config)


if __name__ == "__main__":
    main()

