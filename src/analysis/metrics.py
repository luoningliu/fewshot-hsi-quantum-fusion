from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support


def class_distribution(y: np.ndarray, class_names: dict[int, str], one_based_labels: bool = False) -> pd.DataFrame:
    rows = []
    for label in sorted(class_names):
        train_label = label - 1 if one_based_labels else label
        rows.append(
            {
                "class_id": int(train_label),
                "original_label": int(label) if one_based_labels else int(train_label),
                "class_name": class_names[label],
                "count": int(np.sum(y == train_label)),
            }
        )
    return pd.DataFrame(rows)


def band_statistics(cube: np.ndarray, band_names: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for i in range(cube.shape[-1]):
        values = cube[..., i].astype(np.float64)
        rows.append(
            {
                "band": band_names[i] if band_names else i,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return {
        "OA": float(np.mean(y_true == y_pred)),
        "AA": float(np.mean(per_class_recall)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "Macro-F1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: dict[int, str]) -> pd.DataFrame:
    labels = sorted(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_id": labels,
            "class_name": [class_names[label] for label in labels],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

