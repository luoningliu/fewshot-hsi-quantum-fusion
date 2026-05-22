from __future__ import annotations

import numpy as np

from src.analysis.metrics import classification_metrics


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    return classification_metrics(y_true, y_pred, labels=list(range(num_classes)))

