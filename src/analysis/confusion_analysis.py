from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion_pair_errors(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, true_label in enumerate(labels):
        total = max(int(cm[i].sum()), 1)
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": int(cm[i, j]),
                        "error_rate_within_true_class": float(cm[i, j] / total),
                    }
                )
    return pd.DataFrame(rows)

