from __future__ import annotations

import numpy as np


def class_centers(features: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(label): features[labels == label].mean(axis=0) for label in np.unique(labels)}

