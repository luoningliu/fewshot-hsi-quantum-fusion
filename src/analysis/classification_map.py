from __future__ import annotations

import numpy as np


def predictions_to_map(rows: np.ndarray, cols: np.ndarray, preds: np.ndarray, shape: tuple[int, int], background: int = 0) -> np.ndarray:
    output = np.full(shape, background, dtype=np.int64)
    output[rows, cols] = preds + 1
    return output

