from __future__ import annotations

import numpy as np


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.uint8)
    return ((values - min_value) / (max_value - min_value) * 255).astype(np.uint8)

