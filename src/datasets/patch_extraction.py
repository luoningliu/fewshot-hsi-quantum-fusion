from __future__ import annotations

import numpy as np


def extract_center_patches(cube: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch_size: int) -> np.ndarray:
    if patch_size % 2 != 1:
        raise ValueError(f"patch_size must be odd, got {patch_size}")
    radius = patch_size // 2
    padded = np.pad(cube, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    patches = np.empty((len(rows), patch_size, patch_size, cube.shape[-1]), dtype=np.float32)
    for i, (row, col) in enumerate(zip(rows, cols)):
        row_p = int(row) + radius
        col_p = int(col) + radius
        patches[i] = padded[row_p - radius : row_p + radius + 1, col_p - radius : col_p + radius + 1, :]
    return patches

