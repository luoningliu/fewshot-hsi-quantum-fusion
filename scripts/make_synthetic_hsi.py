from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio


def main() -> None:
    rng = np.random.default_rng(42)
    height, width, bands = 32, 28, 12
    cube = rng.normal(0.0, 0.1, size=(height, width, bands)).astype(np.float32)
    gt = np.zeros((height, width), dtype=np.uint8)
    regions = [
        (slice(2, 14), slice(2, 12), 1, 0.4),
        (slice(2, 14), slice(14, 26), 2, 0.8),
        (slice(16, 29), slice(2, 12), 3, 1.2),
        (slice(16, 29), slice(14, 26), 4, 1.6),
    ]
    spectral_axis = np.linspace(0.0, 1.0, bands, dtype=np.float32)
    for row_slice, col_slice, label, offset in regions:
        gt[row_slice, col_slice] = label
        signature = offset + np.sin((label + 1) * spectral_axis)
        cube[row_slice, col_slice, :] += signature

    out_dir = Path("data/synthetic_hsi/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    sio.savemat(out_dir / "synthetic_hsi.mat", {"synthetic_hsi": cube})
    sio.savemat(out_dir / "synthetic_hsi_gt.mat", {"synthetic_hsi_gt": gt})
    print(f"Wrote synthetic HSI data under {out_dir}")


if __name__ == "__main__":
    main()

