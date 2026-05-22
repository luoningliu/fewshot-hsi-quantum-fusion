# Stage 1 Data Protocol Report

| Dataset | Raw shape | Samples | Patch shape | PCA | EVR sum | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Synthetic HSI | [32, 28, 12] | 550 | [5, 5, 6] | 6 | 0.977600 | 55 | 55 | 440 |

## Fixed Protocol

- Background label is ignored.
- Labels are stored as zero-based class ids in processed files.
- Per-band normalization and PCA are fit on all non-background pixels.
- Patch extraction uses reflect padding.
- Main split is stratified train/validation/test = 10%/10%/80% with seed 42.
