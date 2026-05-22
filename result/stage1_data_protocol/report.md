# Stage 1 Data Protocol Report

| Dataset | Raw shape | Samples | Patch shape | PCA | EVR sum | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Indian Pines | [145, 145, 220] | 10249 | [9, 9, 30] | 30 | 0.961312 | 1024 | 1025 | 8200 |
| Pavia University | [610, 340, 103] | 42776 | [9, 9, 30] | 30 | 0.999679 | 4277 | 4277 | 34222 |
| Salinas | [512, 217, 224] | 54129 | [9, 9, 30] | 30 | 0.998638 | 5412 | 5413 | 43304 |

## Fixed Protocol

- Background label is ignored.
- Labels are stored as zero-based class ids in processed files.
- Per-band normalization and PCA are fit on all non-background pixels.
- Patch extraction uses reflect padding.
- Main split is stratified train/validation/test = 10%/10%/80% with seed 42.
