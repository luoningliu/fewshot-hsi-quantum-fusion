# CNN1D / CNN3D Indian Pines Bugfix Report

## What was wrong

- The original Stage 3 run used only 15 epochs with patience 4 on CPU. The training logs show early collapse/undertraining rather than a validated working baseline.
- The CNN model forwards accepted ambiguous tensors without asserting the intended spectral/channel dimensions, so a shape regression could silently train the wrong operator.
- The Stage 3 logs did not record first-batch tensor shapes, gradient norms, train accuracy, validation loss, or sanity checks, making the abnormal numbers hard to diagnose.
- Stage 1 fits normalization and PCA on all non-background pixels by design. This debug run keeps the same split but rebuilds inputs with train-only normalization and PCA for strict supervised evaluation.

## Files changed

- `src/models/classical/cnn1d.py`
- `src/models/classical/cnn3d.py`
- `scripts/debug_cnn1d_cnn3d_indian_pines.py`

## Correct input shapes

- 1D-CNN dataset tensor: `[B, 30]`; Conv1d tensor: `[B, 1, 30]`.
- 3D-CNN dataset tensor: `[B, 9, 9, 30]`; Conv3d tensor: `[B, 1, 30, 9, 9]`.

## Before / After

| Model | Before OA | After OA | Before Macro-F1 | After Macro-F1 | Status |
|---|---:|---:|---:|---:|---|
| CNN1D | 13.98 | 59.28 | 2.03 | 46.33 | fixed |
| CNN3D | 27.00 | 63.68 | 4.48 | 49.00 | fixed |

## Sanity checks

| Model | Tiny subset train acc | Tiny loss start | Tiny loss end | Label-shuffle val acc |
|---|---:|---:|---:|---:|
| CNN1D | 100.00 | 2.8042 | 0.6857 | 23.80 |
| CNN3D | 100.00 | 2.7952 | 0.5212 | 23.90 |

## Remaining issues

- The debug run is a compact CPU-oriented run, not the full hyperparameter grid requested for final paper-quality reporting.
- Indian Pines has very small classes; Macro-F1 remains sensitive to the 10/10/80 split and should be reported with per-class metrics.
- The label-shuffle OA is below the real-label run but above the ideal 6.25% uniform random baseline because OA is strongly affected by the imbalanced validation distribution. Treat Macro-F1 and per-class metrics as the safer leakage diagnostic.

## Reliability recommendation

The baselines are reliable for debugging if shape assertions pass, tiny subset overfit exceeds 95%, and validation/test metrics improve over the abnormal Stage 3 run. For final comparison, run the documented grid and select checkpoints only by validation Macro-F1 or validation OA.
