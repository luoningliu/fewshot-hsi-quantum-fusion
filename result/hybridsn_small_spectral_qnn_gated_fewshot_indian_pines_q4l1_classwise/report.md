# HybridSN-small + Spectral QNN Gated Fusion: Indian Pines Few-shot

This experiment freezes the HybridSN-small encoder for each shot/seed, feeds the center PCA spectrum to a QNN branch, and fuses it with the classical encoder logits using a learned scalar gate.

## Comparison

|   shot |   HybridSN-small OA |   Spectral Gated QNN OA |   Delta OA |   HybridSN-small Macro-F1 |   Spectral Gated QNN Macro-F1 |   Delta Macro-F1 |   runs |
|-------:|--------------------:|------------------------:|-----------:|--------------------------:|------------------------------:|-----------------:|-------:|
|      5 |               72.02 |                   71.19 |      -0.83 |                     63.81 |                         63.72 |            -0.08 |      5 |
|     10 |               80.12 |                   80.72 |       0.6  |                     71.53 |                         71.54 |             0.01 |      5 |

## Formula

`logits = Linear(z_classical) + gate([z_classical, x_spectral]) * QNN(x_spectral)`

The splits and encoder checkpoints are identical to the HybridSN-small baseline.
