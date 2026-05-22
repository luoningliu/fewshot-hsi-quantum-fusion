# HybridSN-small + Residual QNN Few-shot: Indian Pines

This experiment reuses the exact HybridSN-small few-shot splits. The classical HybridSN-small encoder checkpoint from each shot/seed is frozen; a residual QNN head is trained on the same support set and selected by validation Macro-F1.

## Comparison

|   shot |   HybridSN-small OA |   QNN OA |   Delta OA |   HybridSN-small Macro-F1 |   QNN Macro-F1 |   Delta Macro-F1 |   runs |
|-------:|--------------------:|---------:|-----------:|--------------------------:|---------------:|-----------------:|-------:|
|      1 |               41.49 |     5.54 |     -35.95 |                     37.75 |           4.39 |           -33.36 |      1 |

## Notes

- This is a head-level quantum comparison on top of the trained HybridSN-small encoder, not an end-to-end quantum encoder.
- It tests whether our residual QNN head improves the few-shot classifier under identical support/validation/test splits.
