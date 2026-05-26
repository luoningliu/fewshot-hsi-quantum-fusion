# QNN ResidualSafe-A Minibatch Analysis

Result directory: `result/qnn_residualsafe_supcon_minibatch_salinas_pavia_10shot_20260527_001133/`

## Mean Comparison

| dataset          |   shot | model                              |   runs |   baseline_oa |   residualsafe_oa |   delta_vs_hybridsn_oa |   original_supcon_oa |   delta_vs_original_supcon_oa |   baseline_macro_f1 |   residualsafe_macro_f1 |   delta_vs_hybridsn_macro_f1 |   original_supcon_macro_f1 |   delta_vs_original_supcon_macro_f1 |   baseline_weighted_f1 |   residualsafe_weighted_f1 |   delta_vs_hybridsn_weighted_f1 |   original_supcon_weighted_f1 |   delta_vs_original_supcon_weighted_f1 |
|:-----------------|-------:|:-----------------------------------|-------:|--------------:|------------------:|-----------------------:|---------------------:|------------------------------:|--------------------:|------------------------:|-----------------------------:|---------------------------:|------------------------------------:|-----------------------:|---------------------------:|--------------------------------:|------------------------------:|---------------------------------------:|
| pavia_university |     10 | Spectral QNN ResidualSafe + SupCon |      5 |         82.26 |             84.47 |                   2.21 |                86.35 |                         -1.88 |               79.2  |                   83.39 |                         4.2  |                      86.59 |                                -3.2 |                  82.8  |                      85.34 |                            2.53 |                         87.03 |                                  -1.7  |
| salinas          |     10 | Spectral QNN ResidualSafe + SupCon |      5 |         93.6  |             92.32 |                  -1.28 |                91.71 |                          0.61 |               95.44 |                   95.6  |                         0.16 |                      95.7  |                                -0.1 |                  93.62 |                      92.32 |                           -1.29 |                         91.6  |                                   0.72 |

## Interpretation

- Salinas 10-shot: residual-safe improves over the original SupCon QNN on OA and Weighted-F1, and keeps Macro-F1 positive versus HybridSN-small, but OA and Weighted-F1 remain below HybridSN-small. It reduces but does not close the negative-transfer case.
- Pavia University 10-shot: residual-safe falls below the original SupCon QNN and only slightly exceeds HybridSN-small on mean OA / Weighted-F1 while Macro-F1 remains above baseline. The near-zero QNN residual scale appears too conservative for the setting where QNN previously provided strong gains.
- Decision: keep QNN-ResidualSafe-A as a diagnostic partial variant. The next residual-safe variant should use a less restrictive or scheduled scale, e.g. alpha warmup or a larger alpha initialization, before expanding to the full 5/10-shot matrix.
