# QNN Re-uploading SupCon Minimal Batch Analysis

Output directory: `result/qnn_reupload_supcon_minibatch_salinas_pavia_10shot_20260526_103205`

## Summary vs baselines

| dataset          |   shot |   runs |   reupload_delta_vs_hybridsn_OA |   reupload_delta_vs_hybridsn_Macro-F1 |   reupload_delta_vs_hybridsn_Weighted-F1 |   reupload_delta_vs_standard_supcon_OA |   reupload_delta_vs_standard_supcon_Macro-F1 |   reupload_delta_vs_standard_supcon_Weighted-F1 |
|:-----------------|-------:|-------:|--------------------------------:|--------------------------------------:|-----------------------------------------:|---------------------------------------:|---------------------------------------------:|------------------------------------------------:|
| salinas          |     10 |      5 |                      -0.0225947 |                            0.00253473 |                               -0.0229206 |                            -0.00371313 |                                 -8.96274e-05 |                                     -0.00276848 |
| pavia_university |     10 |      5 |                       0.0473706 |                            0.0798903  |                                0.0490179 |                             0.00649357 |                                  0.00596486  |                                      0.00672318 |

## Interpretation

- salinas 10-shot: fail. Delta vs HybridSN-small: OA=-0.0226, Macro-F1=+0.0025, Weighted-F1=-0.0229.
- pavia_university 10-shot: pass. Delta vs HybridSN-small: OA=+0.0474, Macro-F1=+0.0799, Weighted-F1=+0.0490.

Conclusion: q6/l2 data re-uploading with multi-observable readout does not satisfy the PLAN acceptance rule because Salinas 10-shot still underperforms HybridSN-small on OA and Weighted-F1. It also underperforms standard SupCon on Salinas 10-shot Macro-F1 and Weighted-F1. Keep this as a diagnosed negative variant, not the next mainline model.
