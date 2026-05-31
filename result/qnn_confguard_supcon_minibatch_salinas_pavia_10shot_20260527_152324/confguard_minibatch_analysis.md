# QNN ConfidenceGuard-A Minibatch Analysis

Result directory: `result/qnn_confguard_supcon_minibatch_salinas_pavia_10shot_20260527_152324/`

Setting: `gate_context_mode=base_confidence_margin`, `gate_confidence_penalty=0.05`, SupCon, standard q6_l1 QNN.

## Mean Comparison

| dataset          |   shot | model                                 |   runs |   baseline_oa |   confguard_oa |   delta_vs_hybridsn_oa |   original_supcon_oa |   delta_vs_original_supcon_oa |   original_proto_oa |   delta_vs_original_proto_oa |   baseline_macro_f1 |   confguard_macro_f1 |   delta_vs_hybridsn_macro_f1 |   original_supcon_macro_f1 |   delta_vs_original_supcon_macro_f1 |   original_proto_macro_f1 |   delta_vs_original_proto_macro_f1 |   baseline_weighted_f1 |   confguard_weighted_f1 |   delta_vs_hybridsn_weighted_f1 |   original_supcon_weighted_f1 |   delta_vs_original_supcon_weighted_f1 |   original_proto_weighted_f1 |   delta_vs_original_proto_weighted_f1 |
|:-----------------|-------:|:--------------------------------------|-------:|--------------:|---------------:|-----------------------:|---------------------:|------------------------------:|--------------------:|-----------------------------:|--------------------:|---------------------:|-----------------------------:|---------------------------:|------------------------------------:|--------------------------:|-----------------------------------:|-----------------------:|------------------------:|--------------------------------:|------------------------------:|---------------------------------------:|-----------------------------:|--------------------------------------:|
| pavia_university |     10 | Spectral QNN ConfidenceGuard + SupCon |      5 |         82.26 |           85.7 |                   3.43 |                86.35 |                         -0.65 |               86.32 |                        -0.62 |               79.2  |                85.87 |                         6.67 |                      86.59 |                               -0.72 |                     86.13 |                              -0.26 |                  82.8  |                   86.43 |                            3.62 |                         87.03 |                                  -0.6  |                        86.93 |                                 -0.51 |
| salinas          |     10 | Spectral QNN ConfidenceGuard + SupCon |      5 |         93.6  |           92.5 |                  -1.1  |                91.71 |                          0.79 |               90.72 |                         1.78 |               95.44 |                96.1  |                         0.66 |                      95.7  |                                0.4  |                     94.96 |                               1.14 |                  93.62 |                   92.55 |                           -1.07 |                         91.6  |                                   0.95 |                        90.29 |                                  2.26 |

## Gate Seed Summary

| dataset          |   shot |   seed |   mean_gate |   mean_gate_correct |   mean_gate_wrong |   samples |
|:-----------------|-------:|-------:|------------:|--------------------:|------------------:|----------:|
| pavia_university |     10 |      0 |      0.4536 |              0.4494 |            0.4911 |     42596 |
| pavia_university |     10 |      1 |      0.6553 |              0.6576 |            0.6443 |     42596 |
| pavia_university |     10 |      2 |      0.4399 |              0.4302 |            0.5053 |     42596 |
| pavia_university |     10 |      3 |      0.551  |              0.553  |            0.5397 |     42596 |
| pavia_university |     10 |      4 |      0.5404 |              0.5411 |            0.5369 |     42596 |
| salinas          |     10 |      0 |      0.4821 |              0.4794 |            0.4985 |     53809 |
| salinas          |     10 |      1 |      0.5664 |              0.5678 |            0.5432 |     53809 |
| salinas          |     10 |      2 |      0.4442 |              0.4449 |            0.4348 |     53809 |
| salinas          |     10 |      3 |      0.4373 |              0.4389 |            0.4057 |     53809 |
| salinas          |     10 |      4 |      0.3541 |              0.3563 |            0.3149 |     53809 |

## Interpretation

- Salinas 10-shot: ConfidenceGuard-A improves over the original SupCon QNN on OA and Weighted-F1, and improves over Prototype QNN on all main metrics, but still remains below HybridSN-small on OA and Weighted-F1.
- Pavia University 10-shot: ConfidenceGuard-A preserves the strong Pavia-positive result and is close to the original SupCon-QNN line.
- Diagnosis: confidence-aware gating is the most promising negative-transfer guard so far, but `gate_confidence_penalty=0.05` is too weak to close the remaining Salinas gap. Seed0 still dominates the shortfall.
- Decision: continue Direction 5 with a stronger or more direct guard before returning to circuit/metric complexity. Next variant should test `gate_confidence_penalty=0.1` or multiplicative high-confidence suppression.
