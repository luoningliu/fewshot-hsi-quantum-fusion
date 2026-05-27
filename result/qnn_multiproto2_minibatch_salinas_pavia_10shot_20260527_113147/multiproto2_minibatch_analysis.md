# QNN MultiProto-2 Minibatch Analysis

Result directory: `result/qnn_multiproto2_minibatch_salinas_pavia_10shot_20260527_113147/`

Setting: `spectral_qnn_multiproto`, `prototypes_per_class=2`, Prototype Loss, standard q6_l1 QNN.

## Mean Comparison

| dataset          |   shot | model                                      |   runs |   baseline_oa |   multiproto_oa |   delta_vs_hybridsn_oa |   original_proto_oa |   delta_vs_original_proto_oa |   original_supcon_oa |   delta_vs_original_supcon_oa |   baseline_macro_f1 |   multiproto_macro_f1 |   delta_vs_hybridsn_macro_f1 |   original_proto_macro_f1 |   delta_vs_original_proto_macro_f1 |   original_supcon_macro_f1 |   delta_vs_original_supcon_macro_f1 |   baseline_weighted_f1 |   multiproto_weighted_f1 |   delta_vs_hybridsn_weighted_f1 |   original_proto_weighted_f1 |   delta_vs_original_proto_weighted_f1 |   original_supcon_weighted_f1 |   delta_vs_original_supcon_weighted_f1 |
|:-----------------|-------:|:-------------------------------------------|-------:|--------------:|----------------:|-----------------------:|--------------------:|-----------------------------:|---------------------:|------------------------------:|--------------------:|----------------------:|-----------------------------:|--------------------------:|-----------------------------------:|---------------------------:|------------------------------------:|-----------------------:|-------------------------:|--------------------------------:|-----------------------------:|--------------------------------------:|------------------------------:|---------------------------------------:|
| pavia_university |     10 | Spectral QNN MultiProto-2 + Prototype Loss |      5 |         82.26 |           86.31 |                   4.05 |               86.32 |                        -0.01 |                86.35 |                         -0.04 |               79.2  |                 86.12 |                         6.92 |                     86.13 |                              -0.01 |                      86.59 |                               -0.47 |                  82.8  |                    86.93 |                            4.12 |                        86.93 |                                 -0.01 |                         87.03 |                                  -0.1  |
| salinas          |     10 | Spectral QNN MultiProto-2 + Prototype Loss |      5 |         93.6  |           90.66 |                  -2.94 |               90.72 |                        -0.06 |                91.71 |                         -1.05 |               95.44 |                 94.94 |                        -0.51 |                     94.96 |                              -0.03 |                      95.7  |                               -0.77 |                  93.62 |                    90.22 |                           -3.4  |                        90.29 |                                 -0.07 |                         91.6  |                                  -1.38 |

## Paired Seed Delta vs HybridSN-small

| dataset          |   shot |   seed |   delta_oa |   delta_macro_f1 |   delta_weighted_f1 |
|:-----------------|-------:|-------:|-----------:|-----------------:|--------------------:|
| pavia_university |     10 |      0 |     0.0086 |           0.0139 |              0.0089 |
| pavia_university |     10 |      1 |     0.1085 |           0.2214 |              0.1229 |
| pavia_university |     10 |      2 |     0.0652 |           0.0679 |              0.0554 |
| pavia_university |     10 |      3 |     0.0199 |           0.0377 |              0.022  |
| pavia_university |     10 |      4 |     0      |           0.0051 |             -0.003  |
| salinas          |     10 |      0 |    -0.0632 |          -0.0157 |             -0.0831 |
| salinas          |     10 |      1 |    -0.026  |          -0.008  |             -0.0256 |
| salinas          |     10 |      2 |    -0.0193 |          -0.0085 |             -0.0205 |
| salinas          |     10 |      3 |     0.0074 |           0.0058 |              0.0075 |
| salinas          |     10 |      4 |    -0.0458 |           0.0011 |             -0.0481 |

## Interpretation

- Salinas 10-shot: MultiProto-2 fails the acceptance gate. OA and Weighted-F1 are worse than HybridSN-small, original Prototype QNN, and original SupCon QNN. Seed0 and seed4 are especially damaging.
- Pavia University 10-shot: MultiProto-2 is strong and exceeds HybridSN-small by a larger margin than the previous Prototype/SupCon QNN lines on OA and Macro-F1.
- Diagnosis: deterministic sub-prototypes can help Pavia, but Salinas negative transfer is not solved by simply splitting each class prototype. The Salinas failure is likely tied to seed/class-specific confusion and the metric loss over-attracting ambiguous high-support classes.
- Decision: do not expand MultiProto-2 to the full 5/10-shot matrix. The next Direction 3 variant should use a safer class-conditional mechanism, e.g. per-class prototype temperature or confidence-gated metric loss, rather than fixed deterministic sub-prototypes.
