# Spectral QNN Gated Fusion + SupCon Cross-dataset Runs

## Completed Runs

| dataset          | model                                        |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:---------------------------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      5 |     85.06 |     2.46 |     88.38 |     1.3  |        80.89 |        2.99 |           83.62 |           2.22 |              85.81 |              2.14 |               4.8 |                   1116 |
| salinas          | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      5 |     90.43 |     3.94 |     95.67 |     1.65 |        89.39 |        4.33 |           94.87 |           1.91 |              90.39 |              4.03 |               4.2 |                   1620 |

## Seedwise Results

| dataset          | model                                        |   shot |   seed |       OA |       AA |    Kappa |   Macro-F1 |   Weighted-F1 |   best_epoch |   train_time_seconds |   test_time_seconds |   trainable_parameters |   train_size |   validation_size |   test_size |   residual_scale_final |   mean_train_guard |
|:-----------------|:---------------------------------------------|-------:|-------:|---------:|---------:|---------:|-----------:|--------------:|-------------:|---------------------:|--------------------:|-----------------------:|-------------:|------------------:|------------:|-----------------------:|-------------------:|
| pavia_university | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      0 | 0.894004 | 0.887875 | 0.861293 |   0.859167 |      0.894193 |            2 |              7.65856 |             79.136  |                   1116 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      1 | 0.833412 | 0.88456  | 0.787399 |   0.809474 |      0.840371 |           13 |             14.445   |             78.8308 |                   1116 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      2 | 0.855503 | 0.901991 | 0.816743 |   0.853272 |      0.869502 |            5 |              9.53771 |             79.0133 |                   1116 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      3 | 0.848014 | 0.883246 | 0.80466  |   0.85016  |      0.850141 |            2 |              7.54278 |             72.6243 |                   1116 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      4 | 0.821861 | 0.861428 | 0.774438 |   0.809147 |      0.836103 |            2 |              6.45758 |             69.6043 |                   1116 |           90 |                90 |       42596 |                      1 |                  1 |
| salinas          | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      0 | 0.828616 | 0.927544 | 0.810688 |   0.913739 |      0.826208 |            8 |             20.3583  |             99.9683 |                   1620 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      1 | 0.942835 | 0.978533 | 0.936588 |   0.972261 |      0.943433 |            1 |             12.4003  |            100.574  |                   1620 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      2 | 0.912673 | 0.961535 | 0.903174 |   0.953683 |      0.913145 |            2 |             13.5427  |             99.7781 |                   1620 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      3 | 0.923972 | 0.957702 | 0.915146 |   0.952796 |      0.92253  |            5 |             17.0011  |            100.173  |                   1620 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_prelogit_confguard_gated_supcon |     10 |      4 | 0.913397 | 0.958262 | 0.903968 |   0.951018 |      0.914132 |            5 |             16.8215  |             99.6279 |                   1620 |          160 |               160 |       53809 |                      1 |                  1 |

## Failures

No failed runs.
