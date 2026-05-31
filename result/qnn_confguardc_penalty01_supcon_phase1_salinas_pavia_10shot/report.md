# Spectral QNN Gated Fusion + SupCon Cross-dataset Runs

## Completed Runs

| dataset          | model                               |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:------------------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      5 |     85.77 |     2.59 |     89.48 |     1.93 |        81.76 |        3.23 |           85.93 |           1.68 |              86.49 |              2.48 |              12.6 |                   1477 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      5 |     92.6  |     3.49 |     96.59 |     1.26 |        91.77 |        3.88 |           96.12 |           1.22 |              92.64 |              3.47 |              16.8 |                   2212 |

## Seedwise Results

| dataset          | model                               |   shot |   seed |       OA |       AA |    Kappa |   Macro-F1 |   Weighted-F1 |   best_epoch |   train_time_seconds |   test_time_seconds |   trainable_parameters |   train_size |   validation_size |   test_size |   residual_scale_final |   mean_train_guard |
|:-----------------|:------------------------------------|-------:|-------:|---------:|---------:|---------:|-----------:|--------------:|-------------:|---------------------:|--------------------:|-----------------------:|-------------:|------------------:|------------:|-----------------------:|-------------------:|
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      0 | 0.900296 | 0.902638 | 0.869822 |   0.865747 |      0.901306 |           11 |              11.1965 |             68.6059 |                   1477 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      1 | 0.824819 | 0.863508 | 0.775915 |   0.830074 |      0.830966 |           16 |              13.8781 |             67.189  |                   1477 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      2 | 0.870504 | 0.923241 | 0.835234 |   0.878274 |      0.88378  |           12 |              11.7975 |             67.377  |                   1477 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      3 | 0.850948 | 0.894219 | 0.808051 |   0.869739 |      0.852316 |           15 |              13.3603 |             67.1258 |                   1477 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      4 | 0.841769 | 0.890391 | 0.798915 |   0.852715 |      0.856107 |            9 |              10.0881 |             67.3386 |                   1477 |           90 |                90 |       42596 |                      1 |                  1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      0 | 0.858091 | 0.941236 | 0.842156 |   0.93739  |      0.858912 |           15 |              23.7881 |             84.9428 |                   2212 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      1 | 0.946496 | 0.97329  | 0.940505 |   0.967513 |      0.946926 |           10 |              19.0751 |             85.371  |                   2212 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      2 | 0.928116 | 0.967155 | 0.919951 |   0.963023 |      0.928007 |           19 |              27.3564 |             84.8688 |                   2212 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      3 | 0.95248  | 0.975457 | 0.947176 |   0.970567 |      0.953005 |           24 |              32.061  |             84.7353 |                   2212 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      4 | 0.944712 | 0.972129 | 0.938508 |   0.967701 |      0.945133 |           16 |              24.9117 |             84.6097 |                   2212 |          160 |               160 |       53809 |                      1 |                  1 |

## Failures

No failed runs.
