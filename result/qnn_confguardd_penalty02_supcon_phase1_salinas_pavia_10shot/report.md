# Spectral QNN Gated Fusion + SupCon Cross-dataset Runs

## Completed Runs

| dataset          | model                               |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:------------------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      5 |     85.85 |     2.48 |     89.42 |     1.81 |        81.86 |        3.09 |           85.91 |           1.54 |              86.56 |              2.35 |              14.4 |                   1477 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      5 |     92.47 |     3.44 |     96.51 |     1.22 |        91.62 |        3.83 |           96.04 |           1.18 |              92.51 |              3.43 |              15.2 |                   2212 |

## Seedwise Results

| dataset          | model                               |   shot |   seed |       OA |       AA |    Kappa |   Macro-F1 |   Weighted-F1 |   best_epoch |   train_time_seconds |   test_time_seconds |   trainable_parameters |   train_size |   validation_size |   test_size |   residual_scale_final |   mean_train_guard |
|:-----------------|:------------------------------------|-------:|-------:|---------:|---------:|---------:|-----------:|--------------:|-------------:|---------------------:|--------------------:|-----------------------:|-------------:|------------------:|------------:|-----------------------:|-------------------:|
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      0 | 0.900178 | 0.901869 | 0.869612 |   0.866443 |      0.90099  |           11 |              12.1771 |             70.2709 |                   1477 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      1 | 0.828787 | 0.867474 | 0.780926 |   0.833554 |      0.834565 |           18 |              16.807  |             74.6927 |                   1477 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      2 | 0.869847 | 0.922713 | 0.834433 |   0.877658 |      0.883145 |           12 |              13.5612 |             74.6782 |                   1477 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      3 | 0.852216 | 0.890725 | 0.809468 |   0.867193 |      0.853829 |           21 |              18.7741 |             74.3748 |                   1477 |           90 |                90 |       42596 |                      1 |                  1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      4 | 0.841487 | 0.888368 | 0.798486 |   0.85061  |      0.855371 |           10 |              12.2116 |             75.9    |                   1477 |           90 |                90 |       42596 |                      1 |                  1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      0 | 0.857626 | 0.941168 | 0.841625 |   0.937219 |      0.858391 |           15 |              23.5522 |             90.9169 |                   2212 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      1 | 0.94828  | 0.974143 | 0.94249  |   0.968283 |      0.948703 |           10 |              21.8706 |             95.1129 |                   2212 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      2 | 0.926759 | 0.96641  | 0.918466 |   0.961817 |      0.926742 |           12 |              24.1837 |             97.8327 |                   2212 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      3 | 0.945827 | 0.971587 | 0.939736 |   0.96725  |      0.946306 |           23 |              37.2925 |             98.803  |                   2212 |          160 |               160 |       53809 |                      1 |                  1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      4 | 0.944991 | 0.972289 | 0.938816 |   0.967675 |      0.9454   |           16 |              24.7084 |             85.5357 |                   2212 |          160 |               160 |       53809 |                      1 |                  1 |

## Failures

No failed runs.
