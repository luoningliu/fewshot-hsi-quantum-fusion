# Spectral QNN Gated Fusion + SupCon Cross-dataset Runs

## Completed Runs

| dataset          | model                                |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:-------------------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university | spectral_qnn_confguardb_gated_supcon |     10 |      5 |     85.18 |     2.63 |     88.69 |     2.28 |        81.05 |        3.27 |           84.86 |           2.08 |              85.95 |              2.61 |              14.6 |                   1477 |
| salinas          | spectral_qnn_confguardb_gated_supcon |     10 |      5 |     91.98 |     3.61 |     96.22 |     1.4  |        91.07 |        4.03 |           95.69 |           1.36 |              91.94 |              3.7  |              13   |                   2212 |

## Seedwise Results

| dataset          | model                                |   shot |   seed |       OA |       AA |    Kappa |   Macro-F1 |   Weighted-F1 |   best_epoch |   train_time_seconds |   test_time_seconds |   trainable_parameters |   train_size |   validation_size |   test_size |   residual_scale_final |   mean_train_guard |
|:-----------------|:-------------------------------------|-------:|-------:|---------:|---------:|---------:|-----------:|--------------:|-------------:|---------------------:|--------------------:|-----------------------:|-------------:|------------------:|------------:|-----------------------:|-------------------:|
| pavia_university | spectral_qnn_confguardb_gated_supcon |     10 |      0 | 0.891398 | 0.89391  | 0.858646 |   0.850074 |      0.893902 |           11 |              11.2344 |             67.3778 |                   1477 |           90 |                90 |       42596 |                      1 |           0.366227 |
| pavia_university | spectral_qnn_confguardb_gated_supcon |     10 |      1 | 0.815757 | 0.851241 | 0.765042 |   0.81394  |      0.821417 |           17 |              14.5972 |             73.6876 |                   1477 |           90 |                90 |       42596 |                      1 |           0.465511 |
| pavia_university | spectral_qnn_confguardb_gated_supcon |     10 |      2 | 0.869777 | 0.921923 | 0.83426  |   0.876918 |      0.883318 |           13 |              13.0518 |             71.0953 |                   1477 |           90 |                90 |       42596 |                      1 |           0.408193 |
| pavia_university | spectral_qnn_confguardb_gated_supcon |     10 |      3 | 0.845455 | 0.888189 | 0.801981 |   0.859493 |      0.847643 |           18 |              16.5104 |             68.2969 |                   1477 |           90 |                90 |       42596 |                      1 |           0.1671   |
| pavia_university | spectral_qnn_confguardb_gated_supcon |     10 |      4 | 0.836863 | 0.879287 | 0.792778 |   0.842424 |      0.851155 |           14 |              12.8694 |             67.9615 |                   1477 |           90 |                90 |       42596 |                      1 |           0.24283  |
| salinas          | spectral_qnn_confguardb_gated_supcon |     10 |      0 | 0.849263 | 0.935261 | 0.831917 |   0.930593 |      0.84719  |            8 |              18.9818 |             91.6996 |                   2212 |          160 |               160 |       53809 |                      1 |           0.664511 |
| salinas          | spectral_qnn_confguardb_gated_supcon |     10 |      1 | 0.947667 | 0.974704 | 0.941809 |   0.968884 |      0.9481   |           10 |              21.5015 |             94.5601 |                   2212 |          160 |               160 |       53809 |                      1 |           0.370995 |
| salinas          | spectral_qnn_confguardb_gated_supcon |     10 |      2 | 0.929547 | 0.965015 | 0.921388 |   0.962365 |      0.927918 |           14 |              26.1338 |             96.142  |                   2212 |          160 |               160 |       53809 |                      1 |           0.205822 |
| salinas          | spectral_qnn_confguardb_gated_supcon |     10 |      3 | 0.944303 | 0.970866 | 0.938048 |   0.964914 |      0.94487  |           23 |              36.3949 |             95.5888 |                   2212 |          160 |               160 |       53809 |                      1 |           0.185752 |
| salinas          | spectral_qnn_confguardb_gated_supcon |     10 |      4 | 0.928339 | 0.965162 | 0.920463 |   0.957784 |      0.929136 |           10 |              19.2193 |             85.2209 |                   2212 |          160 |               160 |       53809 |                      1 |           0.453411 |

## Failures

No failed runs.
