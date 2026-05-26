# Spectral QNN Gated Fusion + SupCon Cross-dataset Runs

## Completed Runs

| dataset          | model                                  |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:---------------------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      5 |     84.47 |     1.81 |     87.19 |     1.51 |        80.11 |        2.21 |           83.39 |           1.33 |              85.34 |              1.6  |              17.4 |                   1456 |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      5 |     92.32 |     3.62 |     96.35 |     1.26 |        91.48 |        3.99 |           95.6  |           1.54 |              92.32 |              3.67 |              14   |                   2177 |

## Seedwise Results

| dataset          | model                                  |   shot |   seed |       OA |       AA |    Kappa |   Macro-F1 |   Weighted-F1 |   best_epoch |   train_time_seconds |   test_time_seconds |   trainable_parameters |   train_size |   validation_size |   test_size |   residual_scale_final |
|:-----------------|:---------------------------------------|-------:|-------:|---------:|---------:|---------:|-----------:|--------------:|-------------:|---------------------:|--------------------:|-----------------------:|-------------:|------------------:|------------:|-----------------------:|
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      0 | 0.879613 | 0.878092 | 0.843767 |   0.831222 |      0.883829 |           11 |              12.5275 |             73.6025 |                   1456 |           90 |                90 |       42596 |              0.021241  |
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      1 | 0.832637 | 0.846567 | 0.785392 |   0.816118 |      0.839214 |           30 |              23.7424 |             72.7497 |                   1456 |           90 |                90 |       42596 |              0.034279  |
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      2 | 0.838389 | 0.89182  | 0.795245 |   0.842398 |      0.85397  |           12 |              11.7483 |             67.0964 |                   1456 |           90 |                90 |       42596 |              0.0214859 |
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      3 | 0.843624 | 0.877338 | 0.798157 |   0.854449 |      0.84562  |           12 |              11.8301 |             67.2704 |                   1456 |           90 |                90 |       42596 |              0.0219751 |
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      4 | 0.829256 | 0.865699 | 0.783165 |   0.825501 |      0.844198 |           22 |              17.0869 |             67.5033 |                   1456 |           90 |                90 |       42596 |              0.0253798 |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      0 | 0.851921 | 0.939976 | 0.836299 |   0.927722 |      0.850951 |           16 |              26.64   |             89.7967 |                   2177 |          160 |               160 |       53809 |              0.028921  |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      1 | 0.95038  | 0.976192 | 0.944753 |   0.97193  |      0.950462 |           13 |              23.9896 |             91.9918 |                   2177 |          160 |               160 |       53809 |              0.0268886 |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      2 | 0.937613 | 0.967945 | 0.930495 |   0.963501 |      0.937212 |           11 |              22.2349 |             93.0296 |                   2177 |          160 |               160 |       53809 |              0.0244752 |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      3 | 0.945139 | 0.970964 | 0.938993 |   0.964302 |      0.945746 |           21 |              32.4242 |             91.3394 |                   2177 |          160 |               160 |       53809 |              0.0351505 |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      4 | 0.930922 | 0.962296 | 0.923379 |   0.952591 |      0.931819 |            9 |              17.8824 |             85.9425 |                   2177 |          160 |               160 |       53809 |              0.0234756 |

## Failures

No failed runs.
