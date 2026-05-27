# Spectral QNN Gated Fusion + SupCon Cross-dataset Runs

## Completed Runs

| dataset          | model                                  |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:---------------------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      5 |     84.43 |     2.21 |     86.86 |     3.16 |        80.05 |        2.77 |           82.9  |           3.03 |              85.22 |              2.14 |              15.2 |                   1456 |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      5 |     91.42 |     4.16 |     96.02 |     1.65 |        90.5  |        4.56 |           95.13 |           2.1  |              91.26 |              4.56 |              10   |                   2177 |

## Seedwise Results

| dataset          | model                                  |   shot |   seed |       OA |       AA |    Kappa |   Macro-F1 |   Weighted-F1 |   best_epoch |   train_time_seconds |   test_time_seconds |   trainable_parameters |   train_size |   validation_size |   test_size |   residual_scale_final |
|:-----------------|:---------------------------------------|-------:|-------:|---------:|---------:|---------:|-----------:|--------------:|-------------:|---------------------:|--------------------:|-----------------------:|-------------:|------------------:|------------:|-----------------------:|
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      0 | 0.882336 | 0.882665 | 0.847223 |   0.835796 |      0.885745 |           13 |              14.1514 |             76.8203 |                   1456 |           90 |                90 |       42596 |              0.0962189 |
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      1 | 0.814748 | 0.807613 | 0.762212 |   0.771122 |      0.820078 |           14 |              14.97   |             76.2552 |                   1456 |           90 |                90 |       42596 |              0.10947   |
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      2 | 0.846324 | 0.898502 | 0.805176 |   0.849165 |      0.861164 |           12 |              13.8549 |             77.5434 |                   1456 |           90 |                90 |       42596 |              0.0860779 |
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      3 | 0.844985 | 0.880542 | 0.800019 |   0.857053 |      0.846928 |           12 |              14.097  |             77.0104 |                   1456 |           90 |                90 |       42596 |              0.0889292 |
| pavia_university | spectral_qnn_residualsafe_gated_supcon |     10 |      4 | 0.833036 | 0.873585 | 0.78789  |   0.831941 |      0.847187 |           25 |              21.5576 |             77.1794 |                   1456 |           90 |                90 |       42596 |              0.185137  |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      0 | 0.83369  | 0.929064 | 0.816855 |   0.911522 |      0.823876 |            8 |              18.8704 |             92.228  |                   2177 |          160 |               160 |       53809 |              0.0586594 |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      1 | 0.950547 | 0.977863 | 0.945031 |   0.972322 |      0.951019 |           11 |              23.3348 |             97.5852 |                   2177 |          160 |               160 |       53809 |              0.0950018 |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      2 | 0.938821 | 0.968771 | 0.931849 |   0.964349 |      0.938485 |           11 |              23.0027 |             95.7944 |                   2177 |          160 |               160 |       53809 |              0.0931267 |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      3 | 0.918118 | 0.962126 | 0.909051 |   0.954624 |      0.919213 |           11 |              23.7259 |             97.7249 |                   2177 |          160 |               160 |       53809 |              0.0961032 |
| salinas          | spectral_qnn_residualsafe_gated_supcon |     10 |      4 | 0.929714 | 0.963147 | 0.922061 |   0.953495 |      0.930565 |            9 |              21.0693 |             96.0637 |                   2177 |          160 |               160 |       53809 |              0.0713953 |

## Failures

No failed runs.
