# Spectral QNN Gated Fusion + SupCon Cross-dataset Runs

## Completed Runs

| dataset          | model                               |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:------------------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      5 |      85.7 |     2.68 |     89.43 |     2.02 |        81.67 |        3.34 |           85.87 |           1.76 |              86.43 |              2.56 |              12.2 |                   1477 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      5 |      92.5 |     3.43 |     96.58 |     1.25 |        91.66 |        3.82 |           96.1  |           1.19 |              92.55 |              3.42 |              15.4 |                   2212 |

## Seedwise Results

| dataset          | model                               |   shot |   seed |       OA |       AA |    Kappa |   Macro-F1 |   Weighted-F1 |   best_epoch |   train_time_seconds |   test_time_seconds |   trainable_parameters |   train_size |   validation_size |   test_size |   residual_scale_final |
|:-----------------|:------------------------------------|-------:|-------:|---------:|---------:|---------:|-----------:|--------------:|-------------:|---------------------:|--------------------:|-----------------------:|-------------:|------------------:|------------:|-----------------------:|
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      0 | 0.90046  | 0.90309  | 0.87005  |   0.865767 |      0.901531 |           11 |              12.8855 |             76.3734 |                   1477 |           90 |                90 |       42596 |                      1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      1 | 0.822119 | 0.860965 | 0.772516 |   0.827611 |      0.828618 |           15 |              15.2925 |             76.256  |                   1477 |           90 |                90 |       42596 |                      1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      2 | 0.870645 | 0.923308 | 0.835405 |   0.878292 |      0.883935 |           12 |              13.6715 |             77.4525 |                   1477 |           90 |                90 |       42596 |                      1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      3 | 0.850056 | 0.894039 | 0.806947 |   0.869351 |      0.851419 |           14 |              14.916  |             76.5169 |                   1477 |           90 |                90 |       42596 |                      1 |
| pavia_university | spectral_qnn_confguard_gated_supcon |     10 |      4 | 0.841534 | 0.890313 | 0.798621 |   0.852587 |      0.855886 |            9 |              11.9116 |             76.9285 |                   1477 |           90 |                90 |       42596 |                      1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      0 | 0.858314 | 0.941385 | 0.842404 |   0.937622 |      0.859139 |           15 |              26.4679 |             94.8495 |                   2212 |          160 |               160 |       53809 |                      1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      1 | 0.945009 | 0.972729 | 0.938852 |   0.966948 |      0.945449 |           10 |              21.9579 |             96.9681 |                   2212 |          160 |               160 |       53809 |                      1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      2 | 0.926667 | 0.966944 | 0.918353 |   0.962725 |      0.926643 |           19 |              31.9635 |             96.7561 |                   2212 |          160 |               160 |       53809 |                      1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      3 | 0.950454 | 0.975392 | 0.944941 |   0.970059 |      0.951032 |           17 |              29.9453 |             97.2318 |                   2212 |          160 |               160 |       53809 |                      1 |
| salinas          | spectral_qnn_confguard_gated_supcon |     10 |      4 | 0.944712 | 0.972325 | 0.938521 |   0.96773  |      0.94517  |           16 |              28.718  |             96.5193 |                   2212 |          160 |               160 |       53809 |                      1 |

## Failures

No failed runs.
