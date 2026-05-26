# Spectral QNN Gated Fusion + SupCon Cross-dataset Runs

## Completed Runs

| dataset          | model                                       |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:--------------------------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      5 |     87    |     2.03 |     90.98 |     1.35 |        83.32 |        2.52 |           87.19 |           1.07 |              87.71 |              1.88 |              14   |                   1563 |
| salinas          | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      5 |     91.34 |     2.95 |     96.21 |     1.38 |        90.37 |        3.3  |           95.7  |           1.39 |              91.32 |              3.06 |              13.6 |                   2326 |

## Seedwise Results

| dataset          | model                                       |   shot |   seed |       OA |       AA |    Kappa |   Macro-F1 |   Weighted-F1 |   best_epoch |   train_time_seconds |   test_time_seconds |   trainable_parameters |   train_size |   validation_size |   test_size |
|:-----------------|:--------------------------------------------|-------:|-------:|---------:|---------:|---------:|-----------:|--------------:|-------------:|---------------------:|--------------------:|-----------------------:|-------------:|------------------:|------------:|
| pavia_university | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      0 | 0.9064   | 0.905842 | 0.878068 |   0.863569 |      0.909311 |           16 |              26.7506 |             135.4   |                   1563 |           90 |                90 |       42596 |
| pavia_university | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      1 | 0.869002 | 0.904192 | 0.8317   |   0.86792  |      0.874583 |           20 |              31.0713 |             133.006 |                   1563 |           90 |                90 |       42596 |
| pavia_university | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      2 | 0.871983 | 0.936665 | 0.837472 |   0.88904  |      0.883829 |           13 |              23.7433 |             133.186 |                   1563 |           90 |                90 |       42596 |
| pavia_university | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      3 | 0.855033 | 0.90198  | 0.812913 |   0.879046 |      0.856829 |           10 |              20.6122 |             132.925 |                   1563 |           90 |                90 |       42596 |
| pavia_university | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      4 | 0.847521 | 0.900518 | 0.806046 |   0.859757 |      0.860698 |           11 |              21.7053 |             133.17  |                   1563 |           90 |                90 |       42596 |
| salinas          | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      0 | 0.863536 | 0.938437 | 0.847827 |   0.933298 |      0.86104  |           15 |              45.7554 |             168.072 |                   2326 |          160 |               160 |       53809 |
| salinas          | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      1 | 0.946812 | 0.976495 | 0.940902 |   0.972858 |      0.947355 |           10 |              36.5274 |             168.033 |                   2326 |          160 |               160 |       53809 |
| salinas          | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      2 | 0.901838 | 0.956701 | 0.890759 |   0.951377 |      0.901889 |           11 |              38.3807 |             168.092 |                   2326 |          160 |               160 |       53809 |
| salinas          | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      3 | 0.916873 | 0.964038 | 0.9075   |   0.959936 |      0.917288 |           19 |              53.839  |             167.972 |                   2326 |          160 |               160 |       53809 |
| salinas          | spectral_qnn_reupload_multiobs_gated_supcon |     10 |      4 | 0.938059 | 0.974636 | 0.931278 |   0.967294 |      0.938653 |           13 |              42.0929 |             167.924 |                   2326 |          160 |               160 |       53809 |

## Failures

No failed runs.
