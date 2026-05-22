# Indian Pines Results Summary

| family           | model                               |    OA |    AA |   Kappa |   Macro-F1 |   Weighted-F1 | tuned   | note                                                        |
|:-----------------|:------------------------------------|------:|------:|--------:|-----------:|--------------:|:--------|:------------------------------------------------------------|
| tuned_mlp        | mlp_h256_d0.0_lr0.003               | 84.05 | 77.19 |   81.73 |      78.55 |         83.77 | True    | best in tuning grid                                         |
| qnn              | Residual QNN full                   | 78.34 | 78.96 |   75.55 |      77.41 |         78.39 | True    | LayerNorm + pi*tanh + residual; best by validation Macro-F1 |
| qnn              | Residual Reupload Multiobs QNN full | 78.73 | 74.25 |   75.32 |      76.13 |         77.08 | True    | data re-uploading + Z and ZZ readout + residual             |
| tuned_bottleneck | bottleneck_b32_relu_lr0.01          | 81.6  | 75.52 |   79.03 |      75.68 |         81.4  | True    | best in tuning grid                                         |
| tuned_linear     | linear_lr0.01                       | 80.05 | 70.63 |   76.94 |      73.33 |         79.43 | True    | best in tuning grid                                         |
| deep             | hybridsn                            | 81    | 68.69 |   78.27 |      69.97 |         80.48 | False   | initial deep baseline                                       |
| deep             | cnn2d                               | 75.77 | 59.93 |   72.06 |      60.95 |         74.51 | False   | initial deep baseline                                       |
| tuned_svm_rbf    | svm_C10_gammascale                  | 68.4  | 56.13 |   63.48 |      58.36 |         67.07 | True    | best in tuning grid                                         |
| traditional      | svm_rbf                             | 66.07 | 56.11 |   61.2  |      57.57 |         65.54 | False   | PCA spectral vector baseline                                |
| hybridsn_head    | hybridsn_mlp                        | 73.38 | 56.29 |   69.55 |      57.15 |         72.46 | False   | nan                                                         |
| hybridsn_head    | hybridsn_linear                     | 70.38 | 53.67 |   66.1  |      52.65 |         69.05 | False   | nan                                                         |
| traditional      | knn                                 | 60.71 | 47.74 |   54.48 |      47.93 |         58.05 | False   | PCA spectral vector baseline                                |
| traditional      | random_forest                       | 63.29 | 47.36 |   56.81 |      47.46 |         59.21 | False   | PCA spectral vector baseline                                |
| hybridsn_head    | hybridsn_bottleneck                 | 46.74 | 23.57 |   38.74 |      18.53 |         40.31 | False   | nan                                                         |
| hybridsn_head    | hybridsn_qnn                        | 20.44 | 17.25 |   14.08 |       7.08 |          8.4  | False   | qnn_stratified_subset                                       |
| deep             | cnn3d                               | 27    | 12.11 |   10.48 |       4.48 |         12.47 | False   | initial deep baseline                                       |
| deep             | cnn1d                               | 13.98 | 10.57 |    2.29 |       2.03 |          3.99 | False   | initial deep baseline                                       |
