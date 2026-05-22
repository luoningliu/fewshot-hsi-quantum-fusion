# Indian Pines Best-Only Results

| model_group                    | model                               |    OA |    AA |   Kappa |   Macro-F1 |   Weighted-F1 | tuned   | note                                                        |
|:-------------------------------|:------------------------------------|------:|------:|--------:|-----------:|--------------:|:--------|:------------------------------------------------------------|
| MLP Head                       | mlp_h256_d0.0_lr0.003               | 84.05 | 77.19 |   81.73 |      78.55 |         83.77 | True    | best in tuning grid                                         |
| Residual QNN                   | Residual QNN full                   | 78.34 | 78.96 |   75.55 |      77.41 |         78.39 | True    | LayerNorm + pi*tanh + residual; best by validation Macro-F1 |
| Residual Reupload Multiobs QNN | Residual Reupload Multiobs QNN full | 78.73 | 74.25 |   75.32 |      76.13 |         77.08 | True    | data re-uploading + Z and ZZ readout + residual             |
| Bottleneck Head                | bottleneck_b32_relu_lr0.01          | 81.6  | 75.52 |   79.03 |      75.68 |         81.4  | True    | best in tuning grid                                         |
| Linear Head                    | linear_lr0.01                       | 80.05 | 70.63 |   76.94 |      73.33 |         79.43 | True    | best in tuning grid                                         |
| HybridSN                       | hybridsn                            | 81    | 68.69 |   78.27 |      69.97 |         80.48 | False   | initial deep baseline                                       |
| 2D-CNN                         | cnn2d                               | 75.77 | 59.93 |   72.06 |      60.95 |         74.51 | False   | initial deep baseline                                       |
| SVM-RBF                        | svm_C10_gammascale                  | 68.4  | 56.13 |   63.48 |      58.36 |         67.07 | True    | best in tuning grid                                         |
| kNN                            | knn                                 | 60.71 | 47.74 |   54.48 |      47.93 |         58.05 | False   | PCA spectral vector baseline                                |
| Random Forest                  | random_forest                       | 63.29 | 47.36 |   56.81 |      47.46 |         59.21 | False   | PCA spectral vector baseline                                |
| 3D-CNN                         | cnn3d                               | 27    | 12.11 |   10.48 |       4.48 |         12.47 | False   | initial deep baseline                                       |
| 1D-CNN                         | cnn1d                               | 13.98 | 10.57 |    2.29 |       2.03 |          3.99 | False   | initial deep baseline                                       |
