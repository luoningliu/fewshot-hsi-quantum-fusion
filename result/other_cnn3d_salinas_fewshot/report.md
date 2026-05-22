# Other Baselines Few-shot HSI Classification

All models use the same all-way few-shot sampler as HybridSN-small.

- Traditional baselines use center PCA spectral vectors.
- CNN1D uses center PCA spectral vectors.
- CNN2D and CNN3D use PCA spatial-spectral patches.
- PCA is fitted on the full image without labels to match the current HybridSN-small few-shot protocol.

## Summary

| dataset   | model   |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:----------|:--------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| salinas   | cnn3d   |      1 |      5 |     23.94 |     8.71 |     19.87 |     5.75 |        18.11 |        7.09 |           11.1  |           3.58 |              14.87 |              5.8  |              16.4 |                   4016 |
| salinas   | cnn3d   |      5 |      5 |     25.7  |     8.49 |     21.88 |     2.85 |        19.9  |        6.75 |           12.57 |           2.51 |              16.34 |              5.55 |              21.6 |                   4016 |
| salinas   | cnn3d   |     10 |      5 |     41.54 |    18.13 |     40.54 |    18.89 |        36.58 |       18.93 |           31.7  |          19.34 |              34.3  |             20.15 |              62.2 |                   4016 |


## Failed Runs

None.
