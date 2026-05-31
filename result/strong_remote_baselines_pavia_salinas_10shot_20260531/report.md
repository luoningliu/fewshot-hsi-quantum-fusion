# Other Baselines Few-shot HSI Classification

All models use the same all-way few-shot sampler as HybridSN-small.

- Traditional baselines use center PCA spectral vectors.
- CNN1D uses center PCA spectral vectors.
- CNN2D, CNN3D, SSRN-lite, SpectralFormer-lite, and DBDA-lite use PCA spatial-spectral patches.
- The three strong baselines are compact paper-inspired implementations, not official author code.
- PCA is fitted on the full image without labels to match the current HybridSN-small few-shot protocol.

## Summary

| dataset          | model               |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:--------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university | dbda_lite           |     10 |      5 |     88.61 |     1.33 |     91.63 |     1.25 |        85.24 |        1.6  |           89.12 |           1.12 |              88.99 |              1.23 |                78 |                  45061 |
| pavia_university | spectralformer_lite |     10 |      5 |     81.62 |     4.43 |     88.19 |     3.54 |        76.59 |        5.6  |           82.56 |           4.42 |              82.42 |              4.34 |                61 |                  53673 |
| pavia_university | ssrn_lite           |     10 |      5 |     70.33 |     4.56 |     78.35 |     3.05 |        62.49 |        4.62 |           71.4  |           3.49 |              70.29 |              3.38 |                75 |                  28681 |
| salinas          | dbda_lite           |     10 |      5 |     85.39 |     3.99 |     94.4  |     1.32 |        83.87 |        4.34 |           93.16 |           1.98 |              84.49 |              5.36 |                63 |                  45404 |
| salinas          | spectralformer_lite |     10 |      5 |     88.89 |     1.62 |     95.16 |     0.74 |        87.66 |        1.8  |           94.45 |           0.69 |              88.82 |              1.67 |                54 |                  54352 |
| salinas          | ssrn_lite           |     10 |      5 |     77.41 |     3.03 |     88.63 |     1.64 |        75.25 |        3.26 |           84.86 |           2.59 |              73.01 |              5.1  |                75 |                  28800 |


## Failed Runs

None.
