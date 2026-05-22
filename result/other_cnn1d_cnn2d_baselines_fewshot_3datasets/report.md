# Other Baselines Few-shot HSI Classification

All models use the same all-way few-shot sampler as HybridSN-small.

- Traditional baselines use center PCA spectral vectors.
- CNN1D uses center PCA spectral vectors.
- CNN2D and CNN3D use PCA spatial-spectral patches.
- PCA is fitted on the full image without labels to match the current HybridSN-small few-shot protocol.

## Summary

| dataset          | model   |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:--------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| indian_pines     | cnn1d   |      1 |      5 |     22.98 |     3.21 |     30.43 |     3.02 |        15.64 |        2.89 |           19.9  |           4.33 |              21.33 |              3.98 |              65.4 |                   7568 |
| indian_pines     | cnn2d   |      1 |      5 |     38.04 |     7.12 |     55.5  |     3.21 |        32.47 |        6.87 |           36.22 |           5.8  |              37.98 |              7.8  |              36.4 |                  28208 |
| indian_pines     | cnn1d   |      5 |      5 |     32.43 |     3.64 |     39.85 |     1.56 |        24.55 |        2.91 |           25.79 |           2.07 |              29.76 |              4.27 |              61.2 |                   7568 |
| indian_pines     | cnn2d   |      5 |      5 |     64.09 |     2.77 |     76.24 |     2.58 |        59.87 |        3.09 |           58.75 |           3.53 |              65.65 |              2.84 |              61.4 |                  28208 |
| indian_pines     | cnn1d   |     10 |      5 |     36.58 |     5.21 |     47.08 |     3.06 |        29.23 |        4.62 |           33.04 |           3.76 |              35.91 |              6.45 |              66.8 |                   7568 |
| indian_pines     | cnn2d   |     10 |      5 |     73.89 |     3.44 |     83.35 |     1.92 |        70.61 |        3.81 |           67.12 |           1.9  |              74.49 |              3.62 |              61.8 |                  28208 |
| pavia_university | cnn1d   |      1 |      5 |     41.13 |     9.15 |     56.12 |     6.1  |        32.05 |        8.25 |           44.42 |           7.55 |              39.53 |             12.47 |              65.2 |                   7113 |
| pavia_university | cnn2d   |      1 |      5 |     47.01 |     8.18 |     52.51 |     1.63 |        37.83 |        6.59 |           45.66 |           2.63 |              48.96 |              8.48 |              25.4 |                  27753 |
| pavia_university | cnn1d   |      5 |      5 |     59.31 |     5.23 |     70.63 |     3.27 |        49.91 |        5.47 |           60.71 |           6.51 |              59.67 |              6.07 |              89   |                   7113 |
| pavia_university | cnn2d   |      5 |      5 |     70.48 |     6.85 |     75.26 |     3.55 |        63.52 |        7.37 |           69.67 |           3.72 |              71.6  |              6.82 |              57.6 |                  27753 |
| pavia_university | cnn1d   |     10 |      5 |     62.29 |     4.84 |     73.38 |     0.95 |        53.65 |        4.29 |           65.26 |           2.38 |              63.2  |              4.67 |              83.4 |                   7113 |
| pavia_university | cnn2d   |     10 |      5 |     78.68 |     6.06 |     83.74 |     3.25 |        73.26 |        7.08 |           77.86 |           4.14 |              79.57 |              5.8  |              62.4 |                  27753 |
| salinas          | cnn1d   |      1 |      5 |     49.9  |    14.89 |     53.08 |    15.35 |        45.97 |       14.4  |           47.89 |          16.3  |              47.37 |             15.33 |              76   |                   7568 |
| salinas          | cnn2d   |      1 |      5 |     74.22 |     1.95 |     77.74 |     2.05 |        71.26 |        2.28 |           74.91 |           1.92 |              72.6  |              2.95 |              46.6 |                  28208 |
| salinas          | cnn1d   |      5 |      5 |     65.83 |     2.09 |     73.45 |     2.3  |        62.59 |        2.04 |           68.3  |           1.23 |              61.2  |              3.03 |              91.4 |                   7568 |
| salinas          | cnn2d   |      5 |      5 |     88.3  |     3.17 |     92.64 |     1.37 |        87    |        3.5  |           91    |           2    |              88.32 |              3.19 |              56.8 |                  28208 |
| salinas          | cnn1d   |     10 |      5 |     74.91 |     1.1  |     83.13 |     0.94 |        72.34 |        1.15 |           80.69 |           1.01 |              73.89 |              1.61 |              94.8 |                   7568 |
| salinas          | cnn2d   |     10 |      5 |     93.57 |     0.85 |     96.01 |     0.37 |        92.85 |        0.94 |           95.23 |           0.6  |              93.57 |              0.86 |              48.2 |                  28208 |


## Failed Runs

None.
