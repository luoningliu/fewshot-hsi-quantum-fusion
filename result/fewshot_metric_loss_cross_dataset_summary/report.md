# Few-shot Metric-loss Cross-dataset Summary

## Scope

- Datasets: Indian Pines, Pavia University, Salinas.
- Shots: 5-shot and 10-shot.
- Seeds: 0-4 where completed.
- Baseline: HybridSN-small.
- QNN model: Spectral QNN Gated Fusion with classwise gate.
- Losses tested: CE + prototype loss, and CE + supervised contrastive loss on Indian Pines.

## All Model Summary

| dataset          |   shot | model                                      |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|-------:|:-------------------------------------------|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| indian_pines     |      5 | HybridSN-small                             |      5 |     72.02 |     1.67 |     82.66 |     1.45 |        68.66 |        1.9  |           63.81 |           2.17 |              73.27 |              2.06 |             118   |                  99488 |
| indian_pines     |      5 | Spectral QNN Gated Fusion + Prototype Loss |      5 |     72.05 |     1.5  |     82.52 |     0.68 |        68.68 |        1.66 |           64.12 |           1.73 |              72.84 |              2.03 |              22   |                   2176 |
| indian_pines     |      5 | Spectral QNN Gated Fusion + SupCon Loss    |      5 |     72.06 |     1.42 |     82.55 |     0.6  |        68.69 |        1.57 |           64.04 |           1.44 |              72.91 |              1.97 |              22.2 |                   2176 |
| indian_pines     |     10 | HybridSN-small                             |      5 |     80.12 |     2.56 |     88.64 |     0.94 |        77.64 |        2.79 |           71.53 |           2.02 |              80.87 |              2.7  |              85   |                  99488 |
| indian_pines     |     10 | Spectral QNN Gated Fusion + Prototype Loss |      5 |     80.88 |     2.4  |     88.87 |     1.32 |        78.49 |        2.62 |           71.82 |           1.92 |              81.61 |              2.43 |              14.4 |                   2176 |
| indian_pines     |     10 | Spectral QNN Gated Fusion + SupCon Loss    |      5 |     81.07 |     2.32 |     89.05 |     1.38 |        78.7  |        2.54 |           72.26 |           2.08 |              81.79 |              2.33 |              18.2 |                   2176 |
| pavia_university |      5 | HybridSN-small                             |      5 |     75.74 |     2.89 |     78.32 |     3.84 |        69.32 |        3.06 |           71.21 |           4.46 |              75.94 |              2.89 |              78.8 |                  99033 |
| pavia_university |      5 | Spectral QNN Gated Fusion + Prototype Loss |      5 |     77.53 |     3.48 |     82.37 |     2.83 |        71.79 |        3.92 |           77.11 |           2.17 |              78.42 |              3.38 |              21.8 |                   1455 |
| pavia_university |     10 | HybridSN-small                             |      5 |     82.26 |     4.92 |     84.31 |     5.82 |        77.31 |        6.2  |           79.2  |           8.14 |              82.8  |              5.19 |              60.2 |                  99033 |
| pavia_university |     10 | Spectral QNN Gated Fusion + Prototype Loss |      5 |     86.32 |     2.59 |     89.48 |     2.21 |        82.48 |        3.18 |           86.13 |           2.22 |              86.93 |              2.28 |              12.4 |                   1455 |
| salinas          |      5 | HybridSN-small                             |      5 |     86.93 |     7.11 |     90.9  |     8.03 |        85.42 |        8.04 |           89.67 |           9.21 |              85.91 |              9.25 |              76.8 |                  99488 |
| salinas          |      5 | Spectral QNN Gated Fusion + Prototype Loss |      5 |     88.16 |     2.09 |     94.01 |     0.86 |        86.86 |        2.33 |           93.36 |           0.78 |              88.21 |              2.09 |              21   |                   2176 |
| salinas          |     10 | HybridSN-small                             |      5 |     93.6  |     1.59 |     96.06 |     1.46 |        92.88 |        1.76 |           95.44 |           1.55 |              93.62 |              1.57 |              58.8 |                  99488 |
| salinas          |     10 | Spectral QNN Gated Fusion + Prototype Loss |      5 |     90.72 |     3.77 |     95.56 |     1.71 |        89.65 |        4.24 |           94.96 |           1.79 |              90.29 |              4.43 |              12.4 |                   2176 |

## Delta vs HybridSN-small

| dataset          |   shot | model                                      |   runs |   baseline_runs |   baseline_OA |   model_OA |   delta_OA |   baseline_AA |   model_AA |   delta_AA |   baseline_Macro-F1 |   model_Macro-F1 |   delta_Macro-F1 |   baseline_Weighted-F1 |   model_Weighted-F1 |   delta_Weighted-F1 |
|:-----------------|-------:|:-------------------------------------------|-------:|----------------:|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|--------------------:|-----------------:|-----------------:|-----------------------:|--------------------:|--------------------:|
| indian_pines     |      5 | Spectral QNN Gated Fusion + Prototype Loss |      5 |               5 |         72.02 |      72.05 |       0.03 |         82.66 |      82.52 |      -0.14 |               63.81 |            64.12 |             0.32 |                  73.27 |               72.84 |               -0.43 |
| indian_pines     |      5 | Spectral QNN Gated Fusion + SupCon Loss    |      5 |               5 |         72.02 |      72.06 |       0.04 |         82.66 |      82.55 |      -0.11 |               63.81 |            64.04 |             0.23 |                  73.27 |               72.91 |               -0.36 |
| indian_pines     |     10 | Spectral QNN Gated Fusion + Prototype Loss |      5 |               5 |         80.12 |      80.88 |       0.76 |         88.64 |      88.87 |       0.23 |               71.53 |            71.82 |             0.29 |                  80.87 |               81.61 |                0.74 |
| indian_pines     |     10 | Spectral QNN Gated Fusion + SupCon Loss    |      5 |               5 |         80.12 |      81.07 |       0.95 |         88.64 |      89.05 |       0.41 |               71.53 |            72.26 |             0.73 |                  80.87 |               81.79 |                0.92 |
| pavia_university |      5 | Spectral QNN Gated Fusion + Prototype Loss |      5 |               5 |         75.74 |      77.53 |       1.78 |         78.32 |      82.37 |       4.05 |               71.21 |            77.11 |             5.9  |                  75.94 |               78.42 |                2.48 |
| pavia_university |     10 | Spectral QNN Gated Fusion + Prototype Loss |      5 |               5 |         82.26 |      86.32 |       4.05 |         84.31 |      89.48 |       5.17 |               79.2  |            86.13 |             6.93 |                  82.8  |               86.93 |                4.13 |
| salinas          |      5 | Spectral QNN Gated Fusion + Prototype Loss |      5 |               5 |         86.93 |      88.16 |       1.23 |         90.9  |      94.01 |       3.11 |               89.67 |            93.36 |             3.69 |                  85.91 |               88.21 |                2.3  |
| salinas          |     10 | Spectral QNN Gated Fusion + Prototype Loss |      5 |               5 |         93.6  |      90.72 |      -2.88 |         96.06 |      95.56 |      -0.5  |               95.44 |            94.96 |            -0.48 |                  93.62 |               90.29 |               -3.33 |

## Notes

- Indian Pines SupCon is a fair loss-level control against prototype loss under the same QNN architecture.
- Pavia University and Salinas extension used the same few-shot split protocol and the same frozen HybridSN-small encoder strategy.
- The Pavia and Salinas extension was completed for 5-shot and 10-shot; 1-shot was not rerun in this batch because the current research decision point is the stable 5/10-shot setting.
