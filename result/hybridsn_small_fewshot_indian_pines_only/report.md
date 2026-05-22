# HybridSN-small Few-shot Results: Indian Pines

This report filters the completed Indian Pines runs from the full queued experiment. Pavia/Salinas were stopped after Indian Pines completed.

## Protocol

- Model: HybridSNSmall, 3D CNN blocks followed by 2D CNN, global average pooling, small MLP classifier.
- Shots: 1, 5, 10 samples per class.
- Seeds: 0, 1, 2, 3, 4.
- Patch size: 19.
- PCA bands: 30.
- Epochs/patience: 200 / 30.
- PCA fit scope: full_image_unsupervised.
- Trainable parameters: 99,488.

## Aggregated Results

| dataset      |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |   train_size |   validation_size |   test_size |
|:-------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|-------------:|------------------:|------------:|
| indian_pines |      1 |      5 |     41.49 |     7.23 |     60.38 |     1.59 |        36.02 |        7.03 |           37.75 |           3    |              40.25 |              8.02 |              78.8 |                  99488 |           16 |               147 |       10086 |
| indian_pines |      5 |      5 |     72.02 |     1.67 |     82.66 |     1.45 |        68.66 |        1.9  |           63.81 |           2.17 |              73.27 |              2.06 |             118   |                  99488 |           80 |               145 |       10024 |
| indian_pines |     10 |      5 |     80.12 |     2.56 |     88.64 |     0.94 |        77.64 |        2.79 |           71.53 |           2.02 |              80.87 |              2.7  |              85   |                  99488 |          160 |               142 |        9947 |

## Per-run Results

|   shot |   seed |    OA |    AA |   Kappa |   Macro-F1 |   Weighted-F1 |   best_epoch |   train_size |   validation_size |   test_size |
|-------:|-------:|------:|------:|--------:|-----------:|--------------:|-------------:|-------------:|------------------:|------------:|
|      1 |      0 | 53.91 | 61.43 |   48.39 |      43.46 |         54.17 |           80 |           16 |               147 |       10086 |
|      1 |      1 | 44.45 | 62.94 |   38.45 |      36.69 |         42.62 |          102 |           16 |               147 |       10086 |
|      1 |      2 | 37.77 | 59.56 |   32.38 |      34.58 |         37.28 |          105 |           16 |               147 |       10086 |
|      1 |      3 | 38.54 | 59.49 |   33    |      36.97 |         37.13 |           42 |           16 |               147 |       10086 |
|      1 |      4 | 32.8  | 58.49 |   27.9  |      37.04 |         30.06 |           65 |           16 |               147 |       10086 |
|      5 |      0 | 73.34 | 83.4  |   70.26 |      65.84 |         75.1  |          120 |           80 |               145 |       10024 |
|      5 |      1 | 73.35 | 83.78 |   70.1  |      64.74 |         73.87 |          125 |           80 |               145 |       10024 |
|      5 |      2 | 69.02 | 79.82 |   65.21 |      59.74 |         69.25 |          112 |           80 |               145 |       10024 |
|      5 |      3 | 71.35 | 82.87 |   68.03 |      65.2  |         73.89 |          141 |           80 |               145 |       10024 |
|      5 |      4 | 73.02 | 83.43 |   69.7  |      63.5  |         74.23 |           92 |           80 |               145 |       10024 |
|     10 |      0 | 78.48 | 88.39 |   75.88 |      70.33 |         79.32 |           88 |          160 |               142 |        9947 |
|     10 |      1 | 76.62 | 86.98 |   73.82 |      68.6  |         77.05 |           71 |          160 |               142 |        9947 |
|     10 |      2 | 83.11 | 89.66 |   80.93 |      74.15 |         84.16 |           76 |          160 |               142 |        9947 |
|     10 |      3 | 79.42 | 88.84 |   76.83 |      71.21 |         80.14 |           90 |          160 |               142 |        9947 |
|     10 |      4 | 82.99 | 89.35 |   80.75 |      73.39 |         83.7  |          100 |          160 |               142 |        9947 |

## Interpretation

Performance improves monotonically from 1-shot to 10-shot. The 1-shot setting has high OA variance and low Macro-F1, which is expected under strict all-way few-shot training. The 5-shot and 10-shot settings are more reliable baselines for comparing HybridSN-small + quantum bottleneck or other QNN variants.

## Source Artifacts

- Source run directory: `result/hybridsn_small_fewshot_3datasets`
- Filtered all-runs CSV: `result/hybridsn_small_fewshot_indian_pines_only/metrics/all_runs_indian_pines.csv`
- Per-run metrics JSON files: 15 files in `result/hybridsn_small_fewshot_3datasets/metrics`
