# HybridSN-small Few-shot HSI Classification

## Purpose

Evaluate a lightweight HybridSN-style 3D-2D CNN as a classical few-shot baseline for hyperspectral image classification.

## Architecture

HybridSN-small keeps the HybridSN hierarchy: three 3D convolution blocks for spectral-spatial features, reshape spectral depth into channels, one 2D convolution block for spatial abstraction, global average pooling, and a small MLP classifier.

The large original Flatten-Dense classifier is replaced by Global Average Pooling plus Linear(32 -> 64 -> classes), sharply reducing dense parameters.

## Few-shot Protocol

For each dataset, shot, and seed, the support set contains exactly K labeled samples per class. Validation uses min(10 samples per class, 20% of remaining samples per class), and all remaining labeled pixels are used for testing. Splits are stratified and saved under `split_indices/`.

PCA is fitted on the full image without labels because strict 1-shot support sets cannot fit 20/30 PCA components. This is an unsupervised preprocessing step and is documented as `pca_fit_scope=full_image_unsupervised` in each metrics JSON.

## Hyperparameters

- datasets: ['pavia_university', 'salinas']
- shots: [5, 10]
- seeds: [0, 1, 2, 3, 4]
- patch_size: 19
- pca_bands: 30
- dropout: 0.4
- lr: 0.001
- weight_decay: 0.0001
- epochs: 120
- patience: 20

## Aggregated Results

| dataset          |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university |      5 |      5 |     75.74 |     2.89 |     78.32 |     3.84 |        69.32 |        3.06 |           71.21 |           4.46 |              75.94 |              2.89 |              78.8 |                  99033 |
| pavia_university |     10 |      5 |     82.26 |     4.92 |     84.31 |     5.82 |        77.31 |        6.2  |           79.2  |           8.14 |              82.8  |              5.19 |              60.2 |                  99033 |
| salinas          |      5 |      5 |     86.93 |     7.11 |     90.9  |     8.03 |        85.42 |        8.04 |           89.67 |           9.21 |              85.91 |              9.25 |              76.8 |                  99488 |
| salinas          |     10 |      5 |     93.6  |     1.59 |     96.06 |     1.46 |        92.88 |        1.76 |           95.44 |           1.55 |              93.62 |              1.57 |              58.8 |                  99488 |


## Failed or Skipped Runs

None.

## Initial Interpretation

HybridSN-small preserves the key HybridSN design: 3D spectral-spatial feature extraction followed by 2D spatial feature learning. Compared with the original HybridSN, this version replaces the large Flatten-Dense classifier with Global Average Pooling and a small classifier, making it more suitable for strict few-shot settings.

The 1-shot results should be interpreted with caution because the variance can be high. The 5-shot and 10-shot settings are more informative for evaluating whether a lightweight spectral-spatial CNN can learn stable representations under limited labels.

This experiment should serve as the classical lightweight baseline for later comparison with HybridSN-small + classical bottleneck, HybridSN-small + quantum bottleneck, and quantum prototype network.
