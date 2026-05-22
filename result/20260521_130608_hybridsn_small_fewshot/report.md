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

- datasets: ['indian_pines']
- shots: [1]
- seeds: [0]
- patch_size: 19
- pca_bands: 30
- dropout: 0.4
- lr: 0.001
- weight_decay: 0.0001
- epochs: 2
- patience: 2

## Aggregated Results

| dataset      |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| indian_pines |      1 |      1 |      8.12 |        0 |      6.25 |        0 |            0 |           0 |            0.94 |              0 |               1.22 |                 0 |                 1 |                  99488 |


## Failed or Skipped Runs

None.

## Initial Interpretation

HybridSN-small preserves the key HybridSN design: 3D spectral-spatial feature extraction followed by 2D spatial feature learning. Compared with the original HybridSN, this version replaces the large Flatten-Dense classifier with Global Average Pooling and a small classifier, making it more suitable for strict few-shot settings.

The 1-shot results should be interpreted with caution because the variance can be high. The 5-shot and 10-shot settings are more informative for evaluating whether a lightweight spectral-spatial CNN can learn stable representations under limited labels.

This experiment should serve as the classical lightweight baseline for later comparison with HybridSN-small + classical bottleneck, HybridSN-small + quantum bottleneck, and quantum prototype network.
