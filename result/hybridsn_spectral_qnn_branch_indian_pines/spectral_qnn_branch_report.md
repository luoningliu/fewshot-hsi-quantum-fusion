# HybridSN + Spectral Branch: Indian Pines

This experiment freezes the tuned HybridSN encoder embedding and adds a center-pixel PCA spectral branch. The QNN branch is therefore moved earlier than the previous final-head-only QNN.

## Mean Validation Results

| model                     |   runs |   parameters_mean |   time_seconds_mean |   best_val_oa_mean |   best_val_oa_std |   best_val_aa_mean |   best_val_aa_std |   best_val_macro_f1_mean |   best_val_macro_f1_std |
|:--------------------------|-------:|------------------:|--------------------:|-------------------:|------------------:|-------------------:|------------------:|-------------------------:|------------------------:|
| embedding_mlp             |      3 |              9552 |                0.7  |              99.09 |              0.12 |              99.38 |              0.09 |                    99.41 |                    0.08 |
| spectral_mlp_fusion       |      3 |              2772 |                0.77 |              98.89 |              0.05 |              99.17 |              0.06 |                    97.94 |                    0.03 |
| spectral_qnn_fusion       |      3 |              2596 |               35.17 |              98.86 |              0.05 |              99.15 |              0.04 |                    97.92 |                    0.01 |
| embedding_linear          |      3 |              2320 |                0.46 |              98.83 |              0.08 |              99.11 |              0.08 |                    97.85 |                    0.07 |
| spectral_gated_qnn_fusion |      3 |              3071 |               33.45 |              98.76 |              0.05 |              99.07 |              0.06 |                    97.81 |                    0.08 |

## Best Selected Test Result

| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| embedding_mlp | 98.72 | 95.39 | 98.54 | 96.31 | 98.71 |
| Tuned full HybridSN reference | 98.80 | 97.27 | 98.64 | 97.19 | 98.81 |

## Per-Model Test Results

| model                     | selected_by                      |   best_seed |   best_val_macro_f1 |   best_val_oa |   best_val_aa |    OA |    AA |   Kappa |   Macro-F1 |   Weighted-F1 |
|:--------------------------|:---------------------------------|------------:|--------------------:|--------------:|--------------:|------:|------:|--------:|-----------:|--------------:|
| spectral_qnn_fusion       | validation_Macro-F1 within model |           1 |               97.93 |         98.83 |         99.1  | 98.91 | 96.22 |   98.76 |      97.41 |         98.91 |
| embedding_linear          | validation_Macro-F1 within model |           0 |               97.9  |         98.93 |         99.21 | 98.79 | 95.58 |   98.62 |      96.75 |         98.78 |
| spectral_mlp_fusion       | validation_Macro-F1 within model |           2 |               97.97 |         98.93 |         99.21 | 98.82 | 95.31 |   98.65 |      96.67 |         98.8  |
| spectral_gated_qnn_fusion | validation_Macro-F1 within model |           2 |               97.92 |         98.83 |         99.14 | 98.82 | 95.28 |   98.65 |      96.5  |         98.8  |
| embedding_mlp             | validation_Macro-F1 within model |           1 |               99.51 |         99.22 |         99.52 | 98.72 | 95.39 |   98.54 |      96.31 |         98.71 |

## Interpretation

A spectral QNN branch is useful only if it beats the embedding-only probes or at least improves validation Macro-F1 at a comparable parameter budget. Otherwise the current evidence still favors classical heads on frozen HybridSN features.
