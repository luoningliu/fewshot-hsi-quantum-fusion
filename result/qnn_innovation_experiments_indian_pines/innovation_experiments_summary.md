# QNN Innovation Experiments: Indian Pines

This summary consolidates three targeted experiments: moving QNN into a center spectral branch, evaluating heads under spatially isolated split, and comparing QNN heads under a parameter-efficiency protocol.

## Experiment 1: HybridSN + Spectral Branch

| experiment                   | model                     |   runs |   parameters_mean |   time_seconds_mean |   best_val_oa_mean |   best_val_aa_mean |   best_val_macro_f1_mean |
|:-----------------------------|:--------------------------|-------:|------------------:|--------------------:|-------------------:|-------------------:|-------------------------:|
| spectral_branch_random_split | embedding_mlp             |      3 |              9552 |                0.7  |              99.09 |              99.38 |                    99.41 |
| spectral_branch_random_split | spectral_mlp_fusion       |      3 |              2772 |                0.77 |              98.89 |              99.17 |                    97.94 |
| spectral_branch_random_split | spectral_qnn_fusion       |      3 |              2596 |               35.17 |              98.86 |              99.15 |                    97.92 |
| spectral_branch_random_split | embedding_linear          |      3 |              2320 |                0.46 |              98.83 |              99.11 |                    97.85 |
| spectral_branch_random_split | spectral_gated_qnn_fusion |      3 |              3071 |               33.45 |              98.76 |              99.07 |                    97.81 |

### Per-Model Test Metrics

| model                     | selected_by                      |   best_seed |   best_val_macro_f1 |   best_val_oa |   best_val_aa |    OA |    AA |   Kappa |   Macro-F1 |   Weighted-F1 |
|:--------------------------|:---------------------------------|------------:|--------------------:|--------------:|--------------:|------:|------:|--------:|-----------:|--------------:|
| spectral_qnn_fusion       | validation_Macro-F1 within model |           1 |               97.93 |         98.83 |         99.1  | 98.91 | 96.22 |   98.76 |      97.41 |         98.91 |
| embedding_linear          | validation_Macro-F1 within model |           0 |               97.9  |         98.93 |         99.21 | 98.79 | 95.58 |   98.62 |      96.75 |         98.78 |
| spectral_mlp_fusion       | validation_Macro-F1 within model |           2 |               97.97 |         98.93 |         99.21 | 98.82 | 95.31 |   98.65 |      96.67 |         98.8  |
| spectral_gated_qnn_fusion | validation_Macro-F1 within model |           2 |               97.92 |         98.83 |         99.14 | 98.82 | 95.28 |   98.65 |      96.5  |         98.8  |
| embedding_mlp             | validation_Macro-F1 within model |           1 |               99.51 |         99.22 |         99.52 | 98.72 | 95.39 |   98.54 |      96.31 |         98.71 |

## Experiment 2: Spatial Split Head Comparison

| experiment                    | model                           |   runs | parameters_mean   |   time_seconds_mean |   best_val_oa_mean |   best_val_aa_mean |   best_val_macro_f1_mean |
|:------------------------------|:--------------------------------|-------:|:------------------|--------------------:|-------------------:|-------------------:|-------------------------:|
| spatial_split_head_comparison | linear_probe                    |      1 |                   |                0.71 |              49.66 |              17.71 |                    11    |
| spatial_split_head_comparison | mlp_h64                         |      1 |                   |                0.84 |              56.58 |              19.11 |                    14.16 |
| spatial_split_head_comparison | residual_qnn_q4_l1_linear       |      1 |                   |               44.88 |              52.3  |              18.35 |                    12.18 |
| spatial_split_head_comparison | gated_residual_qnn_q4_l1_linear |      1 |                   |               50.35 |              50.77 |              18.02 |                    11.25 |

## Experiment 3: Parameter Efficiency

| experiment                     | model                           |   runs |   parameters_mean |   time_seconds_mean |   best_val_oa_mean |   best_val_aa_mean |   best_val_macro_f1_mean |
|:-------------------------------|:--------------------------------|-------:|------------------:|--------------------:|-------------------:|-------------------:|-------------------------:|
| parameter_matched_random_split | mlp_h64                         |      3 |              9552 |                0.72 |              99.09 |              99.38 |                    99.41 |
| parameter_matched_random_split | residual_qnn_q4_l1_linear       |      3 |              2928 |               49.84 |              98.37 |              98.92 |                    98.98 |
| parameter_matched_random_split | gated_residual_qnn_q4_l1_linear |      3 |              3313 |               45.68 |              98.31 |              98.33 |                    98.51 |
| parameter_matched_random_split | mlp_h128                        |      3 |             18832 |                0.69 |              98.89 |              98.72 |                    98.48 |
| parameter_matched_random_split | bottleneck_b16                  |      3 |              2592 |                0.85 |              98.37 |              97.78 |                    98.41 |
| parameter_matched_random_split | linear_probe                    |      3 |              2320 |                0.45 |              98.83 |              99.11 |                    97.85 |
| parameter_matched_random_split | bottleneck_b8                   |      3 |              1432 |                0.83 |              97.27 |              84.14 |                    84.44 |
| parameter_matched_random_split | bottleneck_b4                   |      3 |               852 |                0.82 |              76.68 |              57.82 |                    54.4  |

## Selected Test Metrics

| Model / Protocol | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Tuned HybridSN random split | 98.80 | 97.27 | 98.64 | 97.19 | 98.81 |
| Best spectral branch random split | 98.72 | 95.39 | 98.54 | 96.31 | 98.71 |
| Spectral QNN branch random split | 98.91 | 96.22 | 98.76 | 97.41 | 98.91 |
| Best parameter-efficiency head | 98.72 | 95.39 | 98.54 | 96.31 | 98.71 |
| HybridSN spatial split | 33.16 | 19.36 | 21.46 | 13.03 | 24.02 |
| Best spatial-split head | 35.90 | 20.68 | 25.68 | 13.78 | 26.48 |

## Conclusion

The spectral QNN branch is the most promising QNN placement so far: when selected within its own model family by validation Macro-F1, it gives the best test Macro-F1 in the spectral-branch pilot. However, its validation mean is still below the embedding MLP probe, so this should be treated as a positive signal that needs more seeds and a stricter confirmation run, not yet as a settled superiority claim. Under spatial split, QNN heads still drop strongly and do not beat the classical MLP head. The parameter-efficiency experiment shows that QNN heads can approach MLP performance with fewer trainable parameters, but they are much slower.
