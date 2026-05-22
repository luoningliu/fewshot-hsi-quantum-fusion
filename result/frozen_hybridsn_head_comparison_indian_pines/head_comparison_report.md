# Frozen HybridSN Head Comparison: Indian Pines

All heads use the same frozen tuned HybridSN encoder embeddings. Head selection uses validation Macro-F1.

## Mean Validation Results

| model              |   runs |   best_val_oa_mean |   best_val_oa_std |   best_val_aa_mean |   best_val_aa_std |   best_val_macro_f1_mean |   best_val_macro_f1_std |
|:-------------------|-------:|-------------------:|------------------:|-------------------:|------------------:|-------------------------:|------------------------:|
| gated_residual_qnn |      5 |              98.54 |              0.38 |              98.38 |              0.81 |                    98.63 |                    0.52 |
| mlp_probe          |      5 |              98.87 |              0.15 |              98.61 |              0.81 |                    98.59 |                    0.52 |
| residual_qnn       |      5 |              98.61 |              0.47 |              98.17 |              0.58 |                    98.57 |                    0.35 |
| linear_probe       |      5 |              98.73 |              0.14 |              98.46 |              0.8  |                    97.93 |                    0.11 |

## Best Selected Test Result

| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| mlp_probe | 98.77 | 95.79 | 98.60 | 96.85 | 98.76 |
| Tuned full HybridSN reference | 98.80 | 97.27 | 98.64 | 97.19 | 98.81 |

## Interpretation

This is the fair head-only comparison. If QNN heads do not exceed MLP/Linear on the same frozen embedding, the QNN contribution should be framed as compact quantum-augmented matching rather than superior accuracy.
