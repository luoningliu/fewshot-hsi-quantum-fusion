# Low-Label HybridSN Head Regularization: Indian Pines

Frozen tuned HybridSN embeddings are reused. Only a stratified fraction of the original training split is used for head training; validation/test splits are unchanged.

## Validation Mean ± Std

|   train_fraction | model              |   runs |   train_size_mean |   best_val_oa_mean |   best_val_oa_std |   best_val_aa_mean |   best_val_aa_std |   best_val_macro_f1_mean |   best_val_macro_f1_std |
|-----------------:|:-------------------|-------:|------------------:|-------------------:|------------------:|-------------------:|------------------:|-------------------------:|------------------------:|
|             0.01 | linear_probe       |      3 |                16 |              98.05 |              0.08 |              98.56 |              0.16 |                    95.76 |                    1.29 |
|             0.01 | mlp_probe          |      3 |                16 |              97.85 |              0.41 |              98.49 |              0.3  |                    95.47 |                    0.69 |
|             0.01 | gated_residual_qnn |      3 |                16 |              95.93 |              0.47 |              96.73 |              0.56 |                    92.08 |                    1.54 |
|             0.01 | residual_qnn       |      3 |                16 |              93.69 |              0.81 |              94.29 |              2.06 |                    89.49 |                    0.92 |
|             0.03 | linear_probe       |      3 |                31 |              97.59 |              0.49 |              97.95 |              0.41 |                    94.89 |                    1.28 |
|             0.03 | mlp_probe          |      3 |                31 |              97.27 |              0.35 |              97.48 |              0.39 |                    93.95 |                    0.62 |
|             0.03 | residual_qnn       |      3 |                31 |              96.94 |              0.83 |              95.02 |              0.78 |                    92.78 |                    1.94 |
|             0.03 | gated_residual_qnn |      3 |                31 |              96.1  |              1.41 |              94.33 |              2.43 |                    91.75 |                    1.74 |
|             0.05 | mlp_probe          |      3 |                51 |              98.5  |              0.36 |              98.3  |              0.71 |                    97.91 |                    1.19 |
|             0.05 | linear_probe       |      3 |                51 |              98.34 |              0.21 |              98.59 |              0.3  |                    97.89 |                    0.69 |
|             0.05 | gated_residual_qnn |      3 |                51 |              97.43 |              0.3  |              96.95 |              0.77 |                    96.12 |                    1.17 |
|             0.05 | residual_qnn       |      3 |                51 |              97.17 |              0.62 |              96.57 |              1.19 |                    95.81 |                    1.24 |
|             0.1  | mlp_probe          |      3 |               102 |              98.57 |              0.36 |              98.39 |              0.87 |                    97.77 |                    0.12 |
|             0.1  | linear_probe       |      3 |               102 |              98.21 |              0.28 |              97.94 |              1.26 |                    97.49 |                    0.22 |
|             0.1  | residual_qnn       |      3 |               102 |              97.63 |              0.44 |              96.68 |              0.29 |                    96.93 |                    0.82 |
|             0.1  | gated_residual_qnn |      3 |               102 |              97.46 |              0.35 |              94    |              3.45 |                    94.54 |                    3.09 |

## Best Selected Test Result

| Model | Fraction | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| mlp_probe | 0.05 | 98.56 | 97.70 | 98.36 | 97.16 | 98.56 |

## Interpretation

Use this table to test whether QNN heads act as useful regularizers when very few labeled samples are available for the classifier head.
