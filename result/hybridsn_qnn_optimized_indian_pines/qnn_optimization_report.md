# HybridSN-Based QNN Optimization Report

Frozen tuned HybridSN encoder is used to extract 128-d embeddings. Only classifier heads are trained.

## Best Test Metrics

| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| gated_residual_qnn | 98.61 | 95.02 | 98.42 | 96.35 | 98.59 |

## Runs

| run_id                    | model              |   learning_rate |   weight_decay |   best_val_macro_f1 |   best_val_oa |   best_val_aa |   epochs_ran |   training_time_seconds |   qubits |   layers | entanglement   |   angle_scale | gate_mode   |
|:--------------------------|:-------------------|----------------:|---------------:|--------------------:|--------------:|--------------:|-------------:|------------------------:|---------:|---------:|:---------------|--------------:|:------------|
| qnn_00_linear_probe       | linear_probe       |           0.003 |         0.0001 |            0.979323 |      0.988293 |      0.991435 |           12 |                0.738065 |      nan |      nan | nan            |     nan       | nan         |
| qnn_01_residual_qnn       | residual_qnn       |           0.003 |         0.0001 |            0.980184 |      0.985366 |      0.975075 |           12 |               38.9347   |        4 |        1 | linear         |       3.14159 | nan         |
| qnn_02_gated_residual_qnn | gated_residual_qnn |           0.003 |         0.0001 |            0.991131 |      0.986341 |      0.991124 |           12 |               38.9995   |        4 |        1 | linear         |       3.14159 | scalar      |

## Interpretation

This evaluates whether a QNN head can exploit the tuned HybridSN representation without changing the encoder. Compare against the tuned full HybridSN classifier before claiming an end-to-end improvement.
