# Spectral QNN Gated Fusion + prototype

## Configuration

- datasets: ['salinas']
- shots: [10]
- seeds: [1, 2]
- loss_mode: prototype
- metric_weight: 0.2
- temperature: 0.2
- qubits: 6
- qnn_layers: 1
- entanglement: linear
- gate_mode: classwise

## Summary

| dataset   | model                        |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:----------|:-----------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| salinas   | spectral_qnn_gated_prototype |     10 |      2 |     92.92 |     0.09 |     96.63 |     0.73 |        92.13 |        0.12 |           96.16 |           0.53 |              92.91 |              0.15 |                19 |                   2176 |


## Failed Runs

None.
