# Spectral QNN Gated Fusion + prototype

## Configuration

- datasets: ['salinas']
- shots: [5]
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
| salinas   | spectral_qnn_gated_prototype |      5 |      2 |     85.73 |     0.54 |     93.33 |     0.89 |        84.14 |        0.57 |           92.93 |           0.92 |              85.77 |              0.45 |              23.5 |                   2176 |


## Failed Runs

None.
