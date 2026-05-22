# Spectral QNN Gated Fusion + prototype

## Configuration

- datasets: ['salinas']
- shots: [5]
- seeds: [3, 4]
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
| salinas   | spectral_qnn_gated_prototype |      5 |      2 |     90.06 |     0.72 |     94.75 |     0.14 |        88.98 |        0.77 |           93.99 |           0.05 |              90.09 |              0.75 |                20 |                   2176 |


## Failed Runs

None.
