# Spectral QNN Gated Fusion + prototype

## Configuration

- datasets: ['salinas']
- shots: [10]
- seeds: [0]
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
| salinas   | spectral_qnn_gated_prototype |     10 |      1 |      84.8 |        0 |     92.95 |        0 |        82.96 |           0 |           92.12 |              0 |              82.92 |                 0 |                 9 |                   2176 |


## Failed Runs

None.
