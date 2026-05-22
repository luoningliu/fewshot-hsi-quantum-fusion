# Spectral QNN Gated Fusion + prototype

## Configuration

- datasets: ['salinas']
- shots: [10]
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
| salinas   | spectral_qnn_gated_prototype |     10 |      2 |     91.49 |     3.55 |      95.8 |     1.48 |        90.51 |        3.97 |           95.19 |            1.5 |              91.35 |              3.73 |               7.5 |                   2176 |


## Failed Runs

None.
