# Spectral QNN Gated Fusion + prototype

## Configuration

- datasets: ['pavia_university']
- shots: [10]
- seeds: [0, 1, 2]
- loss_mode: prototype
- metric_weight: 0.2
- temperature: 0.2
- qubits: 6
- qnn_layers: 1
- entanglement: linear
- gate_mode: classwise

## Summary

| dataset          | model                        |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-----------------|:-----------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| pavia_university | spectral_qnn_gated_prototype |     10 |      3 |     87.63 |     2.16 |     90.53 |     1.67 |        84.11 |        2.64 |           87.07 |           1.12 |              88.15 |              1.89 |             14.67 |                   1455 |


## Failed Runs

None.
