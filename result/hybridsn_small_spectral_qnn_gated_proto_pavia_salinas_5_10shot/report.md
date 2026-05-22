# Spectral QNN Gated Fusion + prototype

## Configuration

- datasets: ['pavia_university']
- shots: [5]
- seeds: [0, 1, 2, 3]
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
| pavia_university | spectral_qnn_gated_prototype |      5 |      4 |     77.38 |     3.87 |     83.13 |     2.68 |        71.76 |        4.38 |           77.45 |           2.31 |              78.39 |              3.77 |             23.75 |                   1455 |


## Failed Runs

None.
