# Spectral QNN Gated Fusion + prototype

## Configuration

- datasets: ['pavia_university']
- shots: [10]
- seeds: [3]
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
| pavia_university | spectral_qnn_gated_prototype |     10 |      1 |     86.18 |        0 |     89.87 |        0 |        82.26 |           0 |           87.36 |              0 |              86.55 |                 0 |                 8 |                   1455 |


## Failed Runs

None.
