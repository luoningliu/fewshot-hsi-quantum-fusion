# Spectral QNN Gated Fusion + prototype

## Configuration

- datasets: ['pavia_university']
- shots: [5]
- seeds: [4]
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
| pavia_university | spectral_qnn_gated_prototype |      5 |      1 |     78.12 |        0 |     79.35 |        0 |        71.91 |           0 |           75.75 |              0 |              78.54 |                 0 |                14 |                   1455 |


## Failed Runs

None.
