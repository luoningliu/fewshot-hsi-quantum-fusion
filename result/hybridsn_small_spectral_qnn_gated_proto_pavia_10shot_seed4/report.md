# Spectral QNN Gated Fusion + prototype

## Configuration

- datasets: ['pavia_university']
- shots: [10]
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
| pavia_university | spectral_qnn_gated_prototype |     10 |      1 |     82.52 |        0 |     85.93 |        0 |        77.82 |           0 |           82.06 |              0 |              83.67 |                 0 |                10 |                   1455 |


## Failed Runs

None.
