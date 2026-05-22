# Spectral QNN Gated Fusion + supcon

## Configuration

- datasets: ['indian_pines']
- shots: [5]
- seeds: [0]
- loss_mode: supcon
- metric_weight: 0.2
- temperature: 0.2
- qubits: 6
- qnn_layers: 1
- entanglement: linear
- gate_mode: classwise

## Summary

| dataset      | model                     |   shot |   runs |   mean_OA |   std_OA |   mean_AA |   std_AA |   mean_Kappa |   std_Kappa |   mean_Macro-F1 |   std_Macro-F1 |   mean_Weighted-F1 |   std_Weighted-F1 |   mean_best_epoch |   trainable_parameters |
|:-------------|:--------------------------|-------:|-------:|----------:|---------:|----------:|---------:|-------------:|------------:|----------------:|---------------:|-------------------:|------------------:|------------------:|-----------------------:|
| indian_pines | spectral_qnn_gated_supcon |      5 |      1 |     28.41 |        0 |     38.01 |        0 |        21.35 |           0 |           20.62 |              0 |              26.81 |                 0 |                 2 |                   2176 |


## Failed Runs

None.
