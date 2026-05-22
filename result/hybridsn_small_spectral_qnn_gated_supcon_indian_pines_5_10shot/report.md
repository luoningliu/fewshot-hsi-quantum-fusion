# Spectral QNN Gated Fusion + supcon

## Configuration

- datasets: ['indian_pines']
- shots: [5, 10]
- seeds: [0, 1, 2, 3, 4]
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
| indian_pines | spectral_qnn_gated_supcon |      5 |      5 |     72.06 |     1.42 |     82.55 |     0.6  |        68.69 |        1.57 |           64.04 |           1.44 |              72.91 |              1.97 |              22.2 |                   2176 |
| indian_pines | spectral_qnn_gated_supcon |     10 |      5 |     81.07 |     2.32 |     89.05 |     1.38 |        78.7  |        2.54 |           72.26 |           2.08 |              81.79 |              2.33 |              18.2 |                   2176 |


## Failed Runs

None.
