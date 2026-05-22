# Spatial Split QNN Head Pilot

Split sizes: train=1580, validation=1633, test=7036

| run_id                          |   best_val_macro_f1 |   best_val_oa |   best_val_aa |   epochs_ran |   training_time_seconds |
|:--------------------------------|--------------------:|--------------:|--------------:|-------------:|------------------------:|
| linear_probe                    |               11    |         49.66 |         17.71 |           10 |                0.706782 |
| mlp_h64                         |               14.16 |         56.58 |         19.11 |           10 |                0.843119 |
| residual_qnn_q4_l1_linear       |               12.18 |         52.3  |         18.35 |            9 |               44.8782   |
| gated_residual_qnn_q4_l1_linear |               11.25 |         50.77 |         18.02 |           10 |               50.3486   |

## Best Test

mlp_h64: OA=35.90, AA=20.68, Macro-F1=13.78, Weighted-F1=26.48

The encoder is trained only on spatial train blocks, then frozen before head comparison.
