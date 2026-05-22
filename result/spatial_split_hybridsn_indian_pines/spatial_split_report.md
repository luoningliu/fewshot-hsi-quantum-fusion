# Spatially Disjoint HybridSN Pilot

Grid-block spatial split. Blocks are mutually exclusive across train/validation/test. This pilot uses fixed block ratios and does not force every class to appear in the training region, making it a strict spatial generalization diagnostic.

|   seed |   best_val_macro_f1 |   best_val_oa |   best_val_aa |   epochs_ran |   training_time_seconds |   train_size |   validation_size |   test_size |   pca_evr_sum |
|-------:|--------------------:|--------------:|--------------:|-------------:|------------------------:|-------------:|------------------:|------------:|--------------:|
|      0 |               16.28 |         60.75 |         19.77 |            8 |                 60.4965 |         1580 |              1633 |        7036 |      0.969597 |
|      1 |               12.4  |         98.09 |         12.43 |           10 |                 96.5854 |         2342 |               940 |        6967 |      0.959971 |

## Best Test

OA=33.16, AA=19.36, Macro-F1=13.03, Weighted-F1=24.02

This should be compared against random pixel split HybridSN OA=98.80 / Macro-F1=97.19.
