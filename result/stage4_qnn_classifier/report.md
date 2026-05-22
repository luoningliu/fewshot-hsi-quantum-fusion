# Stage 4 QNN Classifier Report

| dataset_id       | model               |     OA |     AA |   Kappa |   Macro-F1 |   Weighted-F1 |   training_time_seconds | device   | note                  |   subset_train |   subset_validation |   subset_test |
|:-----------------|:--------------------|-------:|-------:|--------:|-----------:|--------------:|------------------------:|:---------|:----------------------|---------------:|--------------------:|--------------:|
| indian_pines     | hybridsn_linear     |  70.38 |  53.67 |   66.1  |      52.65 |         69.05 |               44.4267   | cpu      | nan                   |            nan |                 nan |           nan |
| indian_pines     | hybridsn_mlp        |  73.38 |  56.29 |   69.55 |      57.15 |         72.46 |                0.251027 | cpu      | nan                   |            nan |                 nan |           nan |
| indian_pines     | hybridsn_bottleneck |  46.74 |  23.57 |   38.74 |      18.53 |         40.31 |                0.193072 | cpu      | nan                   |            nan |                 nan |           nan |
| indian_pines     | hybridsn_qnn        |  20.44 |  17.25 |   14.08 |       7.08 |          8.4  |               11.0127   | cpu      | qnn_stratified_subset |            363 |                 363 |          1350 |
| pavia_university | hybridsn_linear     |  98.07 |  97    |   97.45 |      96.83 |         98.08 |              187.559    | cpu      | nan                   |            nan |                 nan |           nan |
| pavia_university | hybridsn_mlp        |  98.68 |  97.8  |   98.24 |      98.01 |         98.67 |                0.800676 | cpu      | nan                   |            nan |                 nan |           nan |
| pavia_university | hybridsn_bottleneck |  85.23 |  54.37 |   79.89 |      49.94 |         79.68 |                0.736098 | cpu      | nan                   |            nan |                 nan |           nan |
| pavia_university | hybridsn_qnn        | nan    | nan    |  nan    |     nan    |        nan    |              nan        | cpu      | skipped_in_cpu_pilot  |            nan |                 nan |           nan |
| salinas          | hybridsn_linear     |  96.69 |  98.24 |   96.31 |      98.21 |         96.71 |              240.677    | cpu      | nan                   |            nan |                 nan |           nan |
| salinas          | hybridsn_mlp        |  97.57 |  98.85 |   97.3  |      98.87 |         97.57 |                1.32021  | cpu      | nan                   |            nan |                 nan |           nan |
| salinas          | hybridsn_bottleneck |  73.95 |  55.95 |   70.15 |      53.57 |         66.42 |                0.921363 | cpu      | nan                   |            nan |                 nan |           nan |
| salinas          | hybridsn_qnn        | nan    | nan    |  nan    |     nan    |        nan    |              nan        | cpu      | skipped_in_cpu_pilot  |            nan |                 nan |           nan |
