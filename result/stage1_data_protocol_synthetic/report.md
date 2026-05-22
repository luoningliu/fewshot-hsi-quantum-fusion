# Stage 1 Data Protocol Report

- Dataset: LCZ42
- Samples: 60
- Input shape: [8, 8, 4]
- Cities: Berlin, Munich, Cologne
- Bands: B2, B3, B4, B8
- Processed file: `data/lcz42/processed/synthetic_lcz42_8x8_4band_stage1.npz`

## Class Distribution

|   class_id | class_name                      |   count |
|-----------:|:--------------------------------|--------:|
|          0 | Compact built-up area           |      12 |
|          1 | Open built-up area              |      12 |
|          2 | Large low-rise / Heavy industry |      12 |
|          3 | Vegetation                      |      12 |
|          4 | Water                           |      12 |

## Band Statistics

| band   |     mean |      std |         min |      max |   norm_mean |   norm_std |
|:-------|---------:|---------:|------------:|---------:|------------:|-----------:|
| B2     | 0.410655 | 0.279027 | -0.009111   | 0.900425 |    0.410655 |   0.279027 |
| B3     | 0.409753 | 0.278832 | -0.00628803 | 0.898773 |    0.409753 |   0.278832 |
| B4     | 0.409852 | 0.279303 | -0.00253359 | 0.90849  |    0.409852 |   0.279303 |
| B8     | 0.409116 | 0.27904  | -0.00588115 | 0.901725 |    0.409116 |   0.27904  |

## Outputs

- `data_statistics.json`
- `class_distribution.csv`
- `band_statistics.csv`
- `band_mean.json`
- `band_std.json`
- `split_holdout.json`
- `split_fold_1.json` through `split_fold_5.json`
