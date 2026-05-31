# Few-shot QNN paired significance test

## Scope

- Pavia University 10-shot and Salinas 10-shot: ConfidenceGuard-C + SupCon vs HybridSN-small.
- Indian Pines 5-shot and 10-shot: Spectral QNN + SupCon vs HybridSN-small.
- Paired by identical random seed, n=5 for each dataset-shot setting.
- Tests: paired t-test on seed deltas, Wilcoxon signed-rank test, sign test, bootstrap 95% CI of mean delta, Cohen's dz.

## Main Results

### Indian Pines 5-shot

| metric | mean_delta | ci95_low_bootstrap | ci95_high_bootstrap | positive_seeds | negative_seeds | paired_t_p_greater | wilcoxon_p_greater | wilcoxon_p_less | cohen_dz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.0023 | -0.0088 | 0.0133 | 3 | 2 | 0.3608 | 0.3125 | 0.7812 | 0.1710 |
| oa | 0.0004 | -0.0094 | 0.0096 | 2 | 3 | 0.4751 | 0.5938 | 0.5000 | 0.0297 |
| weighted_f1 | -0.0036 | -0.0137 | 0.0048 | 2 | 3 | 0.7216 | 0.5938 | 0.5000 | -0.2863 |

Seed deltas:

| seed | delta_oa | delta_macro_f1 | delta_weighted_f1 |
| --- | --- | --- | --- |
| 0 | -0.0032 | 0.0013 | -0.0046 |
| 1 | -0.0186 | -0.0165 | -0.0237 |
| 2 | 0.0120 | 0.0214 | 0.0057 |
| 3 | -0.0007 | -0.0013 | -0.0029 |
| 4 | 0.0124 | 0.0068 | 0.0077 |

### Indian Pines 10-shot

| metric | mean_delta | ci95_low_bootstrap | ci95_high_bootstrap | positive_seeds | negative_seeds | paired_t_p_greater | wilcoxon_p_greater | wilcoxon_p_less | cohen_dz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.0073 | -0.0058 | 0.0216 | 3 | 2 | 0.2104 | 0.3125 | 0.7812 | 0.4007 |
| oa | 0.0095 | -0.0010 | 0.0200 | 3 | 2 | 0.0964 | 0.1562 | 0.9062 | 0.6996 |
| weighted_f1 | 0.0092 | -0.0010 | 0.0193 | 3 | 2 | 0.1000 | 0.1562 | 0.9062 | 0.6856 |

Seed deltas:

| seed | delta_oa | delta_macro_f1 | delta_weighted_f1 |
| --- | --- | --- | --- |
| 0 | -0.0055 | -0.0116 | -0.0029 |
| 1 | 0.0256 | 0.0204 | 0.0262 |
| 2 | 0.0128 | 0.0030 | 0.0129 |
| 3 | 0.0180 | 0.0315 | 0.0155 |
| 4 | -0.0034 | -0.0067 | -0.0058 |

### Pavia University 10-shot

| metric | mean_delta | ci95_low_bootstrap | ci95_high_bootstrap | positive_seeds | negative_seeds | paired_t_p_greater | wilcoxon_p_greater | wilcoxon_p_less | cohen_dz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.0673 | 0.0219 | 0.1380 | 5 | 0 | 0.0588 | 0.0312 | 1.0000 | 0.8895 |
| oa | 0.0351 | 0.0080 | 0.0622 | 5 | 0 | 0.0461 | 0.0312 | 1.0000 | 0.9863 |
| weighted_f1 | 0.0369 | 0.0083 | 0.0703 | 5 | 0 | 0.0517 | 0.0312 | 1.0000 | 0.9404 |

Seed deltas:

| seed | delta_oa | delta_macro_f1 | delta_weighted_f1 |
| --- | --- | --- | --- |
| 0 | 0.0027 | 0.0035 | 0.0037 |
| 1 | 0.0775 | 0.1969 | 0.0935 |
| 2 | 0.0695 | 0.0654 | 0.0620 |
| 3 | 0.0091 | 0.0337 | 0.0088 |
| 4 | 0.0165 | 0.0372 | 0.0163 |

### Salinas 10-shot

| metric | mean_delta | ci95_low_bootstrap | ci95_high_bootstrap | positive_seeds | negative_seeds | paired_t_p_greater | wilcoxon_p_greater | wilcoxon_p_less | cohen_dz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.0068 | -0.0030 | 0.0195 | 3 | 2 | 0.1878 | 0.2188 | 0.8438 | 0.4455 |
| oa | -0.0100 | -0.0332 | 0.0095 | 2 | 3 | 0.7656 | 0.7812 | 0.3125 | -0.3575 |
| weighted_f1 | -0.0098 | -0.0334 | 0.0113 | 2 | 3 | 0.7559 | 0.7812 | 0.3125 | -0.3411 |

Seed deltas:

| seed | delta_oa | delta_macro_f1 | delta_weighted_f1 |
| --- | --- | --- | --- |
| 0 | -0.0533 | 0.0003 | -0.0537 |
| 1 | -0.0104 | -0.0077 | -0.0101 |
| 2 | -0.0155 | -0.0000 | -0.0156 |
| 3 | 0.0095 | 0.0095 | 0.0097 |
| 4 | 0.0195 | 0.0320 | 0.0209 |

## Interpretation

- With only five seeds, Wilcoxon/sign tests are conservative; results should be reported as paired evidence rather than definitive population-level proof unless p-values and bootstrap CI both support the claim.
- Pavia University 10-shot has consistent positive paired deltas across all seeds for the main metrics, supporting a robust positive QNN effect.
- Salinas 10-shot shows mixed seed signs: Macro-F1 trends positive, but OA and Weighted-F1 remain negative on average and are not positive-significant.
- Indian Pines 10-shot is positive on average but not statistically strong under n=5; Indian Pines 5-shot is essentially neutral.
