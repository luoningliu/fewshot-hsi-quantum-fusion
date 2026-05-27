# Salinas 10-shot 负迁移分析

## Main finding

Salinas 10-shot 中 QNN + Prototype 相对 HybridSN-small 出现负迁移。该负迁移在 OA 和 Macro-F1 上均可见，且与 logit margin 中 negative margin rate 上升一致。

## Evidence table
| dataset   |   shot | comparison                                                   | metric      |   baseline_mean |   qnn_mean |   mean_delta |   std_delta |   ci95_low |   ci95_high |   paired_t_pvalue |   wilcoxon_pvalue |   cohen_d | is_significant_p005   | interpretation    |   n_paired_seeds |
|:----------|-------:|:-------------------------------------------------------------|:------------|----------------:|-----------:|-------------:|------------:|-----------:|------------:|------------------:|------------------:|----------:|:----------------------|:------------------|-----------------:|
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | OA          |        0.936018 |   0.907216 |  -0.0288019  |  0.0273252  | -0.0627306 |  0.00512686 |         0.0779256 |            0.125  | -1.05404  | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | AA          |        0.960622 |   0.95563  |  -0.00499127 |  0.0093773  | -0.0166347 |  0.00665218 |         0.299779  |            0.3125 | -0.532272 | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Kappa       |        0.928798 |   0.896468 |  -0.03233    |  0.030909   | -0.0707086 |  0.00604853 |         0.0794763 |            0.125  | -1.04598  | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Macro-F1    |        0.954418 |   0.949634 |  -0.00478366 |  0.00843171 | -0.015253  |  0.0056857  |         0.273382  |            0.3125 | -0.567341 | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Weighted-F1 |        0.936166 |   0.902897 |  -0.0332687  |  0.0344057  | -0.0759891 |  0.00945161 |         0.0966534 |            0.125  | -0.966954 | False                 | negative_transfer |                5 |

## Per-class degradation
| analysis_type           |   class_id | class_name                |      delta_f1 |   delta_recall | interpretation    |
|:------------------------|-----------:|:--------------------------|--------------:|---------------:|:------------------|
| per_class_f1_delta      |         14 | Vinyard_untrained         |  -0.178994    |   -0.180657    | degraded          |
| per_class_f1_delta      |          7 | Grapes_untrained          |  -0.0729679   |   -0.0499867   | degraded          |
| per_class_f1_delta      |         15 | Vinyard_vertical_trellis  |  -0.0024303   |    0.00257415  | near_neutral      |
| per_class_f1_delta      |          2 | Fallow                    |  -0.00168588  |    0.000102249 | near_neutral      |
| per_class_f1_delta      |         11 | Lettuce_romaine_5wk       |  -0.000742198 |    0.00461458  | near_neutral      |
| per_class_f1_delta      |          0 | Brocoli_green_weeds_1     |  -0.000553662 |   -0.000904977 | near_neutral      |
| per_class_f1_delta      |          1 | Brocoli_green_weeds_2     |  -0.000378218 |   -0.000485699 | near_neutral      |
| per_class_f1_delta      |          9 | Corn_senesced_green_weeds |   0.000380699 |   -0.000736648 | near_neutral      |
| per_class_f1_delta      |          8 | Soil_vinyard_develop      |   0.000687353 |    0           | near_neutral      |
| per_class_f1_delta      |         10 | Lettuce_romaine_4wk       |   0.00284536  |   -0.00400763  | near_neutral      |
| per_class_f1_delta      |          6 | Celery                    |   0.0110142   |    0.0198932   | near_neutral      |
| per_class_f1_delta      |          5 | Stubble                   |   0.0137087   |    0.00639756  | near_neutral      |
| per_class_f1_delta      |         13 | Lettuce_romaine_7wk       |   0.0216824   |    0.012381    | improved          |
| per_class_f1_delta      |         12 | Lettuce_romaine_6wk       |   0.0325764   |    0.0459821   | improved          |
| per_class_f1_delta      |          4 | Fallow_smooth             |   0.0397148   |    0.0616253   | improved          |
| per_class_f1_delta      |          3 | Fallow_rough_plow         |   0.0586032   |    0.00334789  | improved          |
| seedwise_OA_delta       |        nan | seed_0                    | nan           |  nan           | negative_transfer |
| seedwise_OA_delta       |        nan | seed_1                    | nan           |  nan           | negative_transfer |
| seedwise_OA_delta       |        nan | seed_2                    | nan           |  nan           | negative_transfer |
| seedwise_OA_delta       |        nan | seed_3                    | nan           |  nan           | improved          |
| seedwise_OA_delta       |        nan | seed_4                    | nan           |  nan           | negative_transfer |
| seedwise_Macro-F1_delta |        nan | seed_0                    |  -0.0158316   |  nan           | negative_transfer |
| seedwise_Macro-F1_delta |        nan | seed_1                    |  -0.00823525  |  nan           | negative_transfer |
| seedwise_Macro-F1_delta |        nan | seed_2                    |  -0.00676825  |  nan           | negative_transfer |
| seedwise_Macro-F1_delta |        nan | seed_3                    |   0.00576105  |  nan           | improved          |
| seedwise_Macro-F1_delta |        nan | seed_4                    |   0.00115572  |  nan           | improved          |

## Margin or gate evidence if available
| dataset   |   shot | model                    |   Macro-F1_mean |   separation_ratio_mean |   prototype_negative_margin_rate_mean |   mean_true_logit_margin_mean |   negative_logit_margin_rate_mean |
|:----------|-------:|:-------------------------|----------------:|------------------------:|--------------------------------------:|------------------------------:|----------------------------------:|
| salinas   |     10 | hybridsn_small           |        0.954418 |                 8.08233 |                             0.0571094 |                       2.3213  |                         0.0639819 |
| salinas   |     10 | spectral_qnn_gated_proto |        0.949634 |                 6.83662 |                             0.0718058 |                       2.23594 |                         0.0927837 |

Gate values were not saved in the current runs, so gate overuse cannot be directly verified.

## Final interpretation

负迁移更可能来自 Salinas 10-shot classical baseline 已接近饱和，额外 spectral QNN residual branch 在部分 seed/classes 上扰动了已有决策边界。该结论应保持谨慎，后续需要 gate values 和 branch contribution magnitude 进一步验证。
