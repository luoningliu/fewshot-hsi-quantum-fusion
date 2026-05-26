# Salinas 10-shot 负迁移分析

## Main finding

Salinas 10-shot 的负迁移主要发生在 QNN + Prototype Loss 配置：Macro-F1 相对 HybridSN-small 为 -0.0048，OA 为 -0.0288，Weighted-F1 为 -0.0333。QNN + SupCon Loss 将 Macro-F1 拉回到 +0.0026，但 OA 仍为 -0.0189、Weighted-F1 仍为 -0.0202，因此更准确的表述是 SupCon 缓解 Prototype 的 Macro-F1 负迁移，而不是全面超过 classical baseline。

## Evidence table
| dataset   |   shot | comparison                                                   | metric      |   baseline_mean |   qnn_mean |   mean_delta |   std_delta |   ci95_low |   ci95_high |   paired_t_pvalue |   wilcoxon_pvalue |    cohen_d | is_significant_p005   | interpretation    |   n_paired_seeds |
|:----------|-------:|:-------------------------------------------------------------|:------------|----------------:|-----------:|-------------:|------------:|-----------:|------------:|------------------:|------------------:|-----------:|:----------------------|:------------------|-----------------:|
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | OA          |        0.936018 |   0.907216 |  -0.0288019  |  0.0273252  | -0.0627306 |  0.00512686 |         0.0779256 |            0.125  | -1.05404   | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | AA          |        0.960622 |   0.95563  |  -0.00499127 |  0.0093773  | -0.0166347 |  0.00665218 |         0.299779  |            0.3125 | -0.532272  | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Kappa       |        0.928798 |   0.896468 |  -0.03233    |  0.030909   | -0.0707086 |  0.00604853 |         0.0794763 |            0.125  | -1.04598   | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Macro-F1    |        0.954418 |   0.949634 |  -0.00478366 |  0.00843171 | -0.015253  |  0.0056857  |         0.273382  |            0.3125 | -0.567341  | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Weighted-F1 |        0.936166 |   0.902897 |  -0.0332687  |  0.0344057  | -0.0759891 |  0.00945161 |         0.0966534 |            0.125  | -0.966954  | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | OA          |        0.936018 |   0.917137 |  -0.0188816  |  0.0256326  | -0.0507087 |  0.0129455  |         0.174874  |            0.1875 | -0.736624  | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | AA          |        0.960622 |   0.962063 |   0.00144147 |  0.0189299  | -0.0220632 |  0.0249461  |         0.873062  |            0.625  |  0.0761477 | False                 | marginal_positive |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | Kappa       |        0.928798 |   0.907686 |  -0.0211121  |  0.0289293  | -0.0570326 |  0.0148083  |         0.178048  |            0.1875 | -0.729783  | False                 | negative_transfer |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | Macro-F1    |        0.954418 |   0.957042 |   0.00262436 |  0.0171169  | -0.0186291 |  0.0238778  |         0.748983  |            0.625  |  0.153319  | False                 | marginal_positive |                5 |
| salinas   |     10 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | Weighted-F1 |        0.936166 |   0.916014 |  -0.0201521  |  0.0287852  | -0.0558937 |  0.0155895  |         0.192535  |            0.1875 | -0.700086  | False                 | negative_transfer |                5 |

## Prototype vs SupCon stability
| dataset   |   shot |   prototype_delta_macro_f1 |   supcon_delta_macro_f1 |   supcon_minus_prototype_macro_f1 |   prototype_std_delta_macro_f1 |   supcon_std_delta_macro_f1 |   prototype_positive_seed_count_macro_f1 |   supcon_positive_seed_count_macro_f1 |   prototype_negative_seed_count_macro_f1 |   supcon_negative_seed_count_macro_f1 |   prototype_delta_oa |   supcon_delta_oa |   prototype_std_delta_oa |   supcon_std_delta_oa | winner_by_macro_f1_mean   | stability_interpretation                     |
|:----------|-------:|---------------------------:|------------------------:|----------------------------------:|-------------------------------:|----------------------------:|-----------------------------------------:|--------------------------------------:|-----------------------------------------:|--------------------------------------:|---------------------:|------------------:|-------------------------:|----------------------:|:--------------------------|:---------------------------------------------|
| salinas   |     10 |                -0.00478366 |              0.00262436 |                        0.00740801 |                     0.00843171 |                   0.0171169 |                                        2 |                                     1 |                                        3 |                                     4 |           -0.0288019 |        -0.0188816 |                0.0273252 |             0.0256326 | SupCon                    | supcon_mitigates_prototype_negative_transfer |

## Per-class degradation: Prototype
| analysis_type      | comparison                                                   | loss_type   |   class_id | class_name                |     delta_f1 |   delta_recall | interpretation   |
|:-------------------|:-------------------------------------------------------------|:------------|-----------:|:--------------------------|-------------:|---------------:|:-----------------|
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |         14 | Vinyard_untrained         | -0.178994    |   -0.180657    | degraded         |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |          7 | Grapes_untrained          | -0.0729679   |   -0.0499867   | degraded         |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |         15 | Vinyard_vertical_trellis  | -0.0024303   |    0.00257415  | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |          2 | Fallow                    | -0.00168588  |    0.000102249 | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |         11 | Lettuce_romaine_5wk       | -0.000742198 |    0.00461458  | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |          0 | Brocoli_green_weeds_1     | -0.000553662 |   -0.000904977 | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |          1 | Brocoli_green_weeds_2     | -0.000378218 |   -0.000485699 | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |          9 | Corn_senesced_green_weeds |  0.000380699 |   -0.000736648 | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |          8 | Soil_vinyard_develop      |  0.000687353 |    0           | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |         10 | Lettuce_romaine_4wk       |  0.00284536  |   -0.00400763  | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |          6 | Celery                    |  0.0110142   |    0.0198932   | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |          5 | Stubble                   |  0.0137087   |    0.00639756  | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |         13 | Lettuce_romaine_7wk       |  0.0216824   |    0.012381    | improved         |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |         12 | Lettuce_romaine_6wk       |  0.0325764   |    0.0459821   | improved         |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |          4 | Fallow_smooth             |  0.0397148   |    0.0616253   | improved         |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |          3 | Fallow_rough_plow         |  0.0586032   |    0.00334789  | improved         |

## Per-class degradation: SupCon
| analysis_type      | comparison                                                | loss_type   |   class_id | class_name                |     delta_f1 |   delta_recall | interpretation   |
|:-------------------|:----------------------------------------------------------|:------------|-----------:|:--------------------------|-------------:|---------------:|:-----------------|
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |         14 | Vinyard_untrained         | -0.109528    |   -0.0961921   | degraded         |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |          7 | Grapes_untrained          | -0.0598376   |   -0.0615412   | degraded         |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |         15 | Vinyard_vertical_trellis  | -0.000846987 |   -0.00134303  | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |          1 | Brocoli_green_weeds_2     |  9.46349e-08 |    0.000215866 | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |          0 | Brocoli_green_weeds_1     |  4.97838e-05 |   -0.000301659 | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |          8 | Soil_vinyard_develop      |  0.000671084 |    0           | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |          2 | Fallow                    |  0.00090851  |    0.000102249 | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |          9 | Corn_senesced_green_weeds |  0.00233648  |    0.0033763   | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |         11 | Lettuce_romaine_5wk       |  0.00476382  |    0.00629261  | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |         10 | Lettuce_romaine_4wk       |  0.0120384   |   -0.000381679 | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |          5 | Stubble                   |  0.0131532   |    0.00644834  | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |          6 | Celery                    |  0.0131535   |    0.0245013   | near_neutral     |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |         13 | Lettuce_romaine_7wk       |  0.0246168   |    0.00533333  | improved         |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |          4 | Fallow_smooth             |  0.0395488   |    0.0635816   | improved         |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |         12 | Lettuce_romaine_6wk       |  0.0406052   |    0.0720982   | improved         |
| per_class_f1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss | SupCon      |          3 | Fallow_rough_plow         |  0.0603571   |    0.000873362 | improved         |

## Seed-wise degradation
| analysis_type              | comparison                                                   | loss_type   |   class_id | class_name   |     delta_f1 |   delta_recall | interpretation    |
|:---------------------------|:-------------------------------------------------------------|:------------|-----------:|:-------------|-------------:|---------------:|:------------------|
| seedwise_OA_delta          | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_0       | nan          |            nan | negative_transfer |
| seedwise_OA_delta          | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_1       | nan          |            nan | negative_transfer |
| seedwise_OA_delta          | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_2       | nan          |            nan | negative_transfer |
| seedwise_OA_delta          | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_3       | nan          |            nan | improved          |
| seedwise_OA_delta          | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_4       | nan          |            nan | negative_transfer |
| seedwise_Macro-F1_delta    | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_0       |  -0.0158316  |            nan | negative_transfer |
| seedwise_Macro-F1_delta    | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_1       |  -0.00823525 |            nan | negative_transfer |
| seedwise_Macro-F1_delta    | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_2       |  -0.00676825 |            nan | negative_transfer |
| seedwise_Macro-F1_delta    | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_3       |   0.00576105 |            nan | improved          |
| seedwise_Macro-F1_delta    | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_4       |   0.00115572 |            nan | improved          |
| seedwise_Weighted-F1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_0       | nan          |            nan | negative_transfer |
| seedwise_Weighted-F1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_1       | nan          |            nan | negative_transfer |
| seedwise_Weighted-F1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_2       | nan          |            nan | negative_transfer |
| seedwise_Weighted-F1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_3       | nan          |            nan | improved          |
| seedwise_Weighted-F1_delta | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Prototype   |        nan | seed_4       | nan          |            nan | negative_transfer |
| seedwise_OA_delta          | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_0       | nan          |            nan | negative_transfer |
| seedwise_OA_delta          | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_1       | nan          |            nan | negative_transfer |
| seedwise_OA_delta          | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_2       | nan          |            nan | negative_transfer |
| seedwise_OA_delta          | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_3       | nan          |            nan | negative_transfer |
| seedwise_OA_delta          | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_4       | nan          |            nan | improved          |
| seedwise_Macro-F1_delta    | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_0       |  -0.00700837 |            nan | negative_transfer |
| seedwise_Macro-F1_delta    | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_1       |  -0.00390373 |            nan | negative_transfer |
| seedwise_Macro-F1_delta    | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_2       |  -0.00156424 |            nan | negative_transfer |
| seedwise_Macro-F1_delta    | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_3       |  -0.00734987 |            nan | negative_transfer |
| seedwise_Macro-F1_delta    | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_4       |   0.032948   |            nan | improved          |
| seedwise_Weighted-F1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_0       | nan          |            nan | negative_transfer |
| seedwise_Weighted-F1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_1       | nan          |            nan | negative_transfer |
| seedwise_Weighted-F1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_2       | nan          |            nan | negative_transfer |
| seedwise_Weighted-F1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_3       | nan          |            nan | negative_transfer |
| seedwise_Weighted-F1_delta | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | SupCon      |        nan | seed_4       | nan          |            nan | improved          |

## Margin or gate evidence if available
| dataset   |   shot | model                    |   Macro-F1_mean |   separation_ratio_mean |   prototype_negative_margin_rate_mean |   mean_true_logit_margin_mean |   negative_logit_margin_rate_mean |
|:----------|-------:|:-------------------------|----------------:|------------------------:|--------------------------------------:|------------------------------:|----------------------------------:|
| salinas   |     10 | hybridsn_small           |        0.954418 |                 8.08233 |                             0.0571094 |                       2.3213  |                         0.0639819 |
| salinas   |     10 | spectral_qnn_gated_proto |        0.949634 |                 6.83662 |                             0.0718058 |                       2.23594 |                         0.0927837 |

已有 logit margin 分析只覆盖 Prototype 配置，因此支持“Prototype 负迁移伴随最终决策边界 margin 变差”。SupCon 的 logits/gate values 尚未做 evaluation-only 导出，因此目前不能直接证明 SupCon 缓解来自 gate 使用更合理或 margin 改善。

## Final interpretation

负迁移更可能来自 Salinas 10-shot classical baseline 已接近饱和，Prototype Loss 在 Vinyard_untrained 与 Grapes_untrained 等类别上放大了已有混淆。SupCon 对 Macro-F1 有缓解作用，说明对比式约束可能比单一 prototype 约束更稳健；但由于 OA/Weighted-F1 仍下降，这一结果应写成“部分缓解”而不是“解决负迁移”。后续需要保存 SupCon logits、gate values 和 branch contribution magnitude 来验证机制。
