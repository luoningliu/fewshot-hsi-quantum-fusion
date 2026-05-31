# 类别级遥感解释实验汇总

## 实验范围

- 主诊断：ConfidenceGuard-C + SupCon vs HybridSN-small，覆盖 Salinas 10-shot 与 Pavia University 10-shot。
- 补充诊断：Spectral QNN + SupCon vs HybridSN-small，覆盖 Indian Pines 5-shot 与 10-shot。
- 每个设置使用 seeds 0--4，按 class 对齐 precision、recall、F1、accuracy，并比较 confusion pair 的错误计数变化。

## 类别级结论

### Indian Pines 5-shot

- mean per-class delta recall: -0.0011
- mean per-class delta F1: +0.0023
- support-weighted mean delta F1: -0.0036

**F1 下降最大的类别**

| class_name | support_mean | f1_baseline_mean | f1_qnn_mean | delta_f1_mean | delta_recall_mean | negative_f1_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Soybean-clean | 578.0000 | 0.5297 | 0.5000 | -0.0297 | -0.0467 | 2 |
| Stone-Steel-Towers | 78.0000 | 0.5220 | 0.4956 | -0.0264 | 0.0103 | 3 |
| Corn-notill | 1413.0000 | 0.6418 | 0.6236 | -0.0183 | -0.0277 | 3 |
| Woods | 1250.0000 | 0.8785 | 0.8627 | -0.0158 | -0.0056 | 5 |
| Hay-windrowed | 463.0000 | 0.9381 | 0.9251 | -0.0130 | 0.0022 | 3 |

**F1 提升最大的类别**

| class_name | support_mean | f1_baseline_mean | f1_qnn_mean | delta_f1_mean | delta_recall_mean | positive_f1_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Wheat | 190.0000 | 0.5415 | 0.6164 | 0.0749 | 0.0000 | 4 |
| Alfalfa | 33.0000 | 0.3828 | 0.4218 | 0.0391 | 0.0061 | 4 |
| Grass-pasture-mowed | 19.0000 | 0.4271 | 0.4496 | 0.0225 | 0.0000 | 4 |
| Grass-pasture | 468.0000 | 0.7278 | 0.7451 | 0.0173 | 0.0077 | 4 |
| Corn-mintill | 815.0000 | 0.4903 | 0.4978 | 0.0075 | 0.0054 | 2 |

**新增最多的混淆对（QNN 错误数 - HybridSN-small 错误数）**

| true_class_name | pred_class_name | baseline_count_mean | qnn_count_mean | delta_count_mean | increased_seeds |
| --- | --- | --- | --- | --- | --- |
| Corn-mintill | Soybean-notill | 21.3333 | 67.0000 | 45.6667 | 3 |
| Corn-mintill | Soybean-clean | 62.4000 | 100.4000 | 38.0000 | 4 |
| Soybean-mintill | Woods | 75.6667 | 108.6667 | 33.0000 | 3 |
| Corn-notill | Hay-windrowed | 1.0000 | 30.0000 | 29.0000 | 1 |
| Grass-pasture | Woods | 18.0000 | 45.0000 | 27.0000 | 1 |
| Corn-notill | Soybean-notill | 114.4000 | 139.6000 | 25.2000 | 5 |
| Grass-pasture | Corn-notill | 34.0000 | 53.0000 | 19.0000 | 1 |
| Woods | Buildings-Grass-Trees-Drives | 65.6000 | 83.4000 | 17.8000 | 4 |

**减少最多的混淆对**

| true_class_name | pred_class_name | baseline_count_mean | qnn_count_mean | delta_count_mean | decreased_seeds |
| --- | --- | --- | --- | --- | --- |
| Corn-mintill | Wheat | 179.0000 | 135.6000 | -43.4000 | 4 |
| Soybean-notill | Wheat | 84.5000 | 49.5000 | -35.0000 | 2 |
| Soybean-mintill | Alfalfa | 50.0000 | 18.0000 | -32.0000 | 1 |
| Woods | Corn-mintill | 77.0000 | 45.5000 | -31.5000 | 2 |
| Corn-mintill | Buildings-Grass-Trees-Drives | 42.5000 | 12.7500 | -29.7500 | 2 |
| Soybean-mintill | Corn-notill | 115.4000 | 93.6000 | -21.8000 | 3 |
| Soybean-mintill | Wheat | 82.4000 | 63.0000 | -19.4000 | 4 |
| Grass-pasture | Wheat | 44.0000 | 26.2000 | -17.8000 | 5 |

### Indian Pines 10-shot

- mean per-class delta recall: +0.0041
- mean per-class delta F1: +0.0073
- support-weighted mean delta F1: +0.0092

**F1 下降最大的类别**

| class_name | support_mean | f1_baseline_mean | f1_qnn_mean | delta_f1_mean | delta_recall_mean | negative_f1_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Grass-pasture-mowed | 15.0000 | 0.4512 | 0.4249 | -0.0262 | 0.0000 | 3 |
| Buildings-Grass-Trees-Drives | 366.0000 | 0.7797 | 0.7563 | -0.0234 | -0.0005 | 4 |
| Corn | 217.0000 | 0.8680 | 0.8475 | -0.0205 | 0.0009 | 3 |
| Soybean-clean | 573.0000 | 0.6912 | 0.6814 | -0.0098 | 0.0192 | 3 |
| Hay-windrowed | 458.0000 | 0.9686 | 0.9611 | -0.0075 | 0.0000 | 2 |

**F1 提升最大的类别**

| class_name | support_mean | f1_baseline_mean | f1_qnn_mean | delta_f1_mean | delta_recall_mean | positive_f1_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Stone-Steel-Towers | 73.0000 | 0.5617 | 0.6237 | 0.0620 | 0.0000 | 3 |
| Wheat | 185.0000 | 0.7546 | 0.8016 | 0.0470 | 0.0097 | 2 |
| Corn-mintill | 810.0000 | 0.7328 | 0.7774 | 0.0446 | 0.0143 | 4 |
| Corn-notill | 1408.0000 | 0.7395 | 0.7572 | 0.0178 | -0.0058 | 3 |
| Soybean-mintill | 2435.0000 | 0.8133 | 0.8263 | 0.0130 | 0.0193 | 4 |

**新增最多的混淆对（QNN 错误数 - HybridSN-small 错误数）**

| true_class_name | pred_class_name | baseline_count_mean | qnn_count_mean | delta_count_mean | increased_seeds |
| --- | --- | --- | --- | --- | --- |
| Soybean-mintill | Soybean-clean | 65.0000 | 118.5000 | 53.5000 | 2 |
| Woods | Buildings-Grass-Trees-Drives | 34.2500 | 54.2500 | 20.0000 | 3 |
| Soybean-mintill | Woods | 31.7500 | 49.5000 | 17.7500 | 3 |
| Soybean-mintill | Soybean-notill | 63.2000 | 79.2000 | 16.0000 | 3 |
| Soybean-clean | Corn-mintill | 30.6000 | 42.2000 | 11.6000 | 4 |
| Corn-notill | Soybean-clean | 30.2000 | 40.4000 | 10.2000 | 2 |
| Soybean-mintill | Hay-windrowed | 21.4000 | 31.6000 | 10.2000 | 3 |
| Soybean-notill | Corn | 1.0000 | 9.0000 | 8.0000 | 1 |

**减少最多的混淆对**

| true_class_name | pred_class_name | baseline_count_mean | qnn_count_mean | delta_count_mean | decreased_seeds |
| --- | --- | --- | --- | --- | --- |
| Woods | Corn-notill | 109.0000 | 3.0000 | -106.0000 | 1 |
| Soybean-mintill | Corn-mintill | 96.2000 | 39.8000 | -56.4000 | 4 |
| Soybean-mintill | Corn-notill | 220.6000 | 171.0000 | -49.6000 | 4 |
| Woods | Corn-mintill | 33.0000 | 5.0000 | -28.0000 | 2 |
| Soybean-clean | Stone-Steel-Towers | 93.8000 | 75.2000 | -18.6000 | 4 |
| Soybean-mintill | Wheat | 60.2500 | 44.2500 | -16.0000 | 3 |
| Corn-mintill | Wheat | 61.4000 | 47.4000 | -14.0000 | 2 |
| Soybean-clean | Soybean-notill | 17.5000 | 4.0000 | -13.5000 | 2 |

### Pavia University 10-shot

- mean per-class delta recall: +0.0517
- mean per-class delta F1: +0.0673
- support-weighted mean delta F1: +0.0369

**F1 下降最大的类别**

| class_name | support_mean | f1_baseline_mean | f1_qnn_mean | delta_f1_mean | delta_recall_mean | negative_f1_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Bare Soil | 5009.0000 | 0.9254 | 0.9035 | -0.0219 | -0.0033 | 3 |
| Meadows | 18629.0000 | 0.9071 | 0.8974 | -0.0097 | -0.0159 | 3 |
| Bitumen | 1310.0000 | 0.8637 | 0.9104 | 0.0466 | -0.0014 | 1 |
| Gravel | 2079.0000 | 0.7896 | 0.8424 | 0.0528 | -0.0297 | 1 |
| Painted metal sheets | 1325.0000 | 0.9130 | 0.9721 | 0.0591 | 0.0195 | 0 |

**F1 提升最大的类别**

| class_name | support_mean | f1_baseline_mean | f1_qnn_mean | delta_f1_mean | delta_recall_mean | positive_f1_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Self-Blocking Bricks | 3662.0000 | 0.6869 | 0.8418 | 0.1549 | 0.1881 | 5 |
| Shadows | 927.0000 | 0.7287 | 0.8758 | 0.1471 | 0.1392 | 4 |
| Asphalt | 6611.0000 | 0.7194 | 0.8228 | 0.1034 | 0.1433 | 5 |
| Trees | 3044.0000 | 0.5940 | 0.6677 | 0.0737 | 0.0254 | 5 |
| Painted metal sheets | 1325.0000 | 0.9130 | 0.9721 | 0.0591 | 0.0195 | 5 |

**新增最多的混淆对（QNN 错误数 - HybridSN-small 错误数）**

| true_class_name | pred_class_name | baseline_count_mean | qnn_count_mean | delta_count_mean | increased_seeds |
| --- | --- | --- | --- | --- | --- |
| Meadows | Bare Soil | 567.0000 | 789.4000 | 222.4000 | 2 |
| Asphalt | Self-Blocking Bricks | 259.2500 | 454.7500 | 195.5000 | 3 |
| Meadows | Self-Blocking Bricks | 345.0000 | 485.6667 | 140.6667 | 1 |
| Gravel | Self-Blocking Bricks | 132.2000 | 256.2000 | 124.0000 | 3 |
| Meadows | Asphalt | 367.2500 | 483.7500 | 116.5000 | 3 |
| Meadows | Shadows | 57.0000 | 132.6667 | 75.6667 | 1 |
| Meadows | Trees | 1409.2000 | 1481.0000 | 71.8000 | 2 |
| Meadows | Bitumen | 188.0000 | 243.0000 | 55.0000 | 1 |

**减少最多的混淆对**

| true_class_name | pred_class_name | baseline_count_mean | qnn_count_mean | delta_count_mean | decreased_seeds |
| --- | --- | --- | --- | --- | --- |
| Asphalt | Trees | 1512.6000 | 707.2000 | -805.4000 | 4 |
| Self-Blocking Bricks | Gravel | 590.2000 | 56.6000 | -533.6000 | 4 |
| Meadows | Gravel | 687.0000 | 291.5000 | -395.5000 | 1 |
| Meadows | Painted metal sheets | 372.0000 | 0.0000 | -372.0000 | 1 |
| Asphalt | Bitumen | 563.3333 | 328.0000 | -235.3333 | 2 |
| Asphalt | Gravel | 494.6667 | 304.6667 | -190.0000 | 2 |
| Self-Blocking Bricks | Trees | 200.4000 | 83.6000 | -116.8000 | 2 |
| Shadows | Painted metal sheets | 145.2000 | 72.6000 | -72.6000 | 4 |

### Salinas 10-shot

- mean per-class delta recall: +0.0052
- mean per-class delta F1: +0.0068
- support-weighted mean delta F1: -0.0098

**F1 下降最大的类别**

| class_name | support_mean | f1_baseline_mean | f1_qnn_mean | delta_f1_mean | delta_recall_mean | negative_f1_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Vinyard_untrained | 7248.0000 | 0.8168 | 0.7617 | -0.0551 | -0.0248 | 3 |
| Grapes_untrained | 11251.0000 | 0.8796 | 0.8339 | -0.0458 | -0.0651 | 3 |
| Vinyard_vertical_trellis | 1787.0000 | 0.9974 | 0.9967 | -0.0007 | -0.0006 | 2 |
| Brocoli_green_weeds_2 | 3706.0000 | 0.9999 | 0.9999 | 0.0000 | 0.0002 | 0 |
| Brocoli_green_weeds_1 | 1989.0000 | 0.9996 | 0.9997 | 0.0000 | -0.0002 | 1 |

**F1 提升最大的类别**

| class_name | support_mean | f1_baseline_mean | f1_qnn_mean | delta_f1_mean | delta_recall_mean | positive_f1_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Fallow_rough_plow | 1374.0000 | 0.8759 | 0.9431 | 0.0672 | 0.0012 | 4 |
| Fallow_smooth | 2658.0000 | 0.9352 | 0.9772 | 0.0420 | 0.0679 | 4 |
| Lettuce_romaine_6wk | 896.0000 | 0.9332 | 0.9705 | 0.0373 | 0.0685 | 3 |
| Lettuce_romaine_7wk | 1050.0000 | 0.9399 | 0.9648 | 0.0248 | 0.0008 | 3 |
| Stubble | 3939.0000 | 0.9855 | 0.9990 | 0.0134 | 0.0062 | 4 |

**新增最多的混淆对（QNN 错误数 - HybridSN-small 错误数）**

| true_class_name | pred_class_name | baseline_count_mean | qnn_count_mean | delta_count_mean | increased_seeds |
| --- | --- | --- | --- | --- | --- |
| Grapes_untrained | Vinyard_untrained | 1461.2000 | 2214.8000 | 753.6000 | 5 |
| Vinyard_untrained | Grapes_untrained | 1152.6000 | 1368.6000 | 216.0000 | 2 |
| Vinyard_untrained | Fallow_smooth | 0.0000 | 19.0000 | 19.0000 | 1 |
| Celery | Vinyard_vertical_trellis | 4.0000 | 12.0000 | 8.0000 | 1 |
| Lettuce_romaine_5wk | Lettuce_romaine_6wk | 11.7500 | 19.5000 | 7.7500 | 2 |
| Vinyard_vertical_trellis | Fallow_rough_plow | 12.3333 | 15.6667 | 3.3333 | 2 |
| Brocoli_green_weeds_1 | Brocoli_green_weeds_2 | 0.0000 | 3.0000 | 3.0000 | 1 |
| Corn_senesced_green_weeds | Fallow | 21.2000 | 23.8000 | 2.6000 | 3 |

**减少最多的混淆对**

| true_class_name | pred_class_name | baseline_count_mean | qnn_count_mean | delta_count_mean | decreased_seeds |
| --- | --- | --- | --- | --- | --- |
| Fallow_smooth | Fallow_rough_plow | 285.4000 | 108.2000 | -177.2000 | 4 |
| Celery | Stubble | 105.0000 | 4.7500 | -100.2500 | 3 |
| Stubble | Fallow_smooth | 60.5000 | 3.0000 | -57.5000 | 1 |
| Lettuce_romaine_6wk | Lettuce_romaine_7wk | 79.6000 | 23.4000 | -56.2000 | 2 |
| Vinyard_untrained | Fallow_rough_plow | 82.2000 | 41.8000 | -40.4000 | 4 |
| Fallow_smooth | Stubble | 17.0000 | 0.0000 | -17.0000 | 1 |
| Celery | Fallow_smooth | 15.0000 | 0.0000 | -15.0000 | 1 |
| Grapes_untrained | Fallow_rough_plow | 17.5000 | 4.0000 | -13.5000 | 4 |

## 论文表述建议

1. Salinas 10-shot 的负迁移主要不是所有类别一起下降，而是由少数高支持度地物类别的 F1/recall 损失放大为 OA 与 Weighted-F1 下降。
2. QNN 分支在部分光谱相近或 baseline 不稳定的类别上仍能改善 Macro-F1，因此不能简单写成 QNN 无效。
3. 更准确的表述是：QNN spectral residual 的收益具有类别条件性；当强 baseline 已经稳定识别大类时，未校准 residual 会对这些类别造成扰动。
4. 后续应把 validation-calibrated class mask 或 confusion-pair mask 作为主要改进方向。
