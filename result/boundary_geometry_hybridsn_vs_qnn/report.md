# HybridSN-small vs Spectral QNN Boundary Geometry Analysis

## 1. 分析目的

本分析用于验证：QNN 主模型是否在部分 few-shot 设置中改善类别边界。比较对象仅为 HybridSN-small 与 Spectral QNN Gated Fusion + Prototype。

## 2. 方法

- 使用相同 split 的 support/train set 计算 class prototypes，test set 不参与 prototype 计算。
- 所有 feature 先做 L2 normalize；prototype 由 support feature 的 normalized mean 得到并再次 normalize。
- 使用 test query 到 prototype 的欧氏距离计算 intra-class distance、inter-class prototype distance、separation ratio 和 prototype margin。
- margin 小于 0 表示 query 更靠近错误 prototype；pred_label/correct 按最近 prototype 判定，不是分类头输出。

OA / Macro-F1 were read from available result metrics and merged into the geometry summary.

## 3. 主结果表

| dataset          |   shot | model                    |   runs |   OA_mean |   Macro-F1_mean |   mean_intra_distance_mean |   mean_intra_distance_std |   mean_inter_distance_mean |   mean_inter_distance_std |   separation_ratio_mean |   separation_ratio_std |   mean_margin_mean |   mean_margin_std |   negative_margin_rate_mean |   negative_margin_rate_std |   low_margin_rate_mean |   low_margin_rate_std |
|:-----------------|-------:|:-------------------------|-------:|----------:|----------------:|---------------------------:|--------------------------:|---------------------------:|--------------------------:|------------------------:|-----------------------:|-------------------:|------------------:|----------------------------:|---------------------------:|-----------------------:|----------------------:|
| indian_pines     |      5 | hybridsn_small           |      5 |  0.720192 |        0.638053 |                   0.436871 |                  0.011159 |                    1.13005 |                  0.016721 |                 2.58895 |               0.093084 |           0.249726 |          0.028588 |                    0.279589 |                   0.018239 |               0.019314 |              0.002445 |
| indian_pines     |      5 | spectral_qnn_gated_proto |      5 |  0.720511 |        0.641211 |                   0.490838 |                  0.01065  |                    1.1405  |                  0.017234 |                 2.32505 |               0.074202 |           0.23392  |          0.024516 |                    0.271828 |                   0.017037 |               0.023184 |              0.001929 |
| indian_pines     |     10 | hybridsn_small           |      5 |  0.801227 |        0.715332 |                   0.360537 |                  0.016224 |                    1.11725 |                  0.007437 |                 3.10475 |               0.132796 |           0.372272 |          0.030287 |                    0.196119 |                   0.022369 |               0.023686 |              0.00398  |
| indian_pines     |     10 | spectral_qnn_gated_proto |      5 |  0.808847 |        0.718224 |                   0.402161 |                  0.015055 |                    1.12232 |                  0.006442 |                 2.79452 |               0.102778 |           0.352425 |          0.031744 |                    0.18687  |                   0.028202 |               0.027586 |              0.005676 |
| pavia_university |      5 | hybridsn_small           |      5 |  0.758594 |        0.734896 |                   0.474207 |                  0.02916  |                    1.21865 |                  0.031315 |                 2.57936 |               0.170154 |           0.373426 |          0.075167 |                    0.240637 |                   0.038166 |               0.019409 |              0.006207 |
| pavia_university |      5 | spectral_qnn_gated_proto |      5 |  0.775277 |        0.771094 |                   0.517782 |                  0.037784 |                    1.23185 |                  0.018183 |                 2.39035 |               0.157689 |           0.37599  |          0.059573 |                    0.219788 |                   0.029325 |               0.020379 |              0.003402 |
| pavia_university |     10 | hybridsn_small           |      5 |  0.822617 |        0.791976 |                   0.382605 |                  0.032737 |                    1.1892  |                  0.026784 |                 3.13551 |               0.322818 |           0.495604 |          0.068069 |                    0.151263 |                   0.02647  |               0.015856 |              0.006985 |
| pavia_university |     10 | spectral_qnn_gated_proto |      5 |  0.863151 |        0.861272 |                   0.426241 |                  0.038795 |                    1.20339 |                  0.030681 |                 2.84556 |               0.259219 |           0.489892 |          0.057488 |                    0.136839 |                   0.025105 |               0.012381 |              0.002475 |
| salinas          |      5 | hybridsn_small           |      5 |  0.869328 |        0.896679 |                   0.185715 |                  0.018047 |                    1.21279 |                  0.02275  |                 6.59936 |               0.711108 |           0.491673 |          0.069567 |                    0.119742 |                   0.021038 |               0.046781 |              0.022883 |
| salinas          |      5 | spectral_qnn_gated_proto |      5 |  0.881649 |        0.93356  |                   0.219601 |                  0.019485 |                    1.21199 |                  0.025238 |                 5.56046 |               0.483748 |           0.481442 |          0.049824 |                    0.127974 |                   0.02281  |               0.04261  |              0.021746 |
| salinas          |     10 | hybridsn_small           |      5 |  0.936018 |        0.954418 |                   0.153249 |                  0.009643 |                    1.2339  |                  0.006003 |                 8.08233 |               0.488947 |           0.567371 |          0.044186 |                    0.057109 |                   0.008459 |               0.085082 |              0.108312 |
| salinas          |     10 | spectral_qnn_gated_proto |      5 |  0.907216 |        0.949634 |                   0.180745 |                  0.010107 |                    1.23226 |                  0.009595 |                 6.83662 |               0.339251 |           0.556997 |          0.033288 |                    0.071806 |                   0.016662 |               0.078771 |              0.088273 |

## 4. QNN vs HybridSN 差值

| dataset          |   shot | metric               |   hybridsn_mean |   qnn_mean |     delta | better_model             |
|:-----------------|-------:|:---------------------|----------------:|-----------:|----------:|:-------------------------|
| indian_pines     |      5 | mean_inter_distance  |        1.13005  |   1.1405   |  0.010451 | spectral_qnn_gated_proto |
| indian_pines     |      5 | separation_ratio     |        2.58895  |   2.32505  | -0.263891 | hybridsn_small           |
| indian_pines     |      5 | mean_margin          |        0.249726 |   0.23392  | -0.015806 | hybridsn_small           |
| indian_pines     |      5 | median_margin        |        0.390162 |   0.351324 | -0.038838 | hybridsn_small           |
| indian_pines     |      5 | safe_margin_rate     |        0.701097 |   0.704988 |  0.003891 | spectral_qnn_gated_proto |
| indian_pines     |      5 | mean_intra_distance  |        0.436871 |   0.490838 |  0.053967 | hybridsn_small           |
| indian_pines     |      5 | negative_margin_rate |        0.279589 |   0.271828 | -0.007761 | spectral_qnn_gated_proto |
| indian_pines     |      5 | low_margin_rate      |        0.019314 |   0.023184 |  0.003871 | hybridsn_small           |
| indian_pines     |     10 | mean_inter_distance  |        1.11725  |   1.12232  |  0.005068 | spectral_qnn_gated_proto |
| indian_pines     |     10 | separation_ratio     |        3.10475  |   2.79452  | -0.310225 | hybridsn_small           |
| indian_pines     |     10 | mean_margin          |        0.372272 |   0.352425 | -0.019847 | hybridsn_small           |
| indian_pines     |     10 | median_margin        |        0.471224 |   0.428456 | -0.042769 | hybridsn_small           |
| indian_pines     |     10 | safe_margin_rate     |        0.780195 |   0.785543 |  0.005348 | spectral_qnn_gated_proto |
| indian_pines     |     10 | mean_intra_distance  |        0.360537 |   0.402161 |  0.041624 | hybridsn_small           |
| indian_pines     |     10 | negative_margin_rate |        0.196119 |   0.18687  | -0.009249 | spectral_qnn_gated_proto |
| indian_pines     |     10 | low_margin_rate      |        0.023686 |   0.027586 |  0.003901 | hybridsn_small           |
| pavia_university |      5 | mean_inter_distance  |        1.21865  |   1.23185  |  0.013195 | spectral_qnn_gated_proto |
| pavia_university |      5 | separation_ratio     |        2.57936  |   2.39035  | -0.189011 | hybridsn_small           |
| pavia_university |      5 | mean_margin          |        0.373426 |   0.37599  |  0.002564 | spectral_qnn_gated_proto |
| pavia_university |      5 | median_margin        |        0.503245 |   0.474839 | -0.028406 | hybridsn_small           |
| pavia_university |      5 | safe_margin_rate     |        0.739955 |   0.759832 |  0.019878 | spectral_qnn_gated_proto |
| pavia_university |      5 | mean_intra_distance  |        0.474207 |   0.517782 |  0.043575 | hybridsn_small           |
| pavia_university |      5 | negative_margin_rate |        0.240637 |   0.219788 | -0.020848 | spectral_qnn_gated_proto |
| pavia_university |      5 | low_margin_rate      |        0.019409 |   0.020379 |  0.000971 | hybridsn_small           |
| pavia_university |     10 | mean_inter_distance  |        1.1892   |   1.20339  |  0.014187 | spectral_qnn_gated_proto |
| pavia_university |     10 | separation_ratio     |        3.13551  |   2.84556  | -0.289955 | hybridsn_small           |
| pavia_university |     10 | mean_margin          |        0.495604 |   0.489892 | -0.005712 | hybridsn_small           |
| pavia_university |     10 | median_margin        |        0.640851 |   0.611973 | -0.028878 | hybridsn_small           |
| pavia_university |     10 | safe_margin_rate     |        0.832881 |   0.850779 |  0.017898 | spectral_qnn_gated_proto |
| pavia_university |     10 | mean_intra_distance  |        0.382605 |   0.426241 |  0.043636 | hybridsn_small           |
| pavia_university |     10 | negative_margin_rate |        0.151263 |   0.136839 | -0.014424 | spectral_qnn_gated_proto |
| pavia_university |     10 | low_margin_rate      |        0.015856 |   0.012381 | -0.003475 | spectral_qnn_gated_proto |
| salinas          |      5 | mean_inter_distance  |        1.21279  |   1.21199  | -0.000799 | hybridsn_small           |
| salinas          |      5 | separation_ratio     |        6.59936  |   5.56046  | -1.0389   | hybridsn_small           |
| salinas          |      5 | mean_margin          |        0.491673 |   0.481442 | -0.010231 | hybridsn_small           |
| salinas          |      5 | median_margin        |        0.529758 |   0.541055 |  0.011297 | spectral_qnn_gated_proto |
| salinas          |      5 | safe_margin_rate     |        0.833476 |   0.829416 | -0.00406  | hybridsn_small           |
| salinas          |      5 | mean_intra_distance  |        0.185715 |   0.219601 |  0.033887 | hybridsn_small           |
| salinas          |      5 | negative_margin_rate |        0.119742 |   0.127974 |  0.008232 | hybridsn_small           |
| salinas          |      5 | low_margin_rate      |        0.046781 |   0.04261  | -0.004172 | spectral_qnn_gated_proto |
| salinas          |     10 | mean_inter_distance  |        1.2339   |   1.23226  | -0.001636 | hybridsn_small           |
| salinas          |     10 | separation_ratio     |        8.08233  |   6.83662  | -1.24571  | hybridsn_small           |
| salinas          |     10 | mean_margin          |        0.567371 |   0.556997 | -0.010374 | hybridsn_small           |
| salinas          |     10 | median_margin        |        0.67693  |   0.66048  | -0.016449 | hybridsn_small           |
| salinas          |     10 | safe_margin_rate     |        0.857808 |   0.849423 | -0.008385 | hybridsn_small           |
| salinas          |     10 | mean_intra_distance  |        0.153249 |   0.180745 |  0.027495 | hybridsn_small           |
| salinas          |     10 | negative_margin_rate |        0.057109 |   0.071806 |  0.014696 | hybridsn_small           |
| salinas          |     10 | low_margin_rate      |        0.085082 |   0.078771 | -0.006311 | spectral_qnn_gated_proto |

## 5. Paired seed 分析

- indian_pines 5-shot: mean_margin: QNN better in 0/5 seeds; negative_margin_rate: QNN better in 4/5 seeds; separation_ratio: QNN better in 0/5 seeds
- indian_pines 10-shot: mean_margin: QNN better in 0/5 seeds; negative_margin_rate: QNN better in 5/5 seeds; separation_ratio: QNN better in 0/5 seeds
- pavia_university 5-shot: mean_margin: QNN better in 3/5 seeds; negative_margin_rate: QNN better in 5/5 seeds; separation_ratio: QNN better in 0/5 seeds
- pavia_university 10-shot: mean_margin: QNN better in 2/5 seeds; negative_margin_rate: QNN better in 5/5 seeds; separation_ratio: QNN better in 0/5 seeds
- salinas 5-shot: mean_margin: QNN better in 1/5 seeds; negative_margin_rate: QNN better in 2/5 seeds; separation_ratio: QNN better in 0/5 seeds
- salinas 10-shot: mean_margin: QNN better in 1/5 seeds; negative_margin_rate: QNN better in 0/5 seeds; separation_ratio: QNN better in 0/5 seeds

## 6. Margin distribution 分析

- indian_pines 5-shot: QNN margin 均值未右移，需结合 paired seed 和分类指标判断。
- indian_pines 10-shot: QNN margin 均值未右移，需结合 paired seed 和分类指标判断。
- pavia_university 5-shot: QNN margin 均值右移且 negative margin rate 降低。
- pavia_university 10-shot: QNN margin 均值未右移，需结合 paired seed 和分类指标判断。
- salinas 5-shot: QNN margin 均值未右移，需结合 paired seed 和分类指标判断。
- salinas 10-shot: QNN margin 均值未右移，需结合 paired seed 和分类指标判断。

margin distribution 图保存在 `plots/margin_distribution/`，逐个 dataset / shot 比较两模型的 prototype margin。

## 7. 初步结论

- indian_pines 5-shot: geometry 指标未同时满足 separation、margin 和 negative margin rate 改善，性能变化不应直接归因于更清晰的 prototype geometry。
- indian_pines 10-shot: geometry 指标未同时满足 separation、margin 和 negative margin rate 改善，性能变化不应直接归因于更清晰的 prototype geometry。
- pavia_university 5-shot: geometry 指标未同时满足 separation、margin 和 negative margin rate 改善，性能变化不应直接归因于更清晰的 prototype geometry。
- pavia_university 10-shot: geometry 指标未同时满足 separation、margin 和 negative margin rate 改善，性能变化不应直接归因于更清晰的 prototype geometry。
- salinas 5-shot: geometry 指标未同时满足 separation、margin 和 negative margin rate 改善，性能变化不应直接归因于更清晰的 prototype geometry。
- salinas 10-shot: geometry 指标未同时满足 separation、margin 和 negative margin rate 改善，性能变化不应直接归因于更清晰的 prototype geometry。 该设置也符合 classical baseline 接近饱和时可能出现负迁移的谨慎解释。

本报告不把 geometry 指标解释为普遍证明。实验结果支持时，只能说明在相应 dataset / shot 设置中，Spectral QNN Gated Fusion + Prototype 相比 HybridSN-small 表现出更好的 prototype margin 和 separation ratio，因此可以认为它改善了特征空间中的类别边界。
