# HybridSN-small vs Spectral QNN Logit Margin Analysis

## 1. 分析目的

Prototype geometry 不能完全代表最终分类边界，因此本实验直接分析 classifier logits，判断 QNN 融合是否改善真正用于预测的 decision boundary。

## 2. 方法

- true-class logit margin = 真实类别 logit 与最高错误类别 logit 的差。
- top1-top2 margin = 预测第一候选与第二候选 logit 的差。
- 同时计算 softmax probability margin，缓解不同模型 logits scale 不一致的问题。
- negative / low / safe margin rate 分别描述错误侧、靠近边界和远离阈值边界的样本比例。
- 所有 margin 只在已有 split 的 test set 上计算，并做 paired seed 与 per-class 分析。

## 3. 主结果表

| dataset          |   shot | model                    |   runs |   OA_mean |   OA_std |   Macro-F1_mean |   Macro-F1_std |   mean_true_logit_margin_mean |   mean_true_logit_margin_std |   negative_logit_margin_rate_mean |   negative_logit_margin_rate_std |   low_logit_margin_rate_mean |   low_logit_margin_rate_std |   safe_logit_margin_rate_mean |   safe_logit_margin_rate_std |   mean_true_prob_margin_mean |   mean_true_prob_margin_std |   negative_prob_margin_rate_mean |   negative_prob_margin_rate_std |   low_prob_margin_rate_mean |   low_prob_margin_rate_std |   safe_prob_margin_rate_mean |   safe_prob_margin_rate_std |   mean_top1_top2_margin_mean |   mean_top1_top2_prob_margin_mean |
|:-----------------|-------:|:-------------------------|-------:|----------:|---------:|----------------:|---------------:|------------------------------:|-----------------------------:|----------------------------------:|---------------------------------:|-----------------------------:|----------------------------:|------------------------------:|-----------------------------:|-----------------------------:|----------------------------:|---------------------------------:|--------------------------------:|----------------------------:|---------------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------------------------:|
| indian_pines     |      5 | hybridsn_small           |      5 |  0.720192 | 0.016713 |        0.638053 |       0.021696 |                      0.864631 |                     0.137648 |                          0.279808 |                         0.016713 |                     0.012151 |                    0.001692 |                      0.708041 |                     0.01596  |                     0.313186 |                    0.031605 |                         0.279808 |                        0.016713 |                    0.025599 |                   0.003577 |                     0.694593 |                    0.016604 |                      1.66271 |                          0.475668 |
| indian_pines     |      5 | spectral_qnn_gated_proto |      5 |  0.720511 | 0.014953 |        0.641211 |       0.017264 |                      0.957528 |                     0.200356 |                          0.279489 |                         0.014953 |                     0.009218 |                    0.003819 |                      0.711293 |                     0.016276 |                     0.327064 |                    0.068973 |                         0.279489 |                        0.014953 |                    0.01672  |                   0.011513 |                     0.703791 |                    0.020558 |                      1.99542 |                          0.543319 |
| indian_pines     |     10 | hybridsn_small           |      5 |  0.801227 | 0.025553 |        0.715332 |       0.020213 |                      1.88013  |                     0.334558 |                          0.198773 |                         0.025553 |                     0.010938 |                    0.002566 |                      0.790289 |                     0.025828 |                     0.476895 |                    0.057521 |                         0.198773 |                        0.025553 |                    0.017814 |                   0.005649 |                     0.783412 |                    0.026529 |                      2.41666 |                          0.605293 |
| indian_pines     |     10 | spectral_qnn_gated_proto |      5 |  0.808847 | 0.023972 |        0.718224 |       0.0192   |                      1.80259  |                     0.277124 |                          0.191153 |                         0.023972 |                     0.007801 |                    0.001465 |                      0.801046 |                     0.025318 |                     0.489033 |                    0.060384 |                         0.191153 |                        0.023972 |                    0.010898 |                   0.002189 |                     0.797949 |                    0.025444 |                      2.4217  |                          0.640438 |
| pavia_university |      5 | hybridsn_small           |      5 |  0.76577  | 0.043278 |        0.761965 |       0.018694 |                      1.41716  |                     0.500916 |                          0.23423  |                         0.043278 |                     0.012809 |                    0.004324 |                      0.752961 |                     0.045693 |                     0.410117 |                    0.096957 |                         0.23423  |                        0.043278 |                    0.02249  |                   0.009618 |                     0.74328  |                    0.048907 |                      2.0116  |                          0.549902 |
| pavia_university |      5 | spectral_qnn_gated_proto |      5 |  0.775277 | 0.034776 |        0.771094 |       0.021743 |                      1.12807  |                     0.461368 |                          0.224723 |                         0.034776 |                     0.011773 |                    0.004499 |                      0.763505 |                     0.035212 |                     0.345616 |                    0.106037 |                         0.224723 |                        0.034776 |                    0.017753 |                   0.007475 |                     0.757524 |                    0.034885 |                      1.68605 |                          0.490219 |
| pavia_university |     10 | hybridsn_small           |      5 |  0.822617 | 0.049248 |        0.791976 |       0.08137  |                      1.51262  |                     0.622055 |                          0.177383 |                         0.049248 |                     0.017462 |                    0.011548 |                      0.805155 |                     0.05962  |                     0.426845 |                    0.161028 |                         0.177383 |                        0.049248 |                    0.044558 |                   0.049089 |                     0.778059 |                    0.094827 |                      1.83726 |                          0.500917 |
| pavia_university |     10 | spectral_qnn_gated_proto |      5 |  0.863151 | 0.025889 |        0.861272 |       0.022158 |                      1.60143  |                     0.314788 |                          0.136849 |                         0.025889 |                     0.00709  |                    0.001525 |                      0.856062 |                     0.026158 |                     0.480297 |                    0.091458 |                         0.136849 |                        0.025889 |                    0.009963 |                   0.003369 |                     0.853188 |                    0.026233 |                      1.96944 |                          0.574832 |
| salinas          |      5 | hybridsn_small           |      5 |  0.869328 | 0.071067 |        0.896679 |       0.092079 |                      1.64498  |                     0.699677 |                          0.130672 |                         0.071067 |                     0.104708 |                    0.065075 |                      0.76462  |                     0.095208 |                     0.402037 |                    0.178419 |                         0.130672 |                        0.071067 |                    0.174061 |                   0.135567 |                     0.695266 |                    0.198882 |                      1.69821 |                          0.416985 |
| salinas          |      5 | spectral_qnn_gated_proto |      5 |  0.881649 | 0.020916 |        0.93356  |       0.007776 |                      1.88427  |                     0.491856 |                          0.118351 |                         0.020916 |                     0.049153 |                    0.033635 |                      0.832496 |                     0.030684 |                     0.484898 |                    0.089829 |                         0.118351 |                        0.020916 |                    0.056119 |                   0.039074 |                     0.82553  |                    0.035267 |                      1.96223 |                          0.515453 |
| salinas          |     10 | hybridsn_small           |      5 |  0.936018 | 0.01589  |        0.954418 |       0.015492 |                      2.3213   |                     0.790588 |                          0.063982 |                         0.01589  |                     0.100069 |                    0.095767 |                      0.835949 |                     0.10864  |                     0.509174 |                    0.120778 |                         0.063982 |                        0.01589  |                    0.107748 |                   0.099493 |                     0.82827  |                    0.112368 |                      2.34644 |                          0.518478 |
| salinas          |     10 | spectral_qnn_gated_proto |      5 |  0.907216 | 0.037721 |        0.949634 |       0.017938 |                      2.23594  |                     0.757934 |                          0.092784 |                         0.037721 |                     0.037455 |                    0.030841 |                      0.869762 |                     0.066903 |                     0.537574 |                    0.156633 |                         0.092784 |                        0.037721 |                    0.044167 |                   0.035456 |                     0.863049 |                    0.07218  |                      2.29256 |                          0.560095 |

## 4. QNN vs HybridSN 差值

| dataset          |   shot | metric                     |   hybridsn_mean |   qnn_mean |     delta | better_model             |
|:-----------------|-------:|:---------------------------|----------------:|-----------:|----------:|:-------------------------|
| indian_pines     |      5 | mean_true_logit_margin     |        0.864631 |   0.957528 |  0.092896 | spectral_qnn_gated_proto |
| indian_pines     |      5 | median_true_logit_margin   |        1.30083  |   1.85723  |  0.556404 | spectral_qnn_gated_proto |
| indian_pines     |      5 | safe_logit_margin_rate     |        0.708041 |   0.711293 |  0.003252 | spectral_qnn_gated_proto |
| indian_pines     |      5 | mean_top1_top2_margin      |        1.66271  |   1.99542  |  0.332711 | spectral_qnn_gated_proto |
| indian_pines     |      5 | mean_true_prob_margin      |        0.313186 |   0.327064 |  0.013878 | spectral_qnn_gated_proto |
| indian_pines     |      5 | median_true_prob_margin    |        0.405995 |   0.571303 |  0.165308 | spectral_qnn_gated_proto |
| indian_pines     |      5 | safe_prob_margin_rate      |        0.694593 |   0.703791 |  0.009198 | spectral_qnn_gated_proto |
| indian_pines     |      5 | mean_top1_top2_prob_margin |        0.475668 |   0.543319 |  0.06765  | spectral_qnn_gated_proto |
| indian_pines     |      5 | OA                         |        0.720192 |   0.720511 |  0.000319 | spectral_qnn_gated_proto |
| indian_pines     |      5 | Macro-F1                   |        0.638053 |   0.641211 |  0.003159 | spectral_qnn_gated_proto |
| indian_pines     |      5 | negative_logit_margin_rate |        0.279808 |   0.279489 | -0.000319 | spectral_qnn_gated_proto |
| indian_pines     |      5 | low_logit_margin_rate      |        0.012151 |   0.009218 | -0.002933 | spectral_qnn_gated_proto |
| indian_pines     |      5 | negative_prob_margin_rate  |        0.279808 |   0.279489 | -0.000319 | spectral_qnn_gated_proto |
| indian_pines     |      5 | low_prob_margin_rate       |        0.025599 |   0.01672  | -0.008879 | spectral_qnn_gated_proto |
| indian_pines     |     10 | mean_true_logit_margin     |        1.88013  |   1.80259  | -0.077539 | hybridsn_small           |
| indian_pines     |     10 | median_true_logit_margin   |        2.12263  |   2.53592  |  0.413286 | spectral_qnn_gated_proto |
| indian_pines     |     10 | safe_logit_margin_rate     |        0.790289 |   0.801046 |  0.010757 | spectral_qnn_gated_proto |
| indian_pines     |     10 | mean_top1_top2_margin      |        2.41666  |   2.4217   |  0.005044 | spectral_qnn_gated_proto |
| indian_pines     |     10 | mean_true_prob_margin      |        0.476895 |   0.489033 |  0.012138 | spectral_qnn_gated_proto |
| indian_pines     |     10 | median_true_prob_margin    |        0.652701 |   0.737282 |  0.084581 | spectral_qnn_gated_proto |
| indian_pines     |     10 | safe_prob_margin_rate      |        0.783412 |   0.797949 |  0.014537 | spectral_qnn_gated_proto |
| indian_pines     |     10 | mean_top1_top2_prob_margin |        0.605293 |   0.640438 |  0.035145 | spectral_qnn_gated_proto |
| indian_pines     |     10 | OA                         |        0.801227 |   0.808847 |  0.00762  | spectral_qnn_gated_proto |
| indian_pines     |     10 | Macro-F1                   |        0.715332 |   0.718224 |  0.002893 | spectral_qnn_gated_proto |
| indian_pines     |     10 | negative_logit_margin_rate |        0.198773 |   0.191153 | -0.00762  | spectral_qnn_gated_proto |
| indian_pines     |     10 | low_logit_margin_rate      |        0.010938 |   0.007801 | -0.003137 | spectral_qnn_gated_proto |
| indian_pines     |     10 | negative_prob_margin_rate  |        0.198773 |   0.191153 | -0.00762  | spectral_qnn_gated_proto |
| indian_pines     |     10 | low_prob_margin_rate       |        0.017814 |   0.010898 | -0.006917 | spectral_qnn_gated_proto |
| pavia_university |      5 | mean_true_logit_margin     |        1.41716  |   1.12807  | -0.289089 | hybridsn_small           |
| pavia_university |      5 | median_true_logit_margin   |        1.7172   |   1.63139  | -0.085813 | hybridsn_small           |
| pavia_university |      5 | safe_logit_margin_rate     |        0.752961 |   0.763505 |  0.010544 | spectral_qnn_gated_proto |
| pavia_university |      5 | mean_top1_top2_margin      |        2.0116   |   1.68605  | -0.325547 | hybridsn_small           |
| pavia_university |      5 | mean_true_prob_margin      |        0.410117 |   0.345616 | -0.064501 | hybridsn_small           |
| pavia_university |      5 | median_true_prob_margin    |        0.548272 |   0.51714  | -0.031131 | hybridsn_small           |
| pavia_university |      5 | safe_prob_margin_rate      |        0.74328  |   0.757524 |  0.014245 | spectral_qnn_gated_proto |
| pavia_university |      5 | mean_top1_top2_prob_margin |        0.549902 |   0.490219 | -0.059683 | hybridsn_small           |
| pavia_university |      5 | OA                         |        0.76577  |   0.775277 |  0.009507 | spectral_qnn_gated_proto |
| pavia_university |      5 | Macro-F1                   |        0.761965 |   0.771094 |  0.009129 | spectral_qnn_gated_proto |
| pavia_university |      5 | negative_logit_margin_rate |        0.23423  |   0.224723 | -0.009507 | spectral_qnn_gated_proto |
| pavia_university |      5 | low_logit_margin_rate      |        0.012809 |   0.011773 | -0.001037 | spectral_qnn_gated_proto |
| pavia_university |      5 | negative_prob_margin_rate  |        0.23423  |   0.224723 | -0.009507 | spectral_qnn_gated_proto |
| pavia_university |      5 | low_prob_margin_rate       |        0.02249  |   0.017753 | -0.004737 | spectral_qnn_gated_proto |
| pavia_university |     10 | mean_true_logit_margin     |        1.51262  |   1.60143  |  0.088816 | spectral_qnn_gated_proto |
| pavia_university |     10 | median_true_logit_margin   |        1.68578  |   2.09161  |  0.405829 | spectral_qnn_gated_proto |
| pavia_university |     10 | safe_logit_margin_rate     |        0.805155 |   0.856062 |  0.050906 | spectral_qnn_gated_proto |
| pavia_university |     10 | mean_top1_top2_margin      |        1.83726  |   1.96944  |  0.13218  | spectral_qnn_gated_proto |
| pavia_university |     10 | mean_true_prob_margin      |        0.426845 |   0.480297 |  0.053452 | spectral_qnn_gated_proto |
| pavia_university |     10 | median_true_prob_margin    |        0.531483 |   0.642391 |  0.110908 | spectral_qnn_gated_proto |
| pavia_university |     10 | safe_prob_margin_rate      |        0.778059 |   0.853188 |  0.075129 | spectral_qnn_gated_proto |
| pavia_university |     10 | mean_top1_top2_prob_margin |        0.500917 |   0.574832 |  0.073915 | spectral_qnn_gated_proto |
| pavia_university |     10 | OA                         |        0.822617 |   0.863151 |  0.040534 | spectral_qnn_gated_proto |
| pavia_university |     10 | Macro-F1                   |        0.791976 |   0.861272 |  0.069296 | spectral_qnn_gated_proto |
| pavia_university |     10 | negative_logit_margin_rate |        0.177383 |   0.136849 | -0.040534 | spectral_qnn_gated_proto |
| pavia_university |     10 | low_logit_margin_rate      |        0.017462 |   0.00709  | -0.010372 | spectral_qnn_gated_proto |
| pavia_university |     10 | negative_prob_margin_rate  |        0.177383 |   0.136849 | -0.040534 | spectral_qnn_gated_proto |
| pavia_university |     10 | low_prob_margin_rate       |        0.044558 |   0.009963 | -0.034595 | spectral_qnn_gated_proto |
| salinas          |      5 | mean_true_logit_margin     |        1.64498  |   1.88427  |  0.239284 | spectral_qnn_gated_proto |
| salinas          |      5 | median_true_logit_margin   |        1.55157  |   2.14022  |  0.588655 | spectral_qnn_gated_proto |
| salinas          |      5 | safe_logit_margin_rate     |        0.76462  |   0.832496 |  0.067877 | spectral_qnn_gated_proto |
| salinas          |      5 | mean_top1_top2_margin      |        1.69821  |   1.96223  |  0.264023 | spectral_qnn_gated_proto |
| salinas          |      5 | mean_true_prob_margin      |        0.402037 |   0.484898 |  0.082862 | spectral_qnn_gated_proto |
| salinas          |      5 | median_true_prob_margin    |        0.470245 |   0.631108 |  0.160863 | spectral_qnn_gated_proto |
| salinas          |      5 | safe_prob_margin_rate      |        0.695266 |   0.82553  |  0.130264 | spectral_qnn_gated_proto |
| salinas          |      5 | mean_top1_top2_prob_margin |        0.416985 |   0.515453 |  0.098468 | spectral_qnn_gated_proto |
| salinas          |      5 | OA                         |        0.869328 |   0.881649 |  0.012322 | spectral_qnn_gated_proto |
| salinas          |      5 | Macro-F1                   |        0.896679 |   0.93356  |  0.036882 | spectral_qnn_gated_proto |
| salinas          |      5 | negative_logit_margin_rate |        0.130672 |   0.118351 | -0.012322 | spectral_qnn_gated_proto |
| salinas          |      5 | low_logit_margin_rate      |        0.104708 |   0.049153 | -0.055555 | spectral_qnn_gated_proto |
| salinas          |      5 | negative_prob_margin_rate  |        0.130672 |   0.118351 | -0.012322 | spectral_qnn_gated_proto |
| salinas          |      5 | low_prob_margin_rate       |        0.174061 |   0.056119 | -0.117942 | spectral_qnn_gated_proto |
| salinas          |     10 | mean_true_logit_margin     |        2.3213   |   2.23594  | -0.08536  | hybridsn_small           |
| salinas          |     10 | median_true_logit_margin   |        2.24398  |   2.65489  |  0.410915 | spectral_qnn_gated_proto |
| salinas          |     10 | safe_logit_margin_rate     |        0.835949 |   0.869762 |  0.033812 | spectral_qnn_gated_proto |
| salinas          |     10 | mean_top1_top2_margin      |        2.34644  |   2.29256  | -0.05388  | hybridsn_small           |
| salinas          |     10 | mean_true_prob_margin      |        0.509174 |   0.537574 |  0.0284   | spectral_qnn_gated_proto |
| salinas          |     10 | median_true_prob_margin    |        0.623227 |   0.702471 |  0.079245 | spectral_qnn_gated_proto |
| salinas          |     10 | safe_prob_margin_rate      |        0.82827  |   0.863049 |  0.034779 | spectral_qnn_gated_proto |
| salinas          |     10 | mean_top1_top2_prob_margin |        0.518478 |   0.560095 |  0.041617 | spectral_qnn_gated_proto |
| salinas          |     10 | OA                         |        0.936018 |   0.907216 | -0.028802 | hybridsn_small           |
| salinas          |     10 | Macro-F1                   |        0.954418 |   0.949634 | -0.004784 | hybridsn_small           |
| salinas          |     10 | negative_logit_margin_rate |        0.063982 |   0.092784 |  0.028802 | hybridsn_small           |
| salinas          |     10 | low_logit_margin_rate      |        0.100069 |   0.037455 | -0.062614 | spectral_qnn_gated_proto |
| salinas          |     10 | negative_prob_margin_rate  |        0.063982 |   0.092784 |  0.028802 | hybridsn_small           |
| salinas          |     10 | low_prob_margin_rate       |        0.107748 |   0.044167 | -0.06358  | spectral_qnn_gated_proto |

## 5. Paired seed 分析

- indian_pines 5-shot: mean_true_logit_margin: QNN better in 4/5 seeds; negative_logit_margin_rate: QNN better in 3/5 seeds; mean_true_prob_margin: QNN better in 4/5 seeds; Macro-F1: QNN better in 3/5 seeds
- indian_pines 10-shot: mean_true_logit_margin: QNN better in 3/5 seeds; negative_logit_margin_rate: QNN better in 3/5 seeds; mean_true_prob_margin: QNN better in 3/5 seeds; Macro-F1: QNN better in 3/5 seeds
- pavia_university 5-shot: mean_true_logit_margin: QNN better in 2/5 seeds; negative_logit_margin_rate: QNN better in 3/5 seeds; mean_true_prob_margin: QNN better in 2/5 seeds; Macro-F1: QNN better in 3/5 seeds
- pavia_university 10-shot: mean_true_logit_margin: QNN better in 2/5 seeds; negative_logit_margin_rate: QNN better in 4/5 seeds; mean_true_prob_margin: QNN better in 2/5 seeds; Macro-F1: QNN better in 5/5 seeds
- salinas 5-shot: mean_true_logit_margin: QNN better in 3/5 seeds; negative_logit_margin_rate: QNN better in 1/5 seeds; mean_true_prob_margin: QNN better in 3/5 seeds; Macro-F1: QNN better in 1/5 seeds
- salinas 10-shot: mean_true_logit_margin: QNN better in 3/5 seeds; negative_logit_margin_rate: QNN better in 1/5 seeds; mean_true_prob_margin: QNN better in 3/5 seeds; Macro-F1: QNN better in 2/5 seeds

## 6. 与 prototype geometry 结果的关系

- indian_pines 5-shot: 分类性能和 probability logit margin 同步提升，支持 classifier decision boundary 改善。 同时 prototype separation 未提升，说明改善更接近最终 classifier 边界而非 prototype geometry。
- indian_pines 10-shot: 分类性能和 probability logit margin 同步提升，支持 classifier decision boundary 改善。 同时 prototype separation 未提升，说明改善更接近最终 classifier 边界而非 prototype geometry。
- pavia_university 5-shot: 分类性能提升但平均 probability margin 未提升，提升可能集中于少数类别或特定样本。
- pavia_university 10-shot: 分类性能和 probability logit margin 同步提升，支持 classifier decision boundary 改善。 同时 prototype separation 未提升，说明改善更接近最终 classifier 边界而非 prototype geometry。
- salinas 5-shot: 分类性能和 probability logit margin 同步提升，支持 classifier decision boundary 改善。 同时 prototype separation 未提升，说明改善更接近最终 classifier 边界而非 prototype geometry。
- salinas 10-shot: 分类性能未提升。

## 7. Per-class 分析

- indian_pines 5-shot: F1 提升 7/16 类；negative logit margin rate 下降 8/16 类；mean true logit margin 提升 10/16 类。
- indian_pines 10-shot: F1 提升 9/16 类；negative logit margin rate 下降 7/16 类；mean true logit margin 提升 7/16 类。
- pavia_university 5-shot: F1 提升 4/9 类；negative logit margin rate 下降 7/9 类；mean true logit margin 提升 2/9 类。
- pavia_university 10-shot: F1 提升 7/9 类；negative logit margin rate 下降 7/9 类；mean true logit margin 提升 8/9 类。
- salinas 5-shot: F1 提升 13/16 类；negative logit margin rate 下降 12/16 类；mean true logit margin 提升 15/16 类。
- salinas 10-shot: F1 提升 9/16 类；negative logit margin rate 下降 9/16 类；mean true logit margin 提升 8/16 类。

## 8. 初步结论

在某些 dataset / shot 设置中，QNN 提升了最终分类 logits 或 probability margin，说明其可能改善 classifier decision boundary；该判断仍以对应设置为限。
同时满足 mean probability margin 提升且 negative logit margin rate 降低的设置为：indian_pines 5-shot、indian_pines 10-shot、pavia_university 10-shot、salinas 5-shot。

Prototype geometry 与 logit margin 联合表已写入 `metrics/geometry_vs_logit_margin_joint_summary.md`。
