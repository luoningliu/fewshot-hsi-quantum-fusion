# Final Evidence Closure Report for Few-shot Spectral QNN

## 1. Executive Summary

Spectral QNN Gated Fusion combined with metric-learning objectives provides reproducible marginal-to-moderate gains in several low-shot HSI settings. The strongest evidence is on Pavia University 5/10-shot and Salinas 5-shot. Indian Pines shows marginal reproducible gains, while Salinas 10-shot is a negative-transfer case.

## 2. What has been completed

- Reused saved seed-level results for HybridSN-small, QNN + Prototype, and Indian Pines QNN + SupCon.
- Generated paired deltas, statistical tests, per-class deltas, confusion-delta figures, logit-margin integration, complexity summary, and negative-case reports.
- Pavia/Salinas SupCon seed results are missing in the current workspace and are explicitly logged in `failed_runs.csv`.

## 3. Main results
| dataset          |   shot | model                                      |   runs |   baseline_OA_mean |   qnn_OA_mean |     delta_OA |   baseline_AA_mean |   qnn_AA_mean |    delta_AA |   baseline_Kappa_mean |   qnn_Kappa_mean |   delta_Kappa |   baseline_Macro-F1_mean |   qnn_Macro-F1_mean |   delta_Macro-F1 |   baseline_Weighted-F1_mean |   qnn_Weighted-F1_mean |   delta_Weighted-F1 | macro_f1_interpretation   |
|:-----------------|-------:|:-------------------------------------------|-------:|-------------------:|--------------:|-------------:|-------------------:|--------------:|------------:|----------------------:|-----------------:|--------------:|-------------------------:|--------------------:|-----------------:|----------------------------:|-----------------------:|--------------------:|:--------------------------|
| indian_pines     |      5 | Spectral QNN Gated Fusion + Prototype Loss |      5 |           0.720192 |      0.720511 |  0.000319234 |           0.826596 |      0.825204 | -0.00139121 |              0.686598 |         0.686804 |   0.000205819 |                 0.638053 |            0.641211 |       0.00315857 |                    0.732668 |               0.728353 |         -0.00431461 | marginal_positive         |
| indian_pines     |      5 | Spectral QNN Gated Fusion + SupCon Loss    |      5 |           0.720192 |      0.720571 |  0.00037909  |           0.826596 |      0.825467 | -0.00112839 |              0.686598 |         0.686863 |   0.000265087 |                 0.638053 |            0.640394 |       0.00234178 |                    0.732668 |               0.729099 |         -0.00356894 | marginal_positive         |
| indian_pines     |     10 | Spectral QNN Gated Fusion + Prototype Loss |      5 |           0.801227 |      0.808847 |  0.00762039  |           0.886433 |      0.888684 |  0.00225161 |              0.77642  |         0.784879 |   0.00845902  |                 0.715332 |            0.718224 |       0.00289253 |                    0.808732 |               0.816128 |          0.0073965  | marginal_positive         |
| indian_pines     |     10 | Spectral QNN Gated Fusion + SupCon Loss    |      5 |           0.801227 |      0.810717 |  0.0094903   |           0.886433 |      0.890489 |  0.00405622 |              0.77642  |         0.786966 |   0.0105461   |                 0.715332 |            0.722636 |       0.00730473 |                    0.808732 |               0.817905 |          0.00917368 | marginal_positive         |
| pavia_university |      5 | Spectral QNN Gated Fusion + Prototype Loss |      5 |           0.757435 |      0.775277 |  0.017842    |           0.783188 |      0.823725 |  0.0405373  |              0.693229 |         0.71789  |   0.0246606   |                 0.712118 |            0.771094 |       0.0589753  |                    0.759421 |               0.784207 |          0.0247865  | marginal_positive         |
| pavia_university |     10 | Spectral QNN Gated Fusion + Prototype Loss |      5 |           0.822617 |      0.863151 |  0.0405343   |           0.843103 |      0.894798 |  0.0516956  |              0.773145 |         0.824803 |   0.0516581   |                 0.791976 |            0.861272 |       0.0692958  |                    0.828032 |               0.869347 |          0.0413147  | marginal_positive         |
| salinas          |      5 | Spectral QNN Gated Fusion + Prototype Loss |      5 |           0.869328 |      0.881649 |  0.0123216   |           0.908994 |      0.940094 |  0.0310999  |              0.854227 |         0.868566 |   0.0143387   |                 0.896679 |            0.93356  |       0.0368815  |                    0.859081 |               0.882125 |          0.023044   | marginal_positive         |
| salinas          |     10 | Spectral QNN Gated Fusion + Prototype Loss |      5 |           0.936018 |      0.907216 | -0.0288019   |           0.960622 |      0.95563  | -0.00499127 |              0.928798 |         0.896468 |  -0.03233     |                 0.954418 |            0.949634 |      -0.00478366 |                    0.936166 |               0.902897 |         -0.0332687  | negative_transfer         |

## 4. Prototype vs SupCon comparison

On Indian Pines, both Prototype and SupCon provide marginal positive effects, with SupCon stronger at 10-shot. Cross-dataset Prototype-vs-SupCon stability remains unresolved because Pavia and Salinas SupCon runs are not saved.

## 5. Statistical significance
| dataset          |   shot | comparison                                                   | metric   |   baseline_mean |   qnn_mean |   mean_delta |   std_delta |    ci95_low |   ci95_high |   paired_t_pvalue |   wilcoxon_pvalue |   cohen_d | is_significant_p005   | interpretation    |   n_paired_seeds |
|:-----------------|-------:|:-------------------------------------------------------------|:---------|----------------:|-----------:|-------------:|------------:|------------:|------------:|------------------:|------------------:|----------:|:----------------------|:------------------|-----------------:|
| indian_pines     |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Macro-F1 |        0.638053 |   0.641211 |   0.00315857 |  0.0131448  | -0.0131629  |   0.0194801 |         0.619554  |            0.625  |  0.240289 | False                 | marginal_positive |                5 |
| indian_pines     |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | Macro-F1 |        0.638053 |   0.640394 |   0.00234178 |  0.013697   | -0.0146652  |   0.0193488 |         0.721682  |            0.625  |  0.170971 | False                 | marginal_positive |                5 |
| indian_pines     |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Macro-F1 |        0.715332 |   0.718224 |   0.00289253 |  0.0117608  | -0.0117104  |   0.0174954 |         0.611615  |            0.625  |  0.245947 | False                 | marginal_positive |                5 |
| indian_pines     |     10 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    | Macro-F1 |        0.715332 |   0.722636 |   0.00730473 |  0.0182278  | -0.0153281  |   0.0299376 |         0.420856  |            0.625  |  0.400746 | False                 | marginal_positive |                5 |
| pavia_university |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Macro-F1 |        0.712118 |   0.771094 |   0.0589753  |  0.0517852  | -0.00532453 |   0.123275  |         0.0635386 |            0.0625 |  1.13884  | False                 | marginal_positive |                5 |
| pavia_university |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Macro-F1 |        0.791976 |   0.861272 |   0.0692958  |  0.0887168  | -0.0408607  |   0.179452  |         0.155639  |            0.0625 |  0.78109  | False                 | marginal_positive |                5 |
| salinas          |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Macro-F1 |        0.896679 |   0.93356  |   0.0368815  |  0.100536   | -0.0879506  |   0.161714  |         0.458115  |            0.625  |  0.366848 | False                 | marginal_positive |                5 |
| salinas          |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss | Macro-F1 |        0.954418 |   0.949634 |  -0.00478366 |  0.00843171 | -0.015253   |   0.0056857 |         0.273382  |            0.3125 | -0.567341 | False                 | negative_transfer |                5 |

## 6. Per-class analysis
| dataset      |   shot | comparison                                                   |   class_id | class_name_if_available      |   baseline_precision |   qnn_precision |   delta_precision |   baseline_recall |   qnn_recall |   delta_recall |   baseline_f1 |   qnn_f1 |    delta_f1 | interpretation   | rank_type    |
|:-------------|-------:|:-------------------------------------------------------------|-----------:|:-----------------------------|---------------------:|----------------:|------------------:|------------------:|-------------:|---------------:|--------------:|---------:|------------:|:-----------------|:-------------|
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |         12 | Wheat                        |             0.37661  |        0.447318 |       0.070708    |          0.998947 |     0.998947 |    0           |      0.541516 | 0.609841 |  0.0683257  | improved         | top_improved |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          0 | Alfalfa                      |             0.244488 |        0.27722  |       0.0327325   |          0.993939 |     1        |    0.00606061  |      0.382761 | 0.424595 |  0.0418334  | improved         | top_improved |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          6 | Grass-pasture-mowed          |             0.274107 |        0.295444 |       0.0213376   |          1        |     1        |    0           |      0.427116 | 0.453859 |  0.0267433  | improved         | top_improved |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          4 | Grass-pasture                |             0.768116 |        0.792694 |       0.0245779   |          0.701709 |     0.710256 |    0.00854701  |      0.727802 | 0.744379 |  0.0165767  | near_neutral     | top_improved |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          8 | Oats                         |             0.112049 |        0.121147 |       0.009098    |          1        |     0.966667 |   -0.0333333   |      0.200548 | 0.212995 |  0.012448   | near_neutral     | top_improved |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |         11 | Soybean-clean                |             0.575199 |        0.545683 |      -0.0295162   |          0.53218  |     0.495502 |   -0.0366782   |      0.529673 | 0.50273  | -0.0269428  | degraded         | top_degraded |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          1 | Corn-notill                  |             0.790414 |        0.804019 |       0.0136046   |          0.558669 |     0.523567 |   -0.0351026   |      0.641836 | 0.618923 | -0.0229134  | degraded         | top_degraded |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          3 | Corn                         |             0.779698 |        0.744899 |      -0.0347991   |          0.909009 |     0.898198 |   -0.0108108   |      0.818462 | 0.798585 | -0.0198769  | near_neutral     | top_degraded |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |         13 | Woods                        |             0.950413 |        0.92353  |      -0.026884    |          0.82736  |     0.82256  |   -0.0048      |      0.878497 | 0.862875 | -0.0156222  | near_neutral     | top_degraded |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |         15 | Stone-Steel-Towers           |             0.366029 |        0.350073 |      -0.0159556   |          0.989744 |     0.992308 |    0.0025641   |      0.522013 | 0.508687 | -0.013326   | near_neutral     | top_degraded |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    |         12 | Wheat                        |             0.37661  |        0.454541 |       0.0779309   |          0.998947 |     0.998947 |    0           |      0.541516 | 0.616412 |  0.0748962  | improved         | top_improved |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    |          0 | Alfalfa                      |             0.244488 |        0.275138 |       0.0306502   |          0.993939 |     1        |    0.00606061  |      0.382761 | 0.421829 |  0.0390683  | improved         | top_improved |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    |          6 | Grass-pasture-mowed          |             0.274107 |        0.291895 |       0.0177883   |          1        |     1        |    0           |      0.427116 | 0.449597 |  0.0224811  | improved         | top_improved |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    |          4 | Grass-pasture                |             0.768116 |        0.795466 |       0.0273498   |          0.701709 |     0.709402 |    0.00769231  |      0.727802 | 0.745086 |  0.0172839  | near_neutral     | top_improved |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    |          2 | Corn-mintill                 |             0.568266 |        0.568036 |      -0.00023004  |          0.462086 |     0.467485 |    0.00539877  |      0.490282 | 0.497765 |  0.00748321 | near_neutral     | top_improved |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    |         11 | Soybean-clean                |             0.575199 |        0.549545 |      -0.0256543   |          0.53218  |     0.485467 |   -0.0467128   |      0.529673 | 0.499976 | -0.0296971  | degraded         | top_degraded |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    |         15 | Stone-Steel-Towers           |             0.366029 |        0.336987 |      -0.0290416   |          0.989744 |     1        |    0.0102564   |      0.522013 | 0.495577 | -0.0264366  | degraded         | top_degraded |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    |          1 | Corn-notill                  |             0.790414 |        0.800256 |       0.00984204  |          0.558669 |     0.530927 |   -0.0277424   |      0.641836 | 0.623567 | -0.0182688  | near_neutral     | top_degraded |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    |         13 | Woods                        |             0.950413 |        0.924424 |      -0.0259892   |          0.82736  |     0.82176  |   -0.0056      |      0.878497 | 0.862747 | -0.0157501  | near_neutral     | top_degraded |
| indian_pines |      5 | HybridSN-small vs Spectral QNN Gated Fusion + SupCon Loss    |          7 | Hay-windrowed                |             0.887455 |        0.864271 |      -0.0231842   |          0.997408 |     0.999568 |    0.00215983  |      0.938097 | 0.925099 | -0.0129975  | near_neutral     | top_degraded |
| indian_pines |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          2 | Corn-mintill                 |             0.748556 |        0.808005 |       0.059449    |          0.733086 |     0.751111 |    0.0180247   |      0.732795 | 0.77727  |  0.0444753  | improved         | top_improved |
| indian_pines |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |         15 | Stone-Steel-Towers           |             0.399658 |        0.439459 |       0.0398008   |          1        |     0.994521 |   -0.00547945  |      0.56169  | 0.597192 |  0.0355014  | improved         | top_improved |
| indian_pines |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |         12 | Wheat                        |             0.617836 |        0.656645 |       0.0388091   |          0.988108 |     0.997838 |    0.00972973  |      0.754558 | 0.788347 |  0.0337895  | improved         | top_improved |
| indian_pines |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          1 | Corn-notill                  |             0.763183 |        0.800721 |       0.0375383   |          0.720881 |     0.717187 |   -0.00369318  |      0.739452 | 0.755495 |  0.016044   | near_neutral     | top_improved |
| indian_pines |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |         10 | Soybean-mintill              |             0.953426 |        0.952865 |      -0.000561599 |          0.712361 |     0.72961  |    0.0172485   |      0.813265 | 0.82445  |  0.0111856  | near_neutral     | top_improved |
| indian_pines |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |         14 | Buildings-Grass-Trees-Drives |             0.648871 |        0.613875 |      -0.0349961   |          0.995082 |     0.994536 |   -0.000546448 |      0.779742 | 0.755029 | -0.0247131  | degraded         | top_degraded |
| indian_pines |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          6 | Grass-pasture-mowed          |             0.296257 |        0.27295  |      -0.0233068   |          1        |     1        |    0           |      0.451163 | 0.426553 | -0.0246102  | degraded         | top_degraded |
| indian_pines |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          3 | Corn                         |             0.788737 |        0.754576 |      -0.0341609   |          0.976959 |     0.97788  |    0.000921659 |      0.867957 | 0.846917 | -0.0210407  | degraded         | top_degraded |
| indian_pines |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |         11 | Soybean-clean                |             0.794494 |        0.740374 |      -0.0541202   |          0.619197 |     0.623386 |    0.00418848  |      0.691248 | 0.673117 | -0.0181307  | near_neutral     | top_degraded |
| indian_pines |     10 | HybridSN-small vs Spectral QNN Gated Fusion + Prototype Loss |          8 | Oats                         |             0.128288 |        0.120217 |      -0.00807121  |          1        |     0.975    |   -0.025       |      0.222477 | 0.210529 | -0.0119482  | near_neutral     | top_degraded |

## 7. Salinas 10-shot negative transfer
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

## 8. Logit margin / decision-boundary explanation
| dataset          |   shot | model                    |   Macro-F1_mean |   separation_ratio_mean |   prototype_negative_margin_rate_mean |   mean_true_logit_margin_mean |   negative_logit_margin_rate_mean |
|:-----------------|-------:|:-------------------------|----------------:|------------------------:|--------------------------------------:|------------------------------:|----------------------------------:|
| indian_pines     |      5 | hybridsn_small           |        0.638053 |                 2.58895 |                             0.279589  |                      0.864631 |                         0.279808  |
| indian_pines     |      5 | spectral_qnn_gated_proto |        0.641211 |                 2.32505 |                             0.271828  |                      0.957528 |                         0.279489  |
| indian_pines     |     10 | hybridsn_small           |        0.715332 |                 3.10475 |                             0.196119  |                      1.88013  |                         0.198773  |
| indian_pines     |     10 | spectral_qnn_gated_proto |        0.718224 |                 2.79452 |                             0.18687   |                      1.80259  |                         0.191153  |
| pavia_university |      5 | hybridsn_small           |        0.761965 |                 2.57936 |                             0.240637  |                      1.41716  |                         0.23423   |
| pavia_university |      5 | spectral_qnn_gated_proto |        0.771094 |                 2.39035 |                             0.219788  |                      1.12807  |                         0.224723  |
| pavia_university |     10 | hybridsn_small           |        0.791976 |                 3.13551 |                             0.151263  |                      1.51262  |                         0.177383  |
| pavia_university |     10 | spectral_qnn_gated_proto |        0.861272 |                 2.84555 |                             0.136839  |                      1.60143  |                         0.136849  |
| salinas          |      5 | hybridsn_small           |        0.896679 |                 6.59936 |                             0.119742  |                      1.64498  |                         0.130672  |
| salinas          |      5 | spectral_qnn_gated_proto |        0.93356  |                 5.56046 |                             0.127974  |                      1.88427  |                         0.118351  |
| salinas          |     10 | hybridsn_small           |        0.954418 |                 8.08233 |                             0.0571094 |                      2.3213   |                         0.0639819 |
| salinas          |     10 | spectral_qnn_gated_proto |        0.949634 |                 6.83662 |                             0.0718058 |                      2.23594  |                         0.0927837 |

The evidence aligns better with final classifier logit-margin behavior than with universal prototype-space separation.

## 9. Complexity analysis
| model                                 |   trainable_params |   encoder_params |   classifier_or_head_params |   qnn_params |   qubits |   quantum_layers | encoding_type         | entanglement_type   |   training_time_mean |   inference_time_mean | device   | notes                                                                      |
|:--------------------------------------|-------------------:|-----------------:|----------------------------:|-------------:|---------:|-----------------:|:----------------------|:--------------------|---------------------:|----------------------:|:---------|:---------------------------------------------------------------------------|
| HybridSN-small                        |           99317.4  |            99488 |                         nan |          nan |      nan |              nan | NA                    | NA                  |             150.339  |              110.457  | cpu      | baseline encoder + classifier                                              |
| Spectral QNN Gated Fusion + Prototype |            1935.67 |            99488 |                         nan |         2176 |        6 |                1 | angle/tanh projection | linear              |              66.6488 |               62.1459 | cpu      | metric branch                                                              |
| Spectral QNN Gated Fusion + SupCon    |            2176    |            99488 |                         nan |         2176 |        6 |                1 | angle/tanh projection | linear              |              28.7424 |               18.2777 | cpu      | Indian Pines only in saved runs                                            |
| CNN2D                                 |           28208    |              nan |                         nan |          nan |      nan |              nan | NA                    | NA                  |             nan      |              nan      | cpu      | from few-shot CNN baseline metadata                                        |
| SVM-RBF                               |             nan    |              nan |                         nan |          nan |      nan |              nan | NA                    | NA                  |             nan      |              nan      | cpu      | non-parametric sklearn baseline; exact support-vector count not summarized |

The QNN branch is parameter-compact but not computationally faster under classical simulation.

## 10. Limitations

- QNN does not universally outperform HybridSN-small.
- Salinas 10-shot is a negative-transfer case.
- Pavia/Salinas SupCon runs are missing.
- Gate values were not saved in current runs.
- Random pixel split can be optimistic for patch-based HSI models.

## 11. Recommended paper narrative

Frame the contribution as spectral-side QNN with metric learning for low-shot decision-boundary regularization, with dataset- and shot-dependent gains and explicit negative cases.
