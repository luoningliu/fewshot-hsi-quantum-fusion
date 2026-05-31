# 空间隔离实验汇总

## 实验目的

该实验用于回应 random pixel split 可能存在空间邻域泄漏的问题。空间隔离实验将 Indian Pines 按 5 x 5 网格划分为空间块，训练、验证、测试使用互不重叠的 block，因此测试样本与训练样本在空间上隔离。

## 划分设置

- 数据集：Indian Pines。
- Grid：5 x 5 spatial blocks。
- 训练 block 比例：0.20。
- 验证 block 比例：0.12。
- 测试 block：剩余 block。
- HybridSN spatial split seeds：0, 1。
- QNN head pilot：seed 0，先在 spatial train blocks 上训练 encoder，再冻结 encoder 比较 linear / MLP / residual QNN / gated residual QNN heads。

重要限制：该 strict block split 不强制每个类别都出现在训练区域，因此是较严格的空间泛化诊断，不应与 random pixel split 直接作为同一难度协议比较。

## 空间隔离 HybridSN 结果

| seed | best_val_macro_f1 | best_val_oa | best_val_aa | epochs_ran | training_time_seconds | train_size | validation_size | test_size | pca_evr_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 16.28 | 60.75 | 19.77 | 8 | 60.50 | 1580 | 1633 | 7036 | 0.97 |
| 1 | 12.40 | 98.09 | 12.43 | 10 | 96.59 | 2342 | 940 | 6967 | 0.96 |

Best spatial test: OA=33.16, AA=19.36, Macro-F1=13.03, Weighted-F1=24.02.

## 空间隔离 QNN head pilot

| run_id | best_val_macro_f1 | best_val_oa | best_val_aa | epochs_ran | training_time_seconds |
| --- | --- | --- | --- | --- | --- |
| linear_probe | 11.00 | 49.66 | 17.71 | 10 | 0.71 |
| mlp_h64 | 14.16 | 56.58 | 19.11 | 10 | 0.84 |
| residual_qnn_q4_l1_linear | 12.18 | 52.30 | 18.35 | 9 | 44.88 |
| gated_residual_qnn_q4_l1_linear | 11.25 | 50.77 | 18.02 | 10 | 50.35 |

Best head test: mlp_h64, OA=35.90, AA=20.68, Macro-F1=13.78, Weighted-F1=26.48.

QNN heads did not outperform the MLP head under this spatial split pilot. The best validation head is `mlp_h64`; residual QNN and gated residual QNN have lower validation Macro-F1 and much higher training cost.

## Random Split 与 Spatial Split 对照

| protocol | model | OA | AA | Macro-F1 | Weighted-F1 | note |
| --- | --- | --- | --- | --- | --- | --- |
| random pixel split | Tuned HybridSN | 98.80 | 97.27 | 97.19 | 98.81 | from existing random pixel split final report |
| few-shot random pixel split | HybridSN-small 10-shot | 80.12 | 88.64 | 71.53 | 80.87 | mean over seeds 0-4 |
| few-shot random pixel split | Spectral QNN + SupCon 10-shot | 81.07 | 89.05 | 72.26 | 81.79 | mean over seeds 0-4 |
| spatial block split | HybridSN spatial split | 33.16 | 19.36 | 13.03 | 24.02 | strict grid-block split, seed selected by validation Macro-F1 |
| spatial block split | mlp_h64 | 35.90 | 20.68 | 13.78 | 26.48 | frozen spatial encoder with head comparison; best head is validation-selected |

## Class Coverage 诊断

- seed 0/1 中，训练 split 缺失类别总记录数：17。
- 这解释了空间隔离下 Macro-F1 和 AA 大幅下降：部分测试类别在训练区域中没有样本，模型无法学习这些类。

训练集中缺失的类别：

| seed | class_name | train | validation | test |
| --- | --- | --- | --- | --- |
| 0 | Alfalfa | 0.00 | 0.00 | 46.00 |
| 0 | Buildings-Grass-Trees-Drives | 0.00 | 183.00 | 203.00 |
| 0 | Corn | 0.00 | 0.00 | 237.00 |
| 0 | Grass-pasture-mowed | 0.00 | 0.00 | 28.00 |
| 0 | Hay-windrowed | 0.00 | 0.00 | 478.00 |
| 0 | Soybean-clean | 0.00 | 175.00 | 418.00 |
| 0 | Soybean-notill | 0.00 | 0.00 | 972.00 |
| 0 | Stone-Steel-Towers | 0.00 | 0.00 | 93.00 |
| 0 | Wheat | 0.00 | 0.00 | 205.00 |
| 1 | Alfalfa | 0.00 | 0.00 | 46.00 |
| 1 | Corn | 0.00 | 0.00 | 237.00 |
| 1 | Corn-mintill | 0.00 | 0.00 | 830.00 |
| 1 | Grass-pasture | 0.00 | 17.00 | 466.00 |
| 1 | Grass-pasture-mowed | 0.00 | 0.00 | 28.00 |
| 1 | Hay-windrowed | 0.00 | 0.00 | 478.00 |
| 1 | Oats | 0.00 | 0.00 | 20.00 |
| 1 | Wheat | 0.00 | 0.00 | 205.00 |

## 结论

1. random pixel split 的结果显著高于 spatial block split，说明原随机像素协议可能高估了 patch-based HSI 模型的真实空间泛化能力。
2. 在 Indian Pines strict spatial split 下，HybridSN spatial test OA 仅 33.16%，Macro-F1 仅 13.03%，远低于 random pixel split 的 tuned HybridSN OA 98.80%、Macro-F1 97.19%。
3. QNN head pilot 也未显示空间隔离优势；最佳 head 是 MLP，QNN residual head 的验证 Macro-F1 低于 MLP。
4. 当前 spatial split 是严格诊断，不是最终主实验协议。正式论文中应把它作为补充实验，用来说明 random pixel split 的局限和模型空间泛化难度，而不是用它否定 few-shot 主结果。
5. 后续若要做更公平的 spatial few-shot 主实验，需要设计 class-balanced spatial split，保证每个类别在 train/validation/test 中均有样本。
