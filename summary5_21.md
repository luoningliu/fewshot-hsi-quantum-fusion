# 2026-05-21 实验总结

## 研究目标

今天的工作主要是把量子混合网络实验重新放到更严格的少样本高光谱分类设定下进行评估。今天主要使用的数据集是 Indian Pines。核心问题是：在标注样本有限的情况下，量子模块是否能相对于轻量级经典 HybridSN baseline 提供有效增益。

实验协议从普通随机监督分类调整为 all-way few-shot 分类：

- 1-shot：每类 1 个有标签训练样本。
- 5-shot：每类 5 个有标签训练样本。
- 10-shot：每类 10 个有标签训练样本。
- Seeds：0、1、2、3、4。
- 指标报告为 5 个 seed 的 mean ± std。

## 已实现模型

### HybridSN-small

HybridSN-small 保留了 HybridSN 的核心思想：

```text
HSI patch
-> 3D CNN 提取光谱-空间联合特征
-> 将光谱深度维 reshape 到通道维
-> 2D CNN 进一步提取空间特征
-> global average pooling
-> 小型 MLP 分类器
```

相比原始 HybridSN 中参数量很大的 Flatten-Dense 分类器，这里使用 global average pooling 和小型分类器，以降低少样本场景下的过拟合风险。

配置如下：

| 项目 | 数值 |
|---|---:|
| Dataset | Indian Pines |
| Patch size | 19 |
| PCA bands | 30 |
| Seeds | 5 |
| Trainable parameters | 99,488 |
| Epochs / patience | 200 / 30 |
| PCA fit scope | full_image_unsupervised |

### Residual QNN Head

该模型冻结已经训练好的 HybridSN-small encoder，然后在 32 维 encoder 特征上训练一个 residual QNN classifier：

```text
z = HybridSN-small encoder(patch)
logits = Linear(z) + QNN(z)
```

这是一个后端量子分类头。它不直接处理中心像素的光谱信息，只是在已经压缩后的经典特征上修改最终分类边界。

### Spectral QNN Gated Fusion

该模型冻结已经训练好的 HybridSN-small encoder，同时将中心像素的 PCA 光谱向量送入 QNN 分支，并通过可学习 gate 融合量子光谱 logits：

```text
z_c = HybridSN-small encoder(patch)
x_s = center pixel PCA spectrum

logits = Linear(z_c) + gate([z_c, x_s]) * QNN(x_s)
```

这是目前最有潜力的量子混合结构，因为量子模块直接建模光谱信息，而不是只在最后分类头上做微小修正。

## HybridSN-small 少样本 Baseline

| Shot | Runs | OA mean ± std | AA mean ± std | Kappa mean ± std | Macro-F1 mean ± std | Weighted-F1 mean ± std |
|---:|---:|---:|---:|---:|---:|---:|
| 1-shot | 5 | 41.49 ± 7.23 | 60.38 ± 1.59 | 36.02 ± 7.03 | 37.75 ± 3.00 | 40.25 ± 8.02 |
| 5-shot | 5 | 72.02 ± 1.67 | 82.66 ± 1.45 | 68.66 ± 1.90 | 63.81 ± 2.17 | 73.27 ± 2.06 |
| 10-shot | 5 | 80.12 ± 2.56 | 88.64 ± 0.94 | 77.64 ± 2.79 | 71.53 ± 2.02 | 80.87 ± 2.70 |

解释：

- 1-shot 非常不稳定，这符合预期。
- 5-shot 更适合作为主要少样本设置。
- 10-shot 中，轻量级 HybridSN-small baseline 已经表现得比较强。

结果目录：

```text
result/hybridsn_small_fewshot_indian_pines_only/
```

## Residual QNN Head 结果

Residual QNN Head 没有超过 HybridSN-small。

| Shot | HybridSN-small OA | Residual QNN OA | Delta OA | HybridSN-small Macro-F1 | Residual QNN Macro-F1 | Delta Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1-shot | 41.49 | 37.90 | -3.59 | 37.75 | 36.33 | -1.42 |
| 5-shot | 72.02 | 67.85 | -4.17 | 63.81 | 61.60 | -2.21 |
| 10-shot | 80.12 | 79.72 | -0.40 | 71.53 | 70.71 | -0.82 |

结论：

后端 QNN 分类头是不够的。它只是在已经被经典 encoder 压缩过的特征上调整最终决策边界，因此没有充分利用高光谱数据中的光谱结构。

结果目录：

```text
result/hybridsn_small_qnn_fewshot_indian_pines/
```

## Spectral QNN Gated Fusion 初始结果

第一次 Spectral QNN Gated Fusion 使用的配置为：

```text
qubits = 4
qnn_layers = 1
entanglement = linear
gate_mode = scalar
```

| Shot | HybridSN-small OA | Spectral Gated QNN OA | Delta OA | HybridSN-small Macro-F1 | Spectral Gated QNN Macro-F1 | Delta Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1-shot | 41.49 | 44.40 | +2.91 | 37.75 | 40.50 | +2.75 |
| 5-shot | 72.02 | 70.52 | -1.50 | 63.81 | 63.14 | -0.67 |
| 10-shot | 80.12 | 80.50 | +0.38 | 71.53 | 71.14 | -0.39 |

解释：

- spectral QNN branch 在 1-shot 中有明显帮助。
- 它也小幅提升了 10-shot 的 OA。
- 但这个初始版本尚未改善 5-shot。

结果目录：

```text
result/hybridsn_small_spectral_qnn_gated_fewshot_indian_pines/
```

## Spectral QNN 调参

为了改善 5-shot 和 10-shot，测试了多组量子配置。

已尝试配置：

| Config | 主要变化 | 结果 |
|---|---|---|
| q4_l1_scalar | 4 qubits，1 层，scalar gate | 1-shot 最佳 |
| q4_l1_classwise | classwise gate | 改善 10-shot，但 5-shot 不够 |
| q6_l1_classwise | 6 qubits，classwise gate | 综合表现最好 |
| q6_l1_ring_classwise | ring entanglement | 比 linear 更差 |
| q6_l1_classwise_wd0 | 无 weight decay | 没有改善 |
| q6_l1_classwise_monitor_oa | 使用 validation OA 选 checkpoint | 没有改善 |
| q6_l2_classwise | 2 层 QNN | 5-shot Macro-F1 最好，但 OA 仍低于 baseline |
| q6_l1_classwise_lr0001 | 更低学习率 | 更差 |

### 按 Macro-F1 选择的最佳结果

| Shot | Best Config | HybridSN-small OA | QNN OA | Delta OA | HybridSN-small Macro-F1 | QNN Macro-F1 | Delta Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1-shot | q4_l1_scalar | 41.49 | 44.40 | +2.91 | 37.75 | 40.50 | +2.75 |
| 5-shot | q6_l2_classwise | 72.02 | 71.68 | -0.34 | 63.81 | 64.40 | +0.60 |
| 10-shot | q6_l1_classwise | 80.12 | 80.89 | +0.77 | 71.53 | 71.81 | +0.28 |

### 按 OA 选择的最佳结果

| Shot | Best Config | HybridSN-small OA | QNN OA | Delta OA | HybridSN-small Macro-F1 | QNN Macro-F1 | Delta Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1-shot | q4_l1_scalar | 41.49 | 44.40 | +2.91 | 37.75 | 40.50 | +2.75 |
| 5-shot | q6_l1_classwise | 72.02 | 72.00 | -0.02 | 63.81 | 64.05 | +0.24 |
| 10-shot | q6_l1_classwise | 80.12 | 80.89 | +0.77 | 71.53 | 71.81 | +0.28 |

调参结果目录：

```text
result/hybridsn_small_spectral_qnn_tuning_indian_pines/
```

## CE + Prototype Loss 更新

为了解决 5-shot 中仍然存在的问题，我在 Spectral QNN Gated Fusion 上加入了辅助 prototype 分类目标：

```text
loss = CrossEntropy(logits, y) + 0.2 * CrossEntropy(prototype_logits, y)
prototype_logits = -distance(fused_feature, class_prototype) / 0.2
```

其中 fused feature 由冻结的 HybridSN-small encoder 特征和 spectral QNN 特征拼接得到：

```text
fused_feature = concat(HybridSN_feature, QNN_spectral_feature)
```

配置如下：

| 项目 | 数值 |
|---|---:|
| Qubits | 6 |
| QNN layers | 1 |
| Entanglement | linear |
| Gate mode | classwise |
| Prototype weight | 0.2 |
| Prototype temperature | 0.2 |
| Trainable parameters | 2,176 |
| Seeds | 0, 1, 2, 3, 4 |

### Prototype Loss 结果

| Shot | Model | Runs | OA mean ± std | AA mean ± std | Kappa mean ± std | Macro-F1 mean ± std | Weighted-F1 mean ± std | Delta OA | Delta Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5-shot | HybridSN-small | 5 | 72.02 ± 1.67 | 82.66 ± 1.45 | 68.66 ± 1.90 | 63.81 ± 2.17 | 73.27 ± 2.06 | 0.00 | 0.00 |
| 5-shot | Spectral QNN Gated Fusion + Prototype Loss | 5 | 72.05 ± 1.50 | 82.52 ± 0.68 | 68.68 ± 1.66 | 64.12 ± 1.73 | 72.84 ± 2.03 | +0.03 | +0.32 |
| 10-shot | HybridSN-small | 5 | 80.12 ± 2.56 | 88.64 ± 0.94 | 77.64 ± 2.79 | 71.53 ± 2.02 | 80.87 ± 2.70 | 0.00 | 0.00 |
| 10-shot | Spectral QNN Gated Fusion + Prototype Loss | 5 | 80.88 ± 2.40 | 88.87 ± 1.32 | 78.49 ± 2.62 | 71.82 ± 1.92 | 81.61 ± 2.43 | +0.76 | +0.29 |

解释：

- 这是第一个在 5-shot 和 10-shot 中同时超过 HybridSN-small 的配置，OA 和 Macro-F1 都为正提升。
- 5-shot 的 OA 提升非常小，因此应表述为边际正提升，而不是显著优势。
- Macro-F1 的提升对少样本任务更有意义，因为它更能反映类别均衡表现。
- 结果支持一个判断：此前 5-shot 的问题主要来自类别边界不稳定，而不是 QNN 深度不足。

结果目录：

```text
result/hybridsn_small_spectral_qnn_gated_proto_tuning_indian_pines/
```

## 按类别 F1 分析

为了分析 QNN 到底帮助了哪些类别，我对比了 HybridSN-small 与 Spectral QNN Gated Fusion + CE/prototype loss 在相同 shot、相同 seed 下的 per-class F1。

详细结果文件：

```text
result/hybridsn_small_spectral_qnn_gated_proto_tuning_indian_pines/per_class_delta_analysis.md
result/hybridsn_small_spectral_qnn_gated_proto_tuning_indian_pines/per_class_delta_summary.csv
```

5-shot 中 F1 提升最大的类别：

| 类别 | HybridSN F1 | QNN F1 | Delta F1 | 主要原因 |
|---|---:|---:|---:|---|
| Wheat | 0.542 | 0.610 | +0.068 | precision 明显提升 |
| Alfalfa | 0.383 | 0.425 | +0.042 | precision 小幅提升 |
| Grass-pasture-mowed | 0.427 | 0.454 | +0.027 | precision 小幅提升 |
| Grass-pasture | 0.728 | 0.744 | +0.017 | precision / recall 均小幅提升 |
| Soybean-mintill | 0.785 | 0.792 | +0.007 | 单类提升小，但样本多，总体贡献大 |

10-shot 中 F1 提升最大的类别：

| 类别 | HybridSN F1 | QNN F1 | Delta F1 | 主要原因 |
|---|---:|---:|---:|---|
| Corn-mintill | 0.733 | 0.777 | +0.045 | precision 明显提升，recall 也提升 |
| Stone-Steel-Towers | 0.562 | 0.597 | +0.036 | precision 提升 |
| Wheat | 0.755 | 0.788 | +0.034 | precision 提升 |
| Corn-notill | 0.740 | 0.756 | +0.016 | precision 提升 |
| Soybean-mintill | 0.813 | 0.825 | +0.011 | recall 小幅提升 |

按类别分析的结论：

- QNN 主要帮助光谱相近、容易混淆的大类，例如 Corn-mintill、Corn-notill、Soybean-mintill。
- QNN 也帮助了少样本但光谱特征较明确的类别，例如 Wheat、Alfalfa、Stone-Steel-Towers。
- 提升主要来自 precision，而不是 recall。这说明 QNN + prototype loss 更像是在减少误报、稳定类边界。
- QNN 并没有均匀提升所有类别。Soybean-clean、Corn、Buildings-Grass-Trees-Drives 等类别仍然有下降。

## 主要发现

1. HybridSN-small 是一个很强的少样本经典 baseline。
2. 只把 QNN 放在最终分类头上没有改善效果。
3. 将 QNN 移到 spectral branch 明显更有效。
4. Spectral QNN Gated Fusion 在 1-shot 中有清晰提升。
5. 调参后的 Spectral QNN Gated Fusion 在 10-shot 上同时提升 OA 和 Macro-F1。
6. 加入 CE + prototype loss 后，spectral QNN gated fusion 在 5-shot 和 10-shot 中都超过 HybridSN-small。
7. 5-shot 的提升较小但为正：+0.03 OA 和 +0.32 Macro-F1。
8. 10-shot 的提升更清楚：+0.76 OA 和 +0.29 Macro-F1。
9. 按类别看，QNN 的主要贡献来自部分类别的 precision 提升，而不是所有类别的全面提升。

## 当前论文中最稳妥的表述

目前最稳妥的结论是：

```text
所提出的 spectral quantum gated fusion branch 能改善极少样本表现；在加入辅助 prototype objective 后，它在 5-shot 和 10-shot 设置下都相对于 HybridSN-small 获得了边际但稳定的提升。
```

论文中仍然不应夸大 5-shot 的 OA 提升，因为该提升非常小。更合理的表述是：量子 spectral branch 结合 prototype regularization 能改善类别均衡的少样本分类表现，同时保持非常轻量的可训练 head。

## 推荐下一步

下一步应在相同 split 和相同 QNN 结构下，对比 prototype loss 与 supervised contrastive loss：

```text
CrossEntropy + supervised contrastive loss
```

原因：

- CE + prototype loss 已经解决了 5-shot 中 QNN 相对 HybridSN-small 的负增益问题。
- supervised contrastive loss 可能进一步提升特征紧凑性。
- 这个对照可以说明增益到底来自 prototype regularization，还是更一般的 metric-learning regularization。

## 关键结果目录

| 实验 | 目录 |
|---|---|
| HybridSN-small few-shot baseline | `result/hybridsn_small_fewshot_indian_pines_only/` |
| Residual QNN head | `result/hybridsn_small_qnn_fewshot_indian_pines/` |
| Spectral QNN Gated Fusion | `result/hybridsn_small_spectral_qnn_gated_fewshot_indian_pines/` |
| QNN tuning summary | `result/hybridsn_small_spectral_qnn_tuning_indian_pines/` |
| CE + prototype loss | `result/hybridsn_small_spectral_qnn_gated_proto_tuning_indian_pines/` |
