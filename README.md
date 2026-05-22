# HSI-QNN: 高光谱遥感图像分类中的混合量子-经典神经网络

本项目研究混合量子-经典神经网络在高光谱遥感图像分类任务中的有效性。

当前实验主线：

```text
Indian Pines / Pavia University / Salinas few-shot HSI classification
-> PCA + patch extraction
-> HybridSN-small encoder
-> Spectral QNN gated fusion
-> CE + prototype loss / supervised contrastive loss
-> OA / AA / Kappa / Macro-F1 / per-class F1
```

当前最核心问题：

```text
在 5/10-shot 少样本高光谱分类中，量子光谱分支结合 metric-learning objective 后，能否在强经典 HybridSN-small baseline 上带来可解释、可复现的增益？
```

当前最新结论：

```text
1. 只把 QNN 放在最终分类头上效果不好。
2. 将 QNN 放到中心像素光谱分支，并通过 gated fusion 与 HybridSN-small 特征融合，更符合高光谱任务结构。
3. CE + prototype loss 在 Pavia University 5/10-shot 和 Salinas 5-shot 上有明显增益；在 Salinas 10-shot 上出现负迁移。
4. Indian Pines 上 CE + supervised contrastive loss 与 CE + prototype loss 的公平对照表明，metric-learning objective 是少样本 QNN 提升的重要因素。
5. 当前结论不是“QNN 全面超过 HybridSN-small”，而是“量子 spectral branch 在部分少样本设置中改善类边界，且增益与数据集和 shot 难度有关”。
```

---

## 1. 数据集

当前使用三个高光谱遥感数据集：

| Dataset | Raw data | Ground truth |
|---|---|---|
| Indian Pines | `data/indian_pines/raw/Indian_pines.mat` | `data/indian_pines/raw/Indian_pines_gt.mat` |
| Pavia University | `data/pavia_university/raw/PaviaU.mat` | `data/pavia_university/raw/PaviaU_gt.mat` |
| Salinas | `data/salinas/raw/Salinas.mat` | `data/salinas/raw/Salinas_gt.mat` |

已确认数据 shape：

| Dataset | Cube shape | Classes |
|---|---:|---:|
| Indian Pines | `(145, 145, 220)` | 16 |
| Pavia University | `(610, 340, 103)` | 9 |
| Salinas | `(512, 217, 224)` | 16 |

背景标签 `0` 会被忽略。

---

## 2. Stage 1：数据协议

Stage 1 固定数据预处理协议。

默认设置：

```yaml
pca_components: 30
patch_size: 9
split: train / validation / test = 10% / 10% / 80%
seed: 42
normalization: per-band normalization, fit on all non-background pixels
PCA fitting: fit on all non-background pixels
padding: reflect padding
```

运行：

```bash
bash scripts/run_stage1_data_protocol.sh
```

输出：

```text
data/indian_pines/processed/indian_pines_pca30_patch9.npz
data/pavia_university/processed/pavia_university_pca30_patch9.npz
data/salinas/processed/salinas_pca30_patch9.npz

data/indian_pines/splits/split_seed42.json
data/pavia_university/splits/split_seed42.json
data/salinas/splits/split_seed42.json

result/stage1_data_protocol/report.md
```

Stage 1 结果：

| Dataset | Processed patch shape | Split train/val/test |
|---|---:|---:|
| Indian Pines | `(10249, 9, 9, 30)` | `1024 / 1025 / 8200` |
| Pavia University | `(42776, 9, 9, 30)` | `4277 / 4277 / 34222` |
| Salinas | `(54129, 9, 9, 30)` | `5412 / 5413 / 43304` |

报告：

```text
result/stage1_data_protocol/report.md
```

---

## 3. Stage 2：传统 baseline

传统 baseline 使用 PCA 后的中心像素光谱向量：

```text
30-dimensional PCA spectral vector -> classifier
```

模型：

```text
SVM-RBF
Random Forest
kNN
```

运行：

```bash
bash scripts/run_stage2_traditional_baselines.sh
```

结果：

```text
result/stage2_traditional_baselines/summary_table.csv
```

Indian Pines 初始传统 baseline：

| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| SVM-RBF | 66.07 | 56.11 | 61.20 | 57.57 | 65.54 |
| Random Forest | 63.29 | 47.36 | 56.81 | 47.46 | 59.21 |
| kNN | 60.71 | 47.74 | 54.48 | 47.93 | 58.05 |

---

## 4. Stage 3：深度学习 baseline

模型：

```text
1D-CNN
2D-CNN
3D-CNN
HybridSN-style CNN
```

运行：

```bash
bash scripts/run_stage3_deep_baselines.sh
```

结果：

```text
result/stage3_deep_baselines/summary_table.csv
```

Indian Pines 初始深度 baseline：

| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| 1D-CNN | 13.98 | 10.57 | 2.29 | 2.03 | 3.99 |
| 2D-CNN | 75.77 | 59.93 | 72.06 | 60.95 | 74.51 |
| 3D-CNN | 27.00 | 12.11 | 10.48 | 4.48 | 12.47 |
| HybridSN | 81.00 | 68.69 | 78.27 | 69.97 | 80.48 |

注意：这些是第一轮 CPU 初跑配置，并非系统调参结果。

### 4.1 Indian Pines 1D-CNN / 3D-CNN debug 修复

检查发现 Indian Pines 的 1D-CNN 和 3D-CNN 初始结果异常低，主要原因不是标签整体错乱，而是：

- Stage 3 初跑只使用 `15 epochs + patience 4`，训练明显不足。
- 1D/3D CNN forward 缺少输入 shape 断言，错误维度可能静默进入训练。
- 训练日志缺少首批 tensor shape、logits shape、loss、梯度范数、train accuracy 和 validation loss。
- Stage 1 为固定协议，normalization/PCA 默认 fit on all non-background pixels；debug run 额外使用 train-only normalization/PCA 做严格监督评估。

已修复：

```text
src/models/classical/cnn1d.py
src/models/classical/cnn3d.py
scripts/debug_cnn1d_cnn3d_indian_pines.py
```

正确输入约定：

```text
1D-CNN:
dataset input = [B, 30]
Conv1d input  = [B, 1, 30]

3D-CNN:
dataset input = [B, 9, 9, 30]
Conv3d input  = [B, 1, 30, 9, 9]
```

debug 运行：

```bash
python scripts/debug_cnn1d_cnn3d_indian_pines.py \
  --epochs 80 \
  --patience 25 \
  --tiny-epochs 150 \
  --tiny-samples 32 \
  --output result/debug_cnn1d_cnn3d_indian_pines
```

输出：

```text
result/debug_cnn1d_cnn3d_indian_pines/
```

Indian Pines debug 修复后结果：

| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 | Tiny subset overfit |
|---|---:|---:|---:|---:|---:|---:|
| 1D-CNN | 59.28 | 46.99 | 52.97 | 46.33 | 56.63 | 100.00 |
| 3D-CNN | 63.68 | 49.08 | 57.56 | 49.00 | 59.92 | 100.00 |

修复报告：

```text
result/debug_cnn1d_cnn3d_indian_pines/bugfix_report.md
```

注意：debug run 是 CPU 友好的最小修复验证，不是最终论文级网格调参。Indian Pines 类别不均衡明显，正式比较时应同时报告 OA、AA、Macro-F1、Weighted-F1 和 per-class metrics，并只用 validation metric 选择 checkpoint。

---

## 5. Stage 4：HybridSN embedding + 分类头

Stage 4 使用以下流程：

```text
HSI patch
-> HybridSN encoder
-> cached embedding
-> classifier head
```

这样可以公平比较不同分类头：

```text
Linear
MLP
Bottleneck
QNN
Residual QNN
```

缓存的 Indian Pines embedding：

```text
result/stage4_qnn_classifier/indian_pines/hybridsn_embeddings.npz
```

---

## 6. QNN 输入设计修正

最初 QNN 输入为：

```text
embedding -> Linear -> tanh -> QNN
```

检查发现 HybridSN embedding 数值范围较大：

```text
mean/std ≈ -0.081 / 5.466
min/max ≈ -22.38 / 23.83
```

直接送入 `tanh` 后严重饱和：

```text
|tanh| > 0.95 的比例约 55% 到 94%
```

因此改为：

```text
embedding
-> LayerNorm
-> Linear projection
-> π * tanh
-> QNN
```

修正后：

```text
|tanh| > 0.95 的比例 = 0%
```

同时加入 residual 结构：

```text
logits = Linear(embedding) + QNN(embedding)
```

这一步是 QNN 性能从失败状态恢复到可竞争水平的关键。

---

## 7. QNN 加速设置

PennyLane 后端使用：

```python
qml.device("lightning.qubit", wires=qubits)
```

梯度方法：

```python
diff_method="adjoint"
```

相比默认 `default.qubit`，`lightning.qubit + adjoint` 在当前 CPU 环境下更适合 analytic expectation simulation。

当前环境无可用 GPU，因此没有使用 `lightning.gpu`。

---

## 8. Indian Pines：标准监督设置下的最佳结果汇总

每种模型只保留当前已经获得的最佳结果。

完整文件：

```text
result/final_report/indian_pines_best_only_results.csv
result/final_report/indian_pines_best_only_results_percent.csv
result/final_report/indian_pines_best_only_results.md
```

按 `Macro-F1` 排序：

| Model family | Best model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|
| MLP Head | `mlp_h256_d0.0_lr0.003` | 84.05 | 77.19 | 81.73 | 78.55 | 83.77 |
| Residual QNN | `Residual QNN full` | 78.34 | 78.96 | 75.55 | 77.41 | 78.39 |
| Residual Reupload Multiobs QNN | `Residual Reupload Multiobs QNN full` | 78.73 | 74.25 | 75.32 | 76.13 | 77.08 |
| Bottleneck Head | `bottleneck_b32_relu_lr0.01` | 81.60 | 75.52 | 79.03 | 75.68 | 81.40 |
| Linear Head | `linear_lr0.01` | 80.05 | 70.63 | 76.94 | 73.33 | 79.43 |
| HybridSN | `hybridsn` | 81.00 | 68.69 | 78.27 | 69.97 | 80.48 |
| 2D-CNN | `cnn2d` | 75.77 | 59.93 | 72.06 | 60.95 | 74.51 |
| SVM-RBF | `svm_C10_gammascale` | 68.40 | 56.13 | 63.48 | 58.36 | 67.07 |
| kNN | `knn` | 60.71 | 47.74 | 54.48 | 47.93 | 58.05 |
| Random Forest | `random_forest` | 63.29 | 47.36 | 56.81 | 47.46 | 59.21 |
| 3D-CNN | `cnn3d` | 27.00 | 12.11 | 10.48 | 4.48 | 12.47 |
| 1D-CNN | `cnn1d` | 13.98 | 10.57 | 2.29 | 2.03 | 3.99 |

---

## 9. Indian Pines：关键结论

当前最强整体模型是 tuned MLP：

```text
OA       = 84.05
Macro-F1 = 78.55
```

Residual QNN 的结果：

```text
OA       = 78.34
AA       = 78.96
Macro-F1 = 77.41
```

结论：

```text
Residual QNN 没有全面超过 tuned MLP。
```

但 Residual QNN 的 AA 更高：

```text
Residual QNN AA = 78.96
Tuned MLP AA    = 77.19
```

这说明 Residual QNN 可能对类别平均召回更有优势，值得进一步做 per-class recall / F1 分析。

---

## 10. Data Re-uploading + Multi-observable QNN

我们也测试了更复杂的 QNN：

```text
Residual Data Re-uploading QNN
+ Multi-observable readout
```

结构：

```text
embedding
-> LayerNorm
-> Linear projection
-> π * tanh
-> repeated data encoding + trainable quantum blocks
-> measurements: Z_i and Z_i Z_{i+1}
-> classifier
-> residual add with Linear head
```

结果：

| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Residual QNN | 78.34 | 78.96 | 75.55 | 77.41 | 78.39 |
| Reupload + Multiobs Residual QNN | 78.73 | 74.25 | 75.32 | 76.13 | 77.08 |

结论：

```text
更复杂的 data re-uploading + multi-observable 结构没有超过普通 Residual QNN。
```

因此当前 Indian Pines 上最好的 QNN 仍然是：

```text
Residual QNN
qubits = 6
layers = 2
entanglement = linear
LayerNorm + π*tanh angle encoding
selection_metric = validation_Macro-F1
```

---

## 11. 跨数据集 few-shot：HybridSN-small 与 Spectral QNN 最新结果

为了更贴近本文“量子混合网络在少样本高光谱分类中的价值”这一研究目标，后续实验从标准随机监督分类切换到 all-way few-shot 分类。

协议：

```text
Datasets: Indian Pines / Pavia University / Salinas
Main shots: 5 / 10 samples per class
Seeds: 0, 1, 2, 3, 4
Patch size: 19
PCA bands: 30
Metrics: mean ± std over 5 seeds
```

Indian Pines 仍保留 1-shot 分析；当前跨数据集主比较集中在更稳定、更有论文参考价值的 5-shot 和 10-shot。

### 11.1 HybridSN-small baseline

HybridSN-small 保留原始 HybridSN 的核心思想：

```text
HSI patch
-> 3D CNN 提取光谱-空间联合特征
-> reshape spectral depth into channels
-> 2D CNN 提取空间特征
-> global average pooling
-> small MLP classifier
```

与原始 HybridSN 相比，HybridSN-small 去掉了大型 Flatten-Dense 分类器，使用 global average pooling 降低参数量和过拟合风险。

| Shot | Runs | OA mean ± std | AA mean ± std | Kappa mean ± std | Macro-F1 mean ± std | Weighted-F1 mean ± std |
|---:|---:|---:|---:|---:|---:|---:|
| 1-shot | 5 | 41.49 ± 7.23 | 60.38 ± 1.59 | 36.02 ± 7.03 | 37.75 ± 3.00 | 40.25 ± 8.02 |
| 5-shot | 5 | 72.02 ± 1.67 | 82.66 ± 1.45 | 68.66 ± 1.90 | 63.81 ± 2.17 | 73.27 ± 2.06 |
| 10-shot | 5 | 80.12 ± 2.56 | 88.64 ± 0.94 | 77.64 ± 2.79 | 71.53 ± 2.02 | 80.87 ± 2.70 |

结果目录：

```text
result/hybridsn_small_fewshot_indian_pines_only/
```

### 11.2 QNN 结构演化

已尝试的 QNN 结构包括：

| 模型 | 结构 | 结论 |
|---|---|---|
| Residual QNN Head | `logits = Linear(z) + QNN(z)` | 不够有效，只作用于已经压缩的 classical embedding |
| Spectral QNN Gated Fusion | `Linear(z_c) + gate([z_c, x_s]) * QNN(x_s)` | 明显优于 final-head QNN，能利用中心像素光谱信息 |
| Spectral QNN Gated Fusion + Prototype Loss | CE + prototype objective | 当前最好的 few-shot QNN 结构 |

当前采用的 Spectral QNN Gated Fusion：

```text
z_c = HybridSN-small encoder(patch)
x_s = center pixel PCA spectrum

logits = Linear(z_c) + gate([z_c, x_s]) * QNN(x_s)
```

加入 prototype loss 后：

```text
loss = CrossEntropy(logits, y) + 0.2 * CrossEntropy(prototype_logits, y)
prototype_logits = -distance(fused_feature, class_prototype) / 0.2
fused_feature = concat(HybridSN_feature, QNN_spectral_feature)
```

QNN 配置：

| 项目 | 数值 |
|---|---:|
| Qubits | 6 |
| QNN layers | 1 |
| Entanglement | linear |
| Gate mode | classwise |
| Prototype weight | 0.2 |
| Prototype temperature | 0.2 |
| Trainable parameters | 2,176 |

### 11.3 Metric-learning objective 对照与跨数据集结果

Indian Pines 上补充了 CE + supervised contrastive loss，用来和 CE + prototype loss 做同构公平对照。两者使用同一 Spectral QNN Gated Fusion 结构，只替换辅助 metric-learning objective。

| Dataset | Shot | Model | Runs | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Indian Pines | 5 | HybridSN-small | 5 | 72.02 ± 1.67 | 82.66 ± 1.45 | 68.66 ± 1.90 | 63.81 ± 2.17 | 73.27 ± 2.06 |
| Indian Pines | 5 | QNN + Prototype | 5 | 72.05 ± 1.50 | 82.52 ± 0.68 | 68.68 ± 1.66 | 64.12 ± 1.73 | 72.84 ± 2.03 |
| Indian Pines | 5 | QNN + SupCon | 5 | 72.06 ± 1.42 | 82.55 ± 0.60 | 68.69 ± 1.57 | 64.04 ± 1.44 | 72.91 ± 1.97 |
| Indian Pines | 10 | HybridSN-small | 5 | 80.12 ± 2.56 | 88.64 ± 0.94 | 77.64 ± 2.79 | 71.53 ± 2.02 | 80.87 ± 2.70 |
| Indian Pines | 10 | QNN + Prototype | 5 | 80.88 ± 2.40 | 88.87 ± 1.32 | 78.49 ± 2.62 | 71.82 ± 1.92 | 81.61 ± 2.43 |
| Indian Pines | 10 | QNN + SupCon | 5 | 81.07 ± 2.32 | 89.05 ± 1.38 | 78.70 ± 2.54 | 72.26 ± 2.08 | 81.79 ± 2.33 |
| Pavia University | 5 | HybridSN-small | 5 | 75.74 ± 2.89 | 78.32 ± 3.84 | 69.32 ± 3.06 | 71.21 ± 4.46 | 75.94 ± 2.89 |
| Pavia University | 5 | QNN + Prototype | 5 | 77.53 ± 3.48 | 82.37 ± 2.83 | 71.79 ± 3.92 | 77.11 ± 2.17 | 78.42 ± 3.38 |
| Pavia University | 10 | HybridSN-small | 5 | 82.26 ± 4.92 | 84.31 ± 5.82 | 77.31 ± 6.20 | 79.20 ± 8.14 | 82.80 ± 5.19 |
| Pavia University | 10 | QNN + Prototype | 5 | 86.32 ± 2.59 | 89.48 ± 2.21 | 82.48 ± 3.18 | 86.13 ± 2.22 | 86.93 ± 2.28 |
| Salinas | 5 | HybridSN-small | 5 | 86.93 ± 7.11 | 90.90 ± 8.03 | 85.42 ± 8.04 | 89.67 ± 9.21 | 85.91 ± 9.25 |
| Salinas | 5 | QNN + Prototype | 5 | 88.16 ± 2.09 | 94.01 ± 0.86 | 86.86 ± 2.33 | 93.36 ± 0.78 | 88.21 ± 2.09 |
| Salinas | 10 | HybridSN-small | 5 | 93.60 ± 1.59 | 96.06 ± 1.46 | 92.88 ± 1.76 | 95.44 ± 1.55 | 93.62 ± 1.57 |
| Salinas | 10 | QNN + Prototype | 5 | 90.72 ± 3.77 | 95.56 ± 1.71 | 89.65 ± 4.24 | 94.96 ± 1.79 | 90.29 ± 4.43 |

解释：

- Indian Pines 5-shot 中，prototype 与 SupCon 都只带来边际增益；prototype 的 Macro-F1 略高，SupCon 的 OA 略高。
- Indian Pines 10-shot 中，SupCon 优于 prototype，OA 相对 HybridSN-small 提升 +0.95，Macro-F1 提升 +0.73。
- Pavia University 中，QNN + prototype 在 5-shot 与 10-shot 都明显超过 HybridSN-small，Macro-F1 分别提升 +5.90 和 +6.93。
- Salinas 5-shot 中，QNN + prototype 提升 Macro-F1 +3.69，并显著降低方差。
- Salinas 10-shot 中，QNN + prototype 低于 HybridSN-small，说明在 classical baseline 已经接近饱和时，当前量子 head 可能产生负迁移。

结果目录：

```text
result/fewshot_metric_loss_cross_dataset_summary/
```

### 11.4 按类别 F1 分析

为了分析 QNN 到底帮助了哪些类，已对比 HybridSN-small 与 Spectral QNN Gated Fusion + CE/prototype loss 在相同 shot、相同 seed 下的 per-class F1。

详细文件：

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

```text
QNN 主要帮助光谱相近、容易混淆的大类，例如 Corn-mintill、Corn-notill、Soybean-mintill；
也帮助了少样本但光谱特征较明确的类别，例如 Wheat、Alfalfa、Stone-Steel-Towers。
提升主要来自 precision，而不是 recall，说明 QNN + prototype loss 更像是在减少误报、稳定类边界。
```

QNN 并没有均匀提升所有类别。Soybean-clean、Corn、Buildings-Grass-Trees-Drives 等类别仍有下降。

### 11.5 当前少样本结论

当前最稳妥的论文表述：

```text
Spectral QNN Gated Fusion 在少样本场景下并非对所有数据集和 shot 都稳定优于 HybridSN-small；
但在 Indian Pines、Pavia University 和 Salinas 5-shot 上均显示出正向作用，
尤其在 Pavia University 上提升明显。
CE + SupCon 与 CE + prototype 的公平对照说明，metric-learning objective 是提升少样本 QNN 表现的重要因素。
```

需要谨慎的地方：

```text
不能宣称 QNN 在所有少样本高光谱任务中全面超过 HybridSN-small。
更合理的表述是：量子 spectral branch 需要与少样本 metric-learning objective 结合，才能更稳定地改善类边界；
其优势更容易出现在 5-shot 或较不稳定的少样本设置中，而不是 classical baseline 已接近饱和的设置。
```

---

## 12. 常用命令

### 运行 Stage 1

```bash
bash scripts/run_stage1_data_protocol.sh
```

### 运行 Stage 2

```bash
bash scripts/run_stage2_traditional_baselines.sh
```

### 运行 Stage 3

```bash
bash scripts/run_stage3_deep_baselines.sh
```

### 运行 Stage 4 初始 QNN 分类头实验

```bash
bash scripts/run_stage4_qnn_classifier.sh
```

### 运行 Indian Pines QNN sweep

```bash
bash scripts/run_stage4_qnn_sweep_indian_pines.sh
```

### 运行 Indian Pines Residual QNN full test

```bash
bash scripts/run_stage4_residual_qnn_full_indian_pines.sh
```

### 运行 Indian Pines tuned baseline

```bash
bash scripts/run_stage4_tuned_baselines_indian_pines.sh
```

### 运行 Data Re-uploading + Multi-observable QNN

```bash
bash scripts/run_stage4_reupload_multiobs_qnn_full_indian_pines.sh
```

### 运行 HybridSN-small few-shot baseline

```bash
python scripts/run_hybridsn_small_fewshot.py \
  --dataset indian_pines \
  --data_root ./data \
  --shots 1 5 10 \
  --seeds 0 1 2 3 4 \
  --patch_size 19 \
  --pca_bands 30 \
  --batch_size 64 \
  --epochs 200 \
  --patience 30 \
  --output_root ./result
```

### 运行 Spectral QNN Gated Fusion + metric-learning loss

```bash
python scripts/run_hybridsn_small_spectral_qnn_gated_metric_fewshot.py \
  --datasets indian_pines \
  --shots 5 10 \
  --seeds 0 1 2 3 4 \
  --epochs 60 \
  --patience 10 \
  --lr 0.003 \
  --weight_decay 0.0001 \
  --loss_mode prototype \
  --metric_weight 0.2 \
  --temperature 0.2 \
  --qubits 6 \
  --qnn_layers 1 \
  --entanglement linear \
  --gate_mode classwise \
  --batch_size 16 \
  --test_batch_size 128
```

### 编译检查

```bash
python -m compileall src scripts
```

---

## 13. 重要结果文件

```text
result/stage1_data_protocol/report.md
result/stage2_traditional_baselines/summary_table.csv
result/stage3_deep_baselines/summary_table.csv
result/stage4_qnn_classifier/summary_table.csv
result/stage4_residual_qnn_full_indian_pines/summary_table.csv
result/stage4_reupload_multiobs_qnn_full_indian_pines/summary_table.csv
result/stage4_tuned_baselines_indian_pines/best_by_family.csv
result/final_report/indian_pines_best_only_results.md
result/hybridsn_small_fewshot_indian_pines_only/metrics/summary_by_shot.md
result/hybridsn_small_spectral_qnn_tuning_indian_pines/tuning_report.md
result/hybridsn_small_spectral_qnn_gated_proto_tuning_indian_pines/report.md
result/hybridsn_small_spectral_qnn_gated_proto_tuning_indian_pines/per_class_delta_analysis.md
result/fewshot_metric_loss_cross_dataset_summary/report_zh.md
result/fewshot_metric_loss_cross_dataset_summary/all_model_summary.csv
result/fewshot_metric_loss_cross_dataset_summary/comparison_vs_hybridsn_small.csv
summary5_21.md
```

---

## 14. 当前尚未完成的工作

当前 5/10-shot 少样本结果已扩展到 Indian Pines、Pavia University 和 Salinas。

后续建议：

1. 将 `CrossEntropy + supervised contrastive loss` 扩展到 Pavia University 和 Salinas，判断 SupCon 是否能缓解 Salinas 10-shot 负迁移；
2. 对 Salinas 10-shot 做 per-class 和 confusion matrix 分析，定位 QNN 下降来自哪些类；
3. 加入 same-parameter classical spectral branch，对比量子分支是否超过同等参数量经典分支；
4. 对 Pavia University 的明显提升做 per-class F1 分析，确认增益来源；
5. 如果论文强调 5-shot 增益，应同步报告 mean ± std、Macro-F1、AA 和 per-class F1，而不能只报 OA。

---

## 15. 当前可写的谨慎结论

当前实验支持以下结论：

```text
1. 原始 QNN 分类头设计效果有限，主要问题之一是输入饱和和量子层位置不合理。
2. LayerNorm + π*tanh angle encoding + residual connection 能显著改善标准监督设置下的 QNN。
3. 在少样本设置下，final-head QNN 不足以超过 HybridSN-small。
4. 将 QNN 放到中心像素 spectral branch，并使用 gated fusion，比 final-head QNN 更有效。
5. CE + prototype loss 在 Pavia University 5/10-shot 与 Salinas 5-shot 上有明显增益，但 Salinas 10-shot 出现负迁移。
6. Indian Pines 上 CE + SupCon 与 CE + prototype 的公平对照说明，metric-learning objective 是少样本 QNN 表现的重要因素。
7. QNN 的增益不是均匀作用于所有类别，而是与类别混淆、数据集难度和 shot 设置有关。
```

因此目前最稳妥的论文表述是：

```text
量子 spectral branch 结合 metric-learning regularization 在部分少样本高光谱设置中显示出有效增益，尤其在 Pavia University 和 5-shot 场景中更明显；但它并未在所有数据集和 shot 下稳定超过 HybridSN-small，仍需通过 SupCon 跨数据集对照、同参数经典分支和负迁移分析进一步验证。
```
