# 2026-05-25 项目结果总汇

## 0. 总体判断

截至 2026-05-25，本项目已经形成两条不同层级的结果线：

1. **完整监督 / random pixel split 结果线**：用于验证数据协议、模型实现、HybridSN 编码器和 QNN head 的基础可行性。该线中 tuned HybridSN 在 Indian Pines 上已经达到很高性能，但 random pixel split 存在空间邻域信息泄漏风险，不能单独作为论文最核心论证。
2. **all-way few-shot 结果线**：当前更适合作为论文主线。该线评估 Indian Pines、Pavia University、Salinas 的 1/5/10-shot 设置，重点比较 HybridSN-small、传统/普通 CNN baseline，以及 Spectral QNN Gated Fusion + metric-learning objective。

当前最稳妥的论文结论不是“QNN 全面超过经典模型”，而是：

```text
Spectral QNN branch 只有放到中心像素光谱分支，并结合 prototype loss 或 supervised contrastive loss 等 metric-learning objective 后，才在部分 few-shot HSI 设置中显示稳定正向作用。
该作用在 Pavia University 5/10-shot 和 Salinas 5-shot 上最明显，在 Indian Pines 上是边际提升，在 Salinas 10-shot 上出现负迁移。
```

---

## 1. 数据协议结果

主数据集固定协议输出在：

```text
result/stage1_data_protocol/
```

| Dataset | Raw shape | Samples | Patch shape | PCA | EVR sum | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Indian Pines | `[145, 145, 220]` | 10249 | `[9, 9, 30]` | 30 | 0.961312 | 1024 | 1025 | 8200 |
| Pavia University | `[610, 340, 103]` | 42776 | `[9, 9, 30]` | 30 | 0.999679 | 4277 | 4277 | 34222 |
| Salinas | `[512, 217, 224]` | 54129 | `[9, 9, 30]` | 30 | 0.998638 | 5412 | 5413 | 43304 |

协议要点：

- 背景标签 `0` 被忽略。
- 标签重映射为 `0..C-1`。
- 每波段归一化和 PCA 均 fit 在所有非背景像素上。
- patch extraction 使用 reflect padding。
- 主 split 为 stratified 10% / 10% / 80%，seed=42。

辅助数据：

- `result/stage1_data_protocol_synthetic/`：LCZ42 synthetic 小样本协议，60 samples，5 类。
- `result/stage1_data_protocol_synthetic_hsi/`：Synthetic HSI，550 samples，PCA=6，patch size=5。

---

## 2. 完整监督 baseline 与 tuned HybridSN

### 2.1 Stage 2 传统 baseline

输出：

```text
result/stage2_traditional_baselines/
```

| Dataset | Best traditional model | OA | AA | Kappa | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Indian Pines | SVM-RBF | 66.07 | 56.11 | 61.20 | 57.57 |
| Pavia University | SVM-RBF | 94.22 | 92.08 | 92.30 | 92.85 |
| Salinas | SVM-RBF | 92.08 | 95.61 | 91.16 | 95.79 |

传统 baseline 使用 PCA 后中心像素光谱向量，不使用 spatial patch。

### 2.2 Stage 3 深度 baseline

输出：

```text
result/stage3_deep_baselines/
```

| Dataset | Best deep model | OA | AA | Kappa | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Indian Pines | HybridSN | 81.00 | 68.69 | 78.27 | 69.97 |
| Pavia University | HybridSN | 99.09 | 98.44 | 98.79 | 98.34 |
| Salinas | HybridSN | 98.76 | 99.28 | 98.61 | 99.30 |

注意：Stage 3 初跑中 Indian Pines 的 1D-CNN / 3D-CNN 明显 undertrained。后续 debug 修复见：

```text
result/debug_cnn1d_cnn3d_indian_pines/
```

修复后 compact CPU run：

| Model | Before OA | After OA | Before Macro-F1 | After Macro-F1 |
|---|---:|---:|---:|---:|
| CNN1D | 13.98 | 59.28 | 2.03 | 46.33 |
| CNN3D | 27.00 | 63.68 | 4.48 | 49.00 |

### 2.3 Indian Pines tuned HybridSN

输出：

```text
result/hybridsn_tuning_indian_pines/
result/final_report/
```

最佳配置：

```yaml
architecture: hybridsn_base
patch_size: 11
pca_components: 30
learning_rate: 0.001
batch_size: 16
weight_decay: 0.0
dropout: 0.5
loss_name: weighted_ce
class_weight_mode: inverse_sqrt
embedding_dim: 128
seed: 0
```

Indian Pines best-only 汇总：

| Model group | Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|
| MLP Head | `mlp_h256_d0.0_lr0.003` | 84.05 | 77.19 | 81.73 | 78.55 | 83.77 |
| Residual QNN | Residual QNN full | 78.34 | 78.96 | 75.55 | 77.41 | 78.39 |
| Residual Reupload Multiobs QNN | Residual Reupload Multiobs QNN full | 78.73 | 74.25 | 75.32 | 76.13 | 77.08 |
| Bottleneck Head | `bottleneck_b32_relu_lr0.01` | 81.60 | 75.52 | 79.03 | 75.68 | 81.40 |
| Linear Head | `linear_lr0.01` | 80.05 | 70.63 | 76.94 | 73.33 | 79.43 |
| HybridSN initial | HybridSN | 81.00 | 68.69 | 78.27 | 69.97 | 80.48 |
| SVM-RBF tuned | `svm_C10_gammascale` | 68.40 | 56.13 | 63.48 | 58.36 | 67.07 |

tuned HybridSN single best test result：

| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| HybridSN tuned | 98.80 | 97.27 | 98.64 | 97.19 | 98.81 |

解释：

- 该结果证明 HybridSN 在 random pixel split 下非常强。
- 但 random pixel split 对 patch-based HSI 模型可能乐观，后续 spatial split 结果显示泛化难度显著更高。

---

## 3. 早期 QNN head 结果

### 3.1 Stage 4 QNN classifier pilot

输出：

```text
result/stage4_qnn_classifier/
```

这一阶段把 QNN 放在 frozen / extracted HybridSN embedding 后作为分类 head，CPU pilot 只在 Indian Pines 上实际跑 QNN。

| Dataset | Model | OA | AA | Macro-F1 | Note |
|---|---|---:|---:|---:|---|
| Indian Pines | HybridSN linear | 70.38 | 53.67 | 52.65 | full test |
| Indian Pines | HybridSN MLP | 73.38 | 56.29 | 57.15 | full test |
| Indian Pines | HybridSN bottleneck | 46.74 | 23.57 | 18.53 | full test |
| Indian Pines | HybridSN QNN | 20.44 | 17.25 | 7.08 | stratified subset |
| Pavia University | HybridSN MLP | 98.68 | 97.80 | 98.01 | QNN skipped |
| Salinas | HybridSN MLP | 97.57 | 98.85 | 98.87 | QNN skipped |

结论：直接把 QNN 作为最终分类头效果很差，不能作为主线。

### 3.2 QNN sweep / residual QNN

输出：

```text
result/stage4_qnn_sweep_indian_pines/
result/stage4_qnn_optimized_indian_pines/
result/stage4_residual_qnn_full_indian_pines/
result/stage4_reupload_multiobs_qnn_full_indian_pines/
```

关键结果：

- 普通 QNN head sweep 最高 OA 约 21.79%，整体失败。
- residual QNN head 在 subset pilot 中显著好于普通 QNN head，最高 OA 78.67% 左右。
- full-test residual QNN：

| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Residual QNN full | 78.34 | 78.96 | 75.55 | 77.41 | 78.39 |
| Residual Reupload Multiobs QNN full | 78.73 | 74.25 | 75.32 | 76.13 | 77.08 |

解释：

- residual connection 是 QNN head 能工作的关键。
- 但它仍未超过 strong tuned classical heads，尤其不超过 tuned MLP / tuned HybridSN。

---

## 4. Frozen HybridSN head 与参数效率实验

相关输出：

```text
result/frozen_hybridsn_head_comparison_indian_pines/
result/parameter_efficiency_hybridsn_heads_indian_pines/
result/low_label_hybridsn_head_regularization_indian_pines/
result/hybridsn_qnn_optimized_indian_pines/
```

### 4.1 Frozen head fair comparison

所有 head 使用同一个 frozen tuned HybridSN encoder embedding。

| Model | Runs | Val Macro-F1 mean | Best test OA | Best test Macro-F1 |
|---|---:|---:|---:|---:|
| gated residual QNN | 5 | 98.63 | - | - |
| MLP probe | 5 | 98.59 | 98.77 | 96.85 |
| residual QNN | 5 | 98.57 | - | - |
| linear probe | 5 | 97.93 | - | - |

参考 tuned full HybridSN：

| Model | OA | AA | Macro-F1 |
|---|---:|---:|---:|
| Tuned full HybridSN reference | 98.80 | 97.27 | 97.19 |

结论：

- QNN heads 在 validation 上可以接近 MLP/Linear。
- 但在 fair frozen embedding 比较中，没有充分证据说明 QNN head 显著优于 MLP head。
- 更适合表述为 compact quantum-augmented matching，而不是 superior accuracy。

### 4.2 参数效率

| Model | Params mean | Val Macro-F1 mean | Time mean |
|---|---:|---:|---:|
| MLP h64 | 9552 | 99.41 | 0.72s |
| residual QNN q4 l1 linear | 2928 | 98.98 | 49.84s |
| gated residual QNN q4 l1 linear | 3313 | 98.51 | 45.68s |
| bottleneck b16 | 2592 | 98.41 | 0.85s |
| linear probe | 2320 | 97.85 | 0.45s |

结论：

- QNN head 可以用较少参数接近 MLP，但训练速度慢很多。
- 参数效率可以作为辅助卖点，但不是准确率优势证据。

### 4.3 Low-label head regularization

输出：

```text
result/low_label_hybridsn_head_regularization_indian_pines/
```

只使用原训练集的一小部分训练 head：

| Train fraction | Best classical validation model | Best QNN validation signal |
|---:|---|---|
| 0.01 | linear / MLP 更好 | QNN 低于 classical |
| 0.03 | linear / MLP 更好 | residual QNN 接近但仍低 |
| 0.05 | MLP / linear 更好 | QNN 接近 |
| 0.10 | MLP / linear 更好 | QNN 接近 |

Best selected test：

| Model | Fraction | OA | AA | Kappa | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| MLP probe | 0.05 | 98.56 | 97.70 | 98.36 | 97.16 | 98.56 |

结论：在 frozen tuned encoder 上，QNN head 没有显示出比 classical head 更强的 low-label regularization。

---

## 5. Spectral QNN branch：QNN 放置位置改变

早期 head-only QNN 不够好，因此后续把 QNN 从最终分类头移到中心像素光谱分支。

输出：

```text
result/hybridsn_spectral_qnn_branch_indian_pines/
result/qnn_innovation_experiments_indian_pines/
```

结构：

```text
patch -> HybridSN encoder -> embedding
center PCA spectrum -> spectral QNN branch
embedding / spectral branch -> fusion classifier
```

Indian Pines spectral branch pilot：

| Model | Runs | Params mean | Val Macro-F1 mean | Best test OA | Best test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| embedding MLP | 3 | 9552 | 99.41 | 98.72 | 96.31 |
| spectral MLP fusion | 3 | 2772 | 97.94 | 98.82 | 96.67 |
| spectral QNN fusion | 3 | 2596 | 97.92 | 98.91 | 97.41 |
| spectral gated QNN fusion | 3 | 3071 | 97.81 | 98.82 | 96.50 |

结论：

- 把 QNN 放到 spectral branch 比 final head-only 更合理。
- spectral QNN branch 在该 pilot 中取得了最高 test Macro-F1，但 validation mean 仍低于 embedding MLP，因此不能单独宣称已稳定优于经典 head。
- 这条线后来发展为 few-shot Spectral QNN Gated Fusion 主线。

---

## 6. Few-shot baseline 总览

综合输出：

```text
result/all_fewshot_model_summary/
result/other_baselines_fewshot_summary/
result/hybridsn_small_fewshot_3datasets/
```

协议：

- all-way few-shot。
- shots: 1, 5, 10 per class。
- seeds: 0, 1, 2, 3, 4。
- PCA fit on full image without labels。
- HybridSN-small 是当前 few-shot 主 baseline。

### 6.1 Indian Pines

| Shot | Best / key model | OA | Macro-F1 | Notes |
|---:|---|---:|---:|---|
| 1 | Spectral QNN Gated Fusion | 44.40 ± 5.44 | 40.50 ± 3.96 | 高于 HybridSN-small |
| 1 | HybridSN-small | 41.49 ± 7.23 | 37.75 ± 3.00 | strong baseline |
| 5 | QNN + SupCon | 72.06 ± 1.42 | 64.04 ± 1.44 | OA 略高 |
| 5 | QNN + Prototype | 72.05 ± 1.50 | 64.12 ± 1.73 | Macro-F1 略高 |
| 5 | HybridSN-small | 72.02 ± 1.67 | 63.81 ± 2.17 | baseline |
| 10 | QNN + SupCon | 81.07 ± 2.32 | 72.26 ± 2.08 | 当前最佳 |
| 10 | QNN + Prototype | 80.88 ± 2.40 | 71.82 ± 1.92 | 正提升 |
| 10 | HybridSN-small | 80.12 ± 2.56 | 71.53 ± 2.02 | baseline |

Indian Pines 结论：

- 1-shot：spectral QNN gated fusion 有明确正提升。
- 5-shot：prototype / SupCon 都只是边际提升，不能夸大。
- 10-shot：SupCon 最好，OA +0.95，Macro-F1 +0.73。

### 6.2 Pavia University

| Shot | Model | OA | Macro-F1 | Notes |
|---:|---|---:|---:|---|
| 1 | HybridSN-small | 53.19 ± 8.13 | 48.84 ± 5.30 | QNN 未完成 1-shot 主结果 |
| 1 | SVM-RBF / kNN | 51.91 ± 4.57 | 55.01 ± 2.41 | Macro-F1 高于 HybridSN-small |
| 5 | QNN + Prototype | 77.53 ± 3.48 | 77.11 ± 2.17 | 明显提升 |
| 5 | HybridSN-small | 75.74 ± 2.89 | 71.21 ± 4.46 | baseline |
| 10 | QNN + Prototype | 86.32 ± 2.59 | 86.13 ± 2.22 | 明显提升 |
| 10 | HybridSN-small | 82.26 ± 4.92 | 79.20 ± 8.14 | baseline |

Pavia University 结论：

- QNN + Prototype 是当前最强证据。
- 5-shot：OA +1.78，Macro-F1 +5.90。
- 10-shot：OA +4.05，Macro-F1 +6.93。
- 10-shot 中方差也明显降低。

### 6.3 Salinas

| Shot | Model | OA | Macro-F1 | Notes |
|---:|---|---:|---:|---|
| 1 | CNN2D | 74.22 ± 1.95 | 74.91 ± 1.92 | 1-shot 当前最强 baseline |
| 1 | SVM-RBF / kNN | 69.96 ± 2.28 | 72.00 ± 2.12 | 传统模型很强 |
| 5 | CNN2D | 88.30 ± 3.17 | 91.00 ± 2.00 | OA 略高于 QNN |
| 5 | QNN + Prototype | 88.16 ± 2.09 | 93.36 ± 0.78 | Macro-F1 明显最佳 |
| 5 | HybridSN-small | 86.93 ± 7.11 | 89.67 ± 9.21 | 方差大 |
| 10 | HybridSN-small | 93.60 ± 1.59 | 95.44 ± 1.55 | 最强 |
| 10 | CNN2D | 93.57 ± 0.85 | 95.23 ± 0.60 | 接近 HybridSN-small |
| 10 | QNN + Prototype | 90.72 ± 3.77 | 94.96 ± 1.79 | 负迁移 |

Salinas 结论：

- 5-shot：QNN + Prototype 的 Macro-F1 明显提升，且方差显著降低。
- 10-shot：HybridSN-small / CNN2D 已接近饱和，QNN + Prototype 出现 OA -2.88 的负迁移。

---

## 7. Metric-learning QNN 主结果

核心输出：

```text
result/fewshot_metric_loss_cross_dataset_summary/
```

### 7.1 总表

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

### 7.2 相对 HybridSN-small 的变化

| Dataset | Shot | Model | Delta OA | Delta AA | Delta Macro-F1 | Delta Weighted-F1 |
|---|---:|---|---:|---:|---:|---:|
| Indian Pines | 5 | QNN + Prototype | +0.03 | -0.14 | +0.32 | -0.43 |
| Indian Pines | 5 | QNN + SupCon | +0.04 | -0.11 | +0.23 | -0.36 |
| Indian Pines | 10 | QNN + Prototype | +0.76 | +0.23 | +0.29 | +0.74 |
| Indian Pines | 10 | QNN + SupCon | +0.95 | +0.41 | +0.73 | +0.92 |
| Pavia University | 5 | QNN + Prototype | +1.78 | +4.05 | +5.90 | +2.48 |
| Pavia University | 10 | QNN + Prototype | +4.05 | +5.17 | +6.93 | +4.13 |
| Salinas | 5 | QNN + Prototype | +1.23 | +3.11 | +3.69 | +2.30 |
| Salinas | 10 | QNN + Prototype | -2.88 | -0.50 | -0.48 | -3.33 |

### 7.3 直接结论

- **最强正证据**：Pavia University 5/10-shot。
- **清晰正证据**：Salinas 5-shot，特别是 Macro-F1 和方差。
- **边际正证据**：Indian Pines 5/10-shot。
- **负例**：Salinas 10-shot。
- **loss 结论**：Indian Pines 上 Prototype 与 SupCon 的公平对照说明 metric-learning objective 是 QNN few-shot 表现的关键因素。

---

## 8. Spectral QNN gated fusion 调参与 prototype loss

### 8.1 无 metric loss 的 gated fusion 调参

输出：

```text
result/hybridsn_small_spectral_qnn_tuning_indian_pines/
```

Best by Macro-F1：

| Shot | Config | HybridSN-small OA | QNN OA | Delta OA | HybridSN-small Macro-F1 | QNN Macro-F1 | Delta Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | q4_l1_scalar | 41.49 | 44.40 | +2.91 | 37.75 | 40.50 | +2.75 |
| 5 | q6_l2_classwise_5shot | 72.02 | 71.68 | -0.34 | 63.81 | 64.40 | +0.60 |
| 10 | q6_l1_classwise | 80.12 | 80.89 | +0.77 | 71.53 | 71.81 | +0.28 |

解释：

- 不加 metric loss 时，QNN 已经能改善 1-shot 和 10-shot。
- 5-shot 只改善 Macro-F1，不稳定超过 OA。
- 加深 QNN、ring entanglement、zero weight decay、lower LR 等并没有稳定解决 5-shot。

### 8.2 CE + Prototype Loss

输出：

```text
result/hybridsn_small_spectral_qnn_gated_proto_tuning_indian_pines/
result/hybridsn_small_spectral_qnn_gated_proto_fewshot_indian_pines_pw02/
result/hybridsn_small_spectral_qnn_gated_proto_fewshot_indian_pines_pw02_10shot/
```

Indian Pines：

| Shot | Model | OA | Macro-F1 | Delta OA | Delta Macro-F1 |
|---:|---|---:|---:|---:|---:|
| 5 | QNN + Prototype | 72.05 ± 1.50 | 64.12 ± 1.73 | +0.03 | +0.32 |
| 10 | QNN + Prototype | 80.88 ± 2.40 | 71.82 ± 1.92 | +0.76 | +0.29 |

结论：

- Prototype loss 是第一个让 Indian Pines 5/10-shot 同时在 OA 与 Macro-F1 上超过 HybridSN-small 的设置。
- 5-shot OA 提升非常小，应表述为 marginal improvement。
- 更合理的解释是 metric objective 稳定类别边界，而不是 QNN depth 不够。

---

## 9. 边界几何与 logit margin 分析

相关输出：

```text
result/boundary_geometry_hybridsn_vs_qnn/
result/logit_margin_hybridsn_vs_qnn/
```

### 9.1 Prototype geometry

分析对象：

- HybridSN-small
- Spectral QNN Gated Fusion + Prototype

主要结论：

- prototype separation ratio 并没有普遍改善。
- QNN 往往增加 intra-class distance，因此单看 prototype geometry 不能证明 QNN 普遍改善特征空间。
- Pavia 5-shot / 10-shot 的 negative prototype margin rate 有改善，但整体证据仍需谨慎。

### 9.2 Classifier logit margin

logit margin 更贴近最终分类边界。

同时满足 mean probability margin 提升且 negative logit margin rate 降低的设置：

| Dataset | Shot | Interpretation |
|---|---:|---|
| Indian Pines | 5 | 分类性能和 probability margin 同步改善 |
| Indian Pines | 10 | 分类性能和 probability margin 同步改善 |
| Pavia University | 10 | 最强 logit-boundary 证据之一 |
| Salinas | 5 | logit margin 与 Macro-F1 同步改善 |

联合表中的关键数值：

| Dataset | Shot | Model | Macro-F1 | Mean true logit margin | Negative logit margin rate |
|---|---:|---|---:|---:|---:|
| Indian Pines | 5 | HybridSN-small | 0.6381 | 0.8646 | 0.2798 |
| Indian Pines | 5 | QNN + Prototype | 0.6412 | 0.9575 | 0.2795 |
| Indian Pines | 10 | HybridSN-small | 0.7153 | 1.8801 | 0.1988 |
| Indian Pines | 10 | QNN + Prototype | 0.7182 | 1.8026 | 0.1912 |
| Pavia University | 5 | HybridSN-small | 0.7620 | 1.4172 | 0.2342 |
| Pavia University | 5 | QNN + Prototype | 0.7711 | 1.1281 | 0.2247 |
| Pavia University | 10 | HybridSN-small | 0.7920 | 1.5126 | 0.1774 |
| Pavia University | 10 | QNN + Prototype | 0.8613 | 1.6014 | 0.1368 |
| Salinas | 5 | HybridSN-small | 0.8967 | 1.6450 | 0.1307 |
| Salinas | 5 | QNN + Prototype | 0.9336 | 1.8843 | 0.1184 |
| Salinas | 10 | HybridSN-small | 0.9544 | 2.3213 | 0.0640 |
| Salinas | 10 | QNN + Prototype | 0.9496 | 2.2359 | 0.0928 |

解释：

- logit margin 支持“QNN 改善最终 classifier decision boundary”的设置主要是 Indian Pines 5/10、Pavia 10、Salinas 5。
- Pavia 5-shot 虽然分类性能提升，但平均 probability margin 未同步提升，可能是少数类别或特定样本贡献。
- Salinas 10-shot 的 logit margin 与性能都支持负迁移判断。

---

## 10. Spatial split 诊断

输出：

```text
result/spatial_split_hybridsn_indian_pines/
result/spatial_split_qnn_heads_indian_pines/
```

Spatial split 使用 grid-block 空间互斥划分，train/validation/test block 不重叠。

### 10.1 HybridSN spatial split

| Protocol | OA | AA | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Random pixel split tuned HybridSN | 98.80 | 97.27 | 97.19 | 98.81 |
| Spatial split HybridSN pilot | 33.16 | 19.36 | 13.03 | 24.02 |

### 10.2 Spatial split QNN head pilot

| Model | Val Macro-F1 | Val OA | Test OA | Test Macro-F1 |
|---|---:|---:|---:|---:|
| linear probe | 11.00 | 49.66 | - | - |
| MLP h64 | 14.16 | 56.58 | 35.90 | 13.78 |
| residual QNN q4 l1 linear | 12.18 | 52.30 | - | - |
| gated residual QNN q4 l1 linear | 11.25 | 50.77 | - | - |

结论：

- random pixel split 对 patch-based HSI 分类过于乐观。
- spatial split 下模型整体性能急剧下降。
- QNN head 没有解决空间泛化问题，也没有超过 MLP head。
- 这部分适合作为论文中的 limitation / robustness discussion。

---

## 11. 当前最适合写进论文的结果线

### 主线一：Few-shot metric-learning QNN

最适合放主表：

```text
result/fewshot_metric_loss_cross_dataset_summary/report_zh.md
```

建议论文主结论：

```text
Spectral QNN Gated Fusion + metric-learning objective 在多数 low-shot 设置中对 HybridSN-small 有正向作用，
但这种作用依赖 dataset 和 shot，不是普适优势。
```

重点强调：

- Pavia 5/10-shot 是最强提升。
- Salinas 5-shot 提升 Macro-F1 并降低方差。
- Indian Pines 提升边际但可复现。
- Salinas 10-shot 是负例，证明方法不是盲目增益。

### 主线二：为什么不是 final QNN head

可引用：

```text
result/stage4_qnn_classifier/
result/stage4_qnn_sweep_indian_pines/
result/stage4_residual_qnn_full_indian_pines/
result/frozen_hybridsn_head_comparison_indian_pines/
```

结论：

```text
Final-head QNN 单独放在 HybridSN embedding 后不是有效方向；
QNN 需要接触高光谱任务中的 spectral structure，放在 center spectral branch 更合理。
```

### 主线三：边界解释

可引用：

```text
result/logit_margin_hybridsn_vs_qnn/
```

结论：

```text
QNN 提升更体现在 final classifier logit margin，而不是普遍改善 prototype feature geometry。
```

---

## 12. 不建议夸大的结果

1. **不要说 QNN 全面超过 HybridSN-small。**
   Salinas 10-shot 是明确负例。

2. **不要用 random pixel split tuned HybridSN 的 98%+ 结果直接证明实际空间泛化。**
   spatial split pilot 显示性能大幅下降。

3. **不要说 QNN head 比 MLP head 更强。**
   frozen head fair comparison 中 MLP / linear 仍很强，QNN 只能说参数更少时接近。

4. **不要把 prototype geometry 说成普遍改善证据。**
   geometry 指标并不一致；logit margin 是更合适的解释线。

5. **不要把 Indian Pines 5-shot 的 QNN 提升写成显著优势。**
   OA 只有约 +0.03 到 +0.04，Macro-F1 约 +0.23 到 +0.32，是边际但可复现。

---

## 13. 结果目录索引

关键结果目录：

```text
result/stage1_data_protocol/
result/stage2_traditional_baselines/
result/stage3_deep_baselines/
result/stage4_qnn_classifier/
result/stage4_qnn_sweep_indian_pines/
result/stage4_qnn_optimized_indian_pines/
result/stage4_residual_qnn_full_indian_pines/
result/stage4_reupload_multiobs_qnn_full_indian_pines/
result/hybridsn_tuning_indian_pines/
result/final_report/
result/frozen_hybridsn_head_comparison_indian_pines/
result/parameter_efficiency_hybridsn_heads_indian_pines/
result/low_label_hybridsn_head_regularization_indian_pines/
result/hybridsn_spectral_qnn_branch_indian_pines/
result/qnn_innovation_experiments_indian_pines/
result/hybridsn_small_fewshot_3datasets/
result/all_fewshot_model_summary/
result/other_baselines_fewshot_summary/
result/hybridsn_small_spectral_qnn_tuning_indian_pines/
result/hybridsn_small_spectral_qnn_gated_proto_tuning_indian_pines/
result/fewshot_metric_loss_cross_dataset_summary/
result/boundary_geometry_hybridsn_vs_qnn/
result/logit_margin_hybridsn_vs_qnn/
result/spatial_split_hybridsn_indian_pines/
result/spatial_split_qnn_heads_indian_pines/
```

新增/未跟踪但已存在的近期分析目录：

```text
result/boundary_geometry_hybridsn_vs_qnn/
result/logit_margin_hybridsn_vs_qnn/
scripts/analyze_hybridsn_vs_qnn_boundary_geometry.py
scripts/analyze_hybridsn_vs_qnn_logit_margin.py
scripts/run_fair_control_models_fewshot.py
```

---

## 14. 下一步建议

1. 把 `fewshot_metric_loss_cross_dataset_summary` 作为主结果表整理进论文。
2. 为 Pavia University 和 Salinas 补齐 SupCon 对照，确认 Prototype vs SupCon 的差异是否跨数据集稳定。
3. 对 Salinas 10-shot 做负迁移分析：检查 QNN gate、类别级 F1、logit margin 与 support 分布。
4. 选择 1-2 个关键设置做 statistical significance / paired seed 检验。
5. 如果论文需要更强泛化论证，应扩展 spatial split，而不是只依赖 random pixel split。
