# 2026-05-29 项目阶段总结

## 0. 总体判断

截至 2026-05-29，项目目标已经从“超过传统 baseline”进一步收紧为：

```text
在 Indian Pines / Pavia University / Salinas 的 5-shot 和 10-shot few-shot 协议下，
提出一个统一配置的量子混合神经网络，并在 OA 和 Macro-F1 上超过 HybridSN-small。
```

当前结论：

```text
Pavia University 10-shot 可以稳定超过 HybridSN-small。
Indian Pines 10-shot 既有 SupCon QNN 已经超过 HybridSN-small。
Salinas 10-shot 仍是核心失败点：Macro-F1 可以略高于 HybridSN-small，但 OA 仍低于 HybridSN-small。
```

因此，当前仍不能写成：

```text
QNN 全面超过 HybridSN-small。
```

更准确的表述是：

```text
Spectral QNN Gated Fusion + metric-learning 在 Pavia University 和部分 low-shot 设置中有稳定增益，
但在 Salinas 10-shot 的高基线、高置信场景中仍存在负迁移。
```

---

## 1. 最终论文验收标准

本轮讨论后，验收标准明确为：

| 项目 | 决定 |
|---|---|
| 主 baseline | HybridSN-small few-shot |
| 主指标 | OA + Macro-F1 |
| 评估矩阵 | Indian Pines / Pavia University / Salinas × 5-shot / 10-shot |
| seeds | 0, 1, 2, 3, 4 |
| 通过规则 | mean OA 和 mean Macro-F1 均高于 HybridSN-small，且 paired seed delta 至少 3/5 为正 |
| 模型口径 | 一个统一配置，不按 dataset 或 shot 事后挑模型 |

这个标准比“超过 SVM / RF / kNN”严格得多，也更符合论文中无法绕开 HybridSN 的要求。

---

## 2. 5/29 新增实现

### 2.1 ConfidenceGuard-B

新增 high-confidence suppression：

```text
margin_norm = tanh((top1_logit - top2_logit) / 5.0)
guard = floor + (1 - floor) * sigmoid((tau - margin_norm) / temperature)
logits = base_logits + gate * guard * spectral_logits
```

固定参数：

```text
floor = 0.05
tau = 0.35
temperature = 0.08
gate_confidence_penalty = 0.1
```

相关代码：

```text
scripts/run_hybridsn_small_spectral_qnn_gated_metric_fewshot.py
scripts/run_supcon_cross_dataset_fewshot.py
scripts/run_fair_control_models_fewshot.py
```

新增输出：

```text
comparison_vs_hybridsn_small.csv
paired_seed_delta_vs_hybridsn_small.csv
gate_values.csv 中新增 guard / base_margin_norm
```

新增运行脚本：

```text
scripts/run_qnn_confguardb_supcon_phase1.sh
scripts/run_qnn_confguardb_supcon_full.sh
```

### 2.2 ConfidenceGuard-C / D

在 B 失败后，继续测试不使用 margin suppression、仅增强 gate confidence penalty：

```text
ConfidenceGuard-C: gate_confidence_penalty = 0.1
ConfidenceGuard-D: gate_confidence_penalty = 0.2
```

### 2.3 Prelogit QNN residual

新增 `base_logit_mode=pretrained`：

```text
base_logits = 原始 HybridSN-small checkpoint 的分类器输出
final_logits = base_logits + gate * spectral_logits
```

该设计用于验证：

```text
Salinas 10-shot 负迁移是否来自“重新训练 base_head 替代 HybridSN-small 分类器”。
```

实验结果显示该假设不成立，详见第 4 节。

---

## 3. Phase 1 实验结果

Phase 1 固定为：

```text
datasets = Salinas, Pavia University
shot = 10
seeds = 0, 1, 2, 3, 4
baseline = HybridSN-small
```

### 3.1 汇总表

| Variant | Pavia OA delta | Pavia Macro-F1 delta | Pavia pass | Salinas OA delta | Salinas Macro-F1 delta | Salinas pass |
|---|---:|---:|---|---:|---:|---|
| ConfidenceGuard-A, penalty=0.05 | +3.43% | +6.67% | 未按 paired rule 输出 | -1.10% | +0.66% | 否 |
| ConfidenceGuard-B, penalty=0.1 + margin suppression | +2.92% | +5.66% | 是 | -1.62% | +0.25% | 否 |
| ConfidenceGuard-C, penalty=0.1 | +3.51% | +6.73% | 是 | -1.00% | +0.68% | 否 |
| ConfidenceGuard-D, penalty=0.2 | +3.59% | +6.71% | 是 | -1.13% | +0.60% | 否 |
| Prelogit ConfidenceGuard | +2.79% | +4.43% | 是 | -3.17% | -0.57% | 否 |

结论：

```text
Pavia 10-shot 对 QNN spectral branch 很友好，多个 guard 变体都稳定通过。
Salinas 10-shot 的 OA 负迁移没有被 gate penalty、margin suppression 或 prelogit residual 解决。
```

### 3.2 Salinas 10-shot paired seed 结果

#### ConfidenceGuard-C

当前 C 是 Salinas 上相对最好的 guard 变体。

| Seed | Delta OA | Delta Macro-F1 | Delta Weighted-F1 |
|---:|---:|---:|---:|
| 0 | -5.33% | +0.03% | -5.37% |
| 1 | -1.04% | -0.77% | -1.01% |
| 2 | -1.55% | -0.00% | -1.56% |
| 3 | +0.95% | +0.95% | +0.97% |
| 4 | +1.95% | +3.20% | +2.09% |

解释：

```text
seed3 / seed4 已经超过 HybridSN-small；
seed0 / seed1 / seed2 仍低于 HybridSN-small；
其中 seed0 是最大失败源。
```

#### Prelogit ConfidenceGuard

| Seed | Delta OA | Delta Macro-F1 | Delta Weighted-F1 |
|---:|---:|---:|---:|
| 0 | -8.28% | -2.33% | -8.64% |
| 1 | -1.40% | -0.29% | -1.36% |
| 2 | -3.09% | -0.94% | -3.05% |
| 3 | -1.90% | -0.83% | -2.08% |
| 4 | -1.18% | +1.53% | -1.01% |

解释：

```text
直接使用原始 HybridSN-small logits 并不能解决负迁移，反而显著恶化 Salinas 10-shot。
因此，失败不只是由重新训练 base_head 造成的；QNN residual 本身会破坏 Salinas 的 OA。
```

---

## 4. 关键诊断

### 4.1 margin suppression 方向失败

ConfidenceGuard-B 的设计初衷是：

```text
当 HybridSN-small base logits 高置信时，压低 QNN residual；
当 base logits 低 margin 时，允许 QNN 介入。
```

但 Salinas 10-shot 的 gate diagnostics 显示：

```text
错误样本上的 guard 反而更大。
```

以 Salinas seed0 为例：

| 样本类型 | 平均 guard |
|---|---:|
| correct | 0.586 |
| wrong | 0.982 |

解释：

```text
该机制实际放大了低 margin / ambiguous 样本上的 QNN residual，
而这些样本正是 Salinas 10-shot 中最容易被 QNN 拉错的部分。
```

所以 B 不仅没有解决 Salinas，反而比 A/C 更差。

### 4.2 单纯增强 gate penalty 不够

ConfidenceGuard-C 和 D 说明：

```text
gate_confidence_penalty 从 0.05 提到 0.1 后，Salinas OA 略有改善；
继续提高到 0.2 没有继续改善。
```

因此，继续调大 gate penalty 不应作为下一步主线。

### 4.3 prelogit residual 假设被否定

Prelogit 实验原假设：

```text
如果 Salinas 失败来自新训练 base_head 不如原始 HybridSN-small classifier，
那么直接使用原始 HybridSN-small logits 应该能保护 OA。
```

实验结果相反：

```text
Salinas 10-shot OA delta = -3.17%
positive OA seeds = 0/5
```

所以主要问题不是 base_head，而是：

```text
当前 QNN residual 在 Salinas 10-shot 上系统性改变了本来已经较好的决策边界。
```

---

## 5. 当前最好模型候选

如果只看 Phase 1 的 Pavia + Salinas 10-shot，当前最好候选是：

```text
ConfidenceGuard-C
gate_context_mode = base_confidence_margin
gate_confidence_penalty = 0.1
high_confidence_guard_mode = none
loss = CE + SupCon
```

优点：

```text
Pavia 10-shot: OA +3.51%, Macro-F1 +6.73%, 5/5 seeds positive
Salinas 10-shot: Macro-F1 +0.68%, 3/5 seeds positive
```

缺点：

```text
Salinas 10-shot OA 仍为 -1.00%，positive OA seeds 只有 2/5。
```

因此 C 可以作为目前最好的 partial variant，但不能作为“全面超过 HybridSN-small”的最终模型。

---

## 6. 对论文主线的影响

### 6.1 不能采用的强结论

不能写：

```text
所提出 QNN 在三个数据集的 5/10-shot 设置下全面超过 HybridSN-small。
```

原因：

```text
Salinas 10-shot 是重复出现的明确反例。
多个 guard / residual / prelogit 变体均未解决。
```

### 6.2 可以采用的稳妥结论

可以写：

```text
Spectral QNN branch 在 Pavia University 和部分低样本设置中带来稳定增益；
在 Salinas 10-shot 这类 classical baseline 已较强的场景中，QNN residual 可能产生负迁移。
```

更适合的创新点表述：

```text
量子 spectral branch 的有效性依赖数据集结构和样本难度；
其优势不来自简单替换分类头，而来自与 spectral-side gated fusion 和 metric-learning objective 的组合。
```

---

## 7. 下一步建议

不要继续做：

```text
1. 继续增大 gate_confidence_penalty。
2. 继续调 margin_suppression 的 tau / temperature。
3. 继续使用 prelogit residual。
```

这些方向已经被 Phase 1 结果否定或弱化。

更值得做的下一步：

```text
选择性启用 QNN residual：
默认使用 HybridSN-small；
只在验证集证明 QNN 对某些 class、某些 margin 区间或某些 confusion pair 有稳定收益时启用 QNN。
```

候选方向：

1. Class-conditional residual mask：
   ```text
   对 Salinas seed0/1/2 中被 QNN 拉低的类关闭 residual；
   对 seed3/4 中有收益的类保留 residual。
   ```

2. Validation-calibrated residual selection：
   ```text
   在 validation set 上按 class 或 margin bin 估计 QNN 是否优于 HybridSN；
   只在 validation delta > 0 的区域使用 QNN。
   ```

3. Mixture-of-experts fallback：
   ```text
   final_logits = HybridSN logits
   if validation-calibrated condition is satisfied:
       final_logits += small QNN residual
   ```

该方向比继续调单一 gate 更符合当前诊断：Salinas 10-shot 不是所有样本都不适合 QNN，而是 QNN residual 对部分大类 / ambiguous 区域破坏过大。

---

## 8. 结果目录索引

本轮关键输出：

```text
result/qnn_confguardb_supcon_phase1_salinas_pavia_10shot/
result/qnn_confguardc_penalty01_supcon_phase1_salinas_pavia_10shot/
result/qnn_confguardd_penalty02_supcon_phase1_salinas_pavia_10shot/
result/qnn_prelogit_confguard_supcon_phase1_salinas_pavia_10shot/
```

重要文件：

```text
comparison_vs_hybridsn_small.csv
paired_seed_delta_vs_hybridsn_small.csv
summary_by_dataset_shot_metric_qnn.csv
metrics/*_gate_values.csv
```

这些文件构成 5/29 阶段判断的主要证据。
