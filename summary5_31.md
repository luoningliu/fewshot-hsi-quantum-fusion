# 2026-05-31 项目阶段总结

## 0. 时间范围与总体判断

本总结覆盖 2026-05-25 至 2026-05-31 期间的主要工作。过去几天的核心目标已经从“继续证明 QNN 有优势”转为更严格的问题：

```text
在 Indian Pines / Pavia University / Salinas 的 few-shot HSI 分类中，
Spectral QNN Gated Fusion 是否能在统一设置下稳定超过 HybridSN-small？
```

截至 5/31，结论需要保持克制：

```text
Spectral QNN Gated Fusion + metric-learning 在 Pavia University 和部分 low-shot 设置中有稳定正增益；
Indian Pines 5/10-shot 有边际正增益；
Salinas 5-shot 有明显提升；
但 Salinas 10-shot 仍是反复出现的负迁移反例，当前不能宣称 QNN 全面超过 HybridSN-small。
```

因此，当前论文主线更适合写成：

```text
量子 spectral branch 不是通用替换分类头；
它更像少样本场景下的 spectral-side decision-boundary regularizer，
其收益依赖数据集结构、shot 数和 metric-learning objective。
```

---

## 1. 已完成的主实验成果

### 1.1 few-shot 主协议已经成型

当前主协议固定为：

| 项目 | 设置 |
|---|---|
| 数据集 | Indian Pines, Pavia University, Salinas |
| shots | 5-shot, 10-shot |
| seeds | 0, 1, 2, 3, 4 |
| baseline | HybridSN-small |
| 主指标 | OA, Macro-F1, Weighted-F1 |
| 通过规则 | mean OA 和 mean Macro-F1 高于 HybridSN-small，paired seed delta 至少 3/5 为正 |

相关汇总文件：

```text
result/fewshot_metric_loss_cross_dataset_summary/report_zh.md
result/all_fewshot_model_summary/report.md
summary5_25.md
summary5_29.md
```

### 1.2 QNN + metric-learning 的正结果已经比较清楚

当前保留下来的主正结果包括：

| Dataset | Shot | QNN 结果相对 HybridSN-small 的变化 |
|---|---:|---|
| Indian Pines | 5 | OA 约 +0.04，Macro-F1 约 +0.23，属于边际提升 |
| Indian Pines | 10 | OA 约 +0.95，Macro-F1 约 +0.73，SupCon 优于 Prototype |
| Pavia University | 5 | OA 约 +1.78，Macro-F1 约 +5.90 |
| Pavia University | 10 | OA 约 +4.05，Macro-F1 约 +6.93 |
| Salinas | 5 | OA 约 +1.23，Macro-F1 约 +3.69 |
| Salinas | 10 | Prototype QNN 低于 HybridSN-small，OA 约 -2.88，Macro-F1 约 -0.48 |

这说明：

```text
QNN 的有效形式不是早期的直接 QNN classifier head，
而是 center-pixel spectral branch + gated fusion + metric-learning objective。
```

---

## 2. Salinas 10-shot 负迁移诊断

过去几天最重要的工作，是围绕 Salinas 10-shot 这个失败点做了系统排查。结论是：Salinas 10-shot 的问题不是单个超参数偶然没调好，而是当前 QNN residual branch 会在 classical baseline 已经很强时扰动决策边界。

### 2.1 Data Re-uploading QNN

实验：

```text
result/qnn_reupload_supcon_minibatch_salinas_pavia_10shot_20260526_103205/
```

核心结果：

| Dataset | Shot | Delta OA | Delta Macro-F1 | Delta Weighted-F1 |
|---|---:|---:|---:|---:|
| Salinas | 10 | -0.0226 | +0.0025 | -0.0229 |
| Pavia University | 10 | +0.0474 | +0.0799 | +0.0490 |

判断：

```text
Data re-uploading 保留了 Pavia 正增益，但没有解决 Salinas 10-shot 的 OA / Weighted-F1 负迁移。
```

### 2.2 ResidualSafe-A / ResidualSafe-B

实验：

```text
result/qnn_residualsafe_supcon_minibatch_salinas_pavia_10shot_20260527_001133/
result/qnn_residualsafe_b_supcon_minibatch_salinas_pavia_10shot_20260527_102113/
```

ResidualSafe-A 使用 learnable residual scale，`alpha_init=-4.0`；ResidualSafe-B 放宽为 `alpha_init=-2.0` 并加入 warmup。

关键结果：

| Variant | Salinas 10-shot Delta OA | Salinas Delta Macro-F1 | Pavia 10-shot Delta OA | 判断 |
|---|---:|---:|---:|---|
| ResidualSafe-A | -0.0128 | +0.0016 | +0.0221 | 缓解 Salinas，但牺牲 Pavia 增益 |
| ResidualSafe-B | -0.0218 | -0.0032 | +0.0217 | 比 A 更不稳，停止该路线 |

判断：

```text
全局 residual scale 不能可靠解决负迁移；过强会扰动 Salinas，过弱又损失 Pavia 上 QNN 的有效贡献。
```

### 2.3 MultiProto-2

实验：

```text
result/qnn_multiproto2_minibatch_salinas_pavia_10shot_20260527_113147/
```

关键结果：

| Dataset | Shot | Delta OA | Delta Macro-F1 | Delta Weighted-F1 |
|---|---:|---:|---:|---:|
| Pavia University | 10 | +0.0405 | +0.0692 | +0.0412 |
| Salinas | 10 | -0.0294 | -0.0051 | -0.0340 |

判断：

```text
固定 deterministic sub-prototype 能继续支撑 Pavia 正结果，
但没有解决 Salinas 类内复杂性，反而加重 Salinas 10-shot 负迁移。
```

### 2.4 ConfidenceGuard 系列

这是当前最有希望的缓解方向。

主要实验：

```text
result/qnn_confguard_supcon_minibatch_salinas_pavia_10shot_20260527_152324/
result/qnn_confguardb_supcon_phase1_salinas_pavia_10shot/
result/qnn_confguardc_penalty01_supcon_phase1_salinas_pavia_10shot/
result/qnn_confguardd_penalty02_supcon_phase1_salinas_pavia_10shot/
result/qnn_prelogit_confguard_supcon_phase1_salinas_pavia_10shot/
```

汇总结果：

| Variant | Pavia 10-shot Delta OA | Pavia Delta Macro-F1 | Salinas 10-shot Delta OA | Salinas Delta Macro-F1 | 判断 |
|---|---:|---:|---:|---:|---|
| ConfidenceGuard-A | +3.43% | +6.67% | -1.10% | +0.66% | 当前最强缓解之一，但未过 Salinas OA |
| ConfidenceGuard-B | +2.92% | +5.66% | -1.62% | +0.25% | margin suppression 方向失败 |
| ConfidenceGuard-C | +3.51% | +6.73% | -1.00% | +0.68% | 当前 phase-1 最好 partial variant |
| ConfidenceGuard-D | +3.59% | +6.71% | -1.13% | +0.60% | penalty 继续加大没有收益 |
| Prelogit ConfidenceGuard | +2.79% | +4.43% | -3.17% | -0.57% | 否定“base_head 重训导致失败”的假设 |

关键诊断：

```text
ConfidenceGuard-C 是目前最好的 partial variant：
Pavia 10-shot 通过 paired rule；
Salinas 10-shot Macro-F1 为正，但 OA 仍为负，positive OA seeds 只有 2/5。
```

---

## 3. 代码与工具链新增

这几天的主要代码改动集中在 three scripts：

```text
scripts/run_hybridsn_small_spectral_qnn_gated_metric_fewshot.py
scripts/run_supcon_cross_dataset_fewshot.py
scripts/run_fair_control_models_fewshot.py
```

新增能力包括：

1. `gate_context_mode=base_confidence_margin`：把 base softmax confidence 和 top1-top2 margin 纳入 gate 输入。
2. `gate_confidence_penalty`：对高置信 base 样本上的 gate 加惩罚。
3. `high_confidence_guard_mode=margin_suppression`：基于 base margin 对 QNN residual 做乘性 suppression。
4. `base_logit_mode=pretrained`：直接使用原始 HybridSN-small classifier logits，验证负迁移是否来自重新训练 base head。
5. 自动输出 `comparison_vs_hybridsn_small.csv` 和 `paired_seed_delta_vs_hybridsn_small.csv`，用于统一判定是否超过 HybridSN-small。
6. gate diagnostics 扩展：保存 `guard`、`base_margin_norm`、`base_confidence` 等诊断量。

这些改动的价值在于：

```text
后续每个 QNN variant 都可以直接和 HybridSN-small 做 paired seed 对比，
避免只看 mean table 导致误判。
```

---

## 4. 当前可写进论文的结果表述

可以写：

```text
在少样本 HSI 分类中，将 QNN 作为最终分类头并不有效；
但将 QNN 放在中心像素 spectral branch，并通过 gated fusion 与 metric-learning objective 结合，
可以在 Pavia University 5/10-shot、Salinas 5-shot 和 Indian Pines 10-shot 上带来稳定或边际增益。
```

可以写：

```text
Salinas 10-shot 揭示了 hybrid quantum-classical residual branch 的负迁移风险：
当 HybridSN-small 已经形成高置信、较强的决策边界时，
QNN residual 可能改善 Macro-F1 的少数类边界，却降低 OA 和 Weighted-F1。
```

不应写：

```text
QNN 在所有 few-shot HSI 设置下全面超过 HybridSN-small。
```

更合适的创新点：

```text
1. 证明 QNN head 直接替换 classical head 不成立。
2. 提出 spectral-side gated QNN branch + metric-learning 的有效组合。
3. 用 Salinas 10-shot 系统揭示 QNN residual 的负迁移条件。
4. 引入 confidence-aware gate diagnostics，为后续 conditional residual / validation-calibrated QNN 提供依据。
```

---

## 5. 当前失败点与下一步

当前最大失败点仍然是：

```text
Salinas 10-shot：OA 和 Weighted-F1 仍低于 HybridSN-small。
```

已经不建议继续投入的方向：

1. 单纯增加 QNN circuit complexity，例如直接扩展 data re-uploading。
2. 全局 residual scale / warmup。
3. deterministic MultiProto split。
4. 继续增大 `gate_confidence_penalty`。
5. 继续调 `margin_suppression` 的 tau / temperature。
6. 继续使用 pretrained base logits 作为保护机制。

更值得继续的方向：

```text
Validation-calibrated / class-conditional QNN residual selection。
```

具体可以做：

1. 按 class 或 confusion pair 估计 QNN 是否在 validation set 上优于 HybridSN-small。
2. 只在 validation delta 为正的 class / margin bin 上启用 QNN residual。
3. 对 Salinas 中反复受损的类关闭或减弱 QNN residual。
4. 把 ConfidenceGuard-C 作为当前 partial baseline，而不是最终模型。

---

## 6. 最终状态

截至 5/31，项目已经形成了比较完整的证据链：

```text
早期 QNN classifier head 失败
-> residual / gated QNN head 部分可行
-> few-shot spectral QNN + metric learning 成为主线
-> Pavia / Indian Pines / Salinas 5-shot 有正结果
-> Salinas 10-shot 暴露负迁移
-> residual-safe、reuploading、multiproto、confidence guard、prelogit residual 被逐一验证
-> 当前最稳妥方向转向 conditional / validation-calibrated residual selection
```

当前最重要的论文判断是：

```text
本项目的价值不在于证明 QNN 无条件优于 HybridSN，
而在于明确了 QNN 在少样本高光谱分类中“何时有效、如何有效、何时会失败”。
```
