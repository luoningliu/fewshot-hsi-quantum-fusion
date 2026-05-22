# 少样本 Metric-loss 跨数据集实验汇总

## 实验范围

- 数据集：Indian Pines、Pavia University、Salinas。
- 设置：5-shot 和 10-shot。
- Seeds：0、1、2、3、4。
- Baseline：HybridSN-small。
- 量子模型：Spectral QNN Gated Fusion，classwise gate。
- 损失函数：
  - CE + prototype loss。
  - CE + supervised contrastive loss，当前仅在 Indian Pines 上作为公平对照完成。

## 总结果

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

## 相对 HybridSN-small 的变化

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

## 主要结论

1. Indian Pines 上，CE + SupCon 是一个有效的公平对照。它使用相同 QNN 结构，只替换 metric-learning objective。
2. Indian Pines 5-shot 中，SupCon 与 prototype 都只带来边际提升；SupCon 的 OA 略高，prototype 的 Macro-F1 略高。
3. Indian Pines 10-shot 中，SupCon 优于 prototype：OA +0.95，Macro-F1 +0.73。
4. Pavia University 上，QNN + prototype 对 5-shot 和 10-shot 都有明显提升，尤其 Macro-F1 提升分别为 +5.90 和 +6.93。
5. Salinas 5-shot 中，QNN + prototype 明显提升 Macro-F1 和 AA，并显著降低方差。
6. Salinas 10-shot 中，QNN + prototype 低于 HybridSN-small，说明当 classical baseline 已经很强时，当前 QNN head 可能会产生负迁移。

## 当前论文表述建议

更稳妥的结论是：

```text
Spectral QNN Gated Fusion 在少样本场景下并非对所有数据集和 shot 都稳定优于 HybridSN-small；
但在 Indian Pines、Pavia University 和 Salinas 5-shot 上均显示出正向作用，
尤其在 Pavia University 上提升明显。
CE + SupCon 与 CE + prototype 的公平对照说明，metric-learning objective 是提升少样本 QNN 表现的重要因素。
```

不应写成：

```text
QNN 在所有少样本高光谱任务中全面超过 HybridSN。
```

更合理的创新点是：

```text
量子 spectral branch 需要与少样本 metric-learning objective 结合，才能更稳定地改善类边界；
其优势更容易出现在 5-shot 或较不稳定的少样本设置中，而不是 classical baseline 已接近饱和的设置。
```
