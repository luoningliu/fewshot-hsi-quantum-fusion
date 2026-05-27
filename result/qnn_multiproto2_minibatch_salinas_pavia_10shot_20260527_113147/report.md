# 公平对照 few-shot HSI 实验报告

## 1. 实验目的

本实验固定 HybridSN-small encoder checkpoint、few-shot split、seed 和测试流程，用于拆分 QNN 增益来源：

`HybridSN-small -> Frozen Linear -> Spectral MLP -> Spectral QNN`

`frozen_linear_proto` 是额外控制项：它仅在 frozen feature `z` 上加入 prototype loss，用于判断 metric loss 本身的影响。

## 2. 四个模型说明

| 模型                                  | 构造                                                                              | 作用                               |
|:--------------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------|
| HybridSN-small                        | 原始 few-shot HybridSN-small 端到端 baseline                                      | 给出经典 spectral-spatial baseline |
| Frozen HybridSN + Linear head         | 冻结 conv3d/conv2d pooled feature z，仅训练 LayerNorm + Linear                    | 排除重新训练分类 head 的影响       |
| Spectral MLP Gated Fusion + Prototype | z 与中心 PCA spectrum 的 MLP branch 做 gated residual fusion，并加 prototype loss | 经典 center spectral branch 对照   |
| Spectral QNN Gated Fusion + Prototype | 同一 gated fusion/prototype 设置下将 MLP branch 换成 QNN branch                   | 检验 QNN branch 的独立贡献         |

## 3. 主结果表

| Dataset          |   Shot | Model                   |   Runs | OA mean±std   | AA mean±std   | Kappa mean±std   | Macro-F1 mean±std   | Weighted-F1 mean±std   |
|:-----------------|-------:|:------------------------|-------:|:--------------|:--------------|:-----------------|:--------------------|:-----------------------|
| pavia_university |     10 | hybridsn_small          |      5 | 0.8226±0.0492 | 0.8431±0.0582 | 0.7731±0.0620    | 0.7920±0.0814       | 0.8280±0.0519          |
| pavia_university |     10 | spectral_qnn_multiproto |      5 | 0.8631±0.0259 | 0.8947±0.0222 | 0.8247±0.0318    | 0.8612±0.0222       | 0.8693±0.0228          |
| salinas          |     10 | hybridsn_small          |      5 | 0.9360±0.0159 | 0.9606±0.0146 | 0.9288±0.0176    | 0.9544±0.0155       | 0.9362±0.0157          |
| salinas          |     10 | spectral_qnn_multiproto |      5 | 0.9066±0.0373 | 0.9553±0.0171 | 0.8958±0.0420    | 0.9494±0.0178       | 0.9022±0.0439          |

## 4. 相对 HybridSN-small 的提升

| Dataset          |   Shot | Model                   |        ΔOA |         ΔAA |     ΔKappa |   ΔMacro-F1 |   ΔWeighted-F1 |
|:-----------------|-------:|:------------------------|-----------:|------------:|-----------:|------------:|---------------:|
| pavia_university |     10 | spectral_qnn_multiproto |  0.0404592 |  0.0516054  |  0.0515627 |  0.0692059  |      0.0412453 |
| salinas          |     10 | spectral_qnn_multiproto | -0.0293854 | -0.00534857 | -0.0329918 | -0.00506286 |     -0.0339526 |

## 5. QNN vs MLP

No completed comparison rows.

## 6. Paired seed 分析

尚无 Macro-F1 paired seed delta。

## 7. Gate 分析

`mean_gate` 越高表示 spectral residual logits 在最终 logits 中被保留得越多。
- pavia_university 10-shot spectral_qnn_multiproto: mean_gate=0.5401, correct=0.5401, wrong=0.5401
- salinas 10-shot spectral_qnn_multiproto: mean_gate=0.5243, correct=0.5246, wrong=0.5213

## 8. 初步结论

QNN 与 MLP 的 direct comparison 尚未齐全，暂不判断 spectral branch 类型贡献。

## 运行配置

- datasets: ['salinas', 'pavia_university']
- shots: [10]
- seeds: [0, 1, 2, 3, 4]
- models: ['spectral_qnn_multiproto']
- monitor: validation macro_f1
- metric_weight: 0.2
- temperature: 0.2
- failures: 0
