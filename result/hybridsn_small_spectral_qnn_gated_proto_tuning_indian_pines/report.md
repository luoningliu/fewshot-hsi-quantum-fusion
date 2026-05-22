# CE + Prototype Loss Experiment

## Purpose

This experiment continued the Spectral QNN Gated Fusion line by adding an auxiliary prototype classification objective:

```text
loss = CrossEntropy(logits, y) + lambda * CrossEntropy(-distance(feature, prototype) / temperature, y)
```

The goal was to stabilize few-shot class boundaries, especially in the 5-shot setting where prior QNN tuning improved Macro-F1 but did not strictly exceed HybridSN-small OA.

## Model

The model keeps the frozen HybridSN-small encoder and trains a lightweight spectral QNN gated fusion head:

```text
z_c = HybridSN-small encoder(patch)
x_s = center pixel PCA spectrum
q = QNN feature(x_s)
logits = Linear(z_c) + gate([z_c, x_s]) * Linear(q)
prototype logits = -distance([z_c, q], class_prototypes) / temperature
```

Configuration:

| Item | Value |
|---|---:|
| Dataset | Indian Pines |
| Shots | 5, 10 |
| Seeds | 0, 1, 2, 3, 4 |
| Qubits | 6 |
| QNN layers | 1 |
| Entanglement | linear |
| Gate | classwise |
| Prototype weight | 0.2 |
| Prototype temperature | 0.2 |
| Epochs / patience | 60 / 10 |
| Trainable parameters | 2,176 |

## Results

| Shot | Model | Runs | OA | AA | Kappa | Macro-F1 | Weighted-F1 | Delta OA | Delta Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | HybridSN-small | 5 | 72.02 ± 1.67 | 82.66 ± 1.45 | 68.66 ± 1.90 | 63.81 ± 2.17 | 73.27 ± 2.06 | 0.00 | 0.00 |
| 5 | Spectral QNN Gated Fusion + Prototype Loss | 5 | 72.05 ± 1.50 | 82.52 ± 0.68 | 68.68 ± 1.66 | 64.12 ± 1.73 | 72.84 ± 2.03 | +0.03 | +0.32 |
| 10 | HybridSN-small | 5 | 80.12 ± 2.56 | 88.64 ± 0.94 | 77.64 ± 2.79 | 71.53 ± 2.02 | 80.87 ± 2.70 | 0.00 | 0.00 |
| 10 | Spectral QNN Gated Fusion + Prototype Loss | 5 | 80.88 ± 2.40 | 88.87 ± 1.32 | 78.49 ± 2.62 | 71.82 ± 1.92 | 81.61 ± 2.43 | +0.76 | +0.29 |

## Interpretation

CE + prototype loss is the first setting where the spectral QNN gated fusion model strictly exceeds HybridSN-small on both OA and Macro-F1 in 5-shot and 10-shot. The 5-shot OA gain is small, so it should be described as a marginal but positive improvement, while the Macro-F1 gain is more meaningful for class-balanced few-shot evaluation.

The result supports the hypothesis that the earlier 5-shot bottleneck was caused more by unstable class boundaries than by insufficient QNN capacity. Adding a metric/prototype objective is a better direction than only increasing QNN depth.

## Result Folders

| Shot | Directory |
|---:|---|
| 5 | `result/hybridsn_small_spectral_qnn_gated_proto_fewshot_indian_pines_pw02/` |
| 10 | `result/hybridsn_small_spectral_qnn_gated_proto_fewshot_indian_pines_pw02_10shot/` |
