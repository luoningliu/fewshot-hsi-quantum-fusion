# 资源复杂度分析汇总

## 口径说明

- HybridSN-small 的 `trainable_parameters` 是完整 encoder + classifier 的训练参数。
- QNN 分支实验使用已经缓存的 HybridSN-small embedding 和中心像素光谱，因此 `train_time` 只统计 QNN/gate/head 阶段，不包含训练或提取 encoder 特征的成本。
- 部署时 QNN 分支仍需要 HybridSN-small encoder 提供 embedding，因此报告同时给出 `estimated_deployment_parameters = frozen_encoder_parameters + QNN_branch_trainable_parameters`。
- QNN 运行在 PennyLane `lightning.qubit` 经典模拟器上，当前结果不主张量子速度优势。

## 按任务汇总

| dataset | shot | model | runs | trainable_parameters_mean | trainable_parameters_std | train_time_mean_s | train_time_std_s | test_time_mean_s | test_time_std_s | OA_mean | Macro-F1_mean | Weighted-F1_mean | is_qnn_branch | deployment_requires_encoder | frozen_encoder_parameters | estimated_deployment_parameters | qubits | quantum_layers | trainable_quantum_parameters | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| indian_pines | 5 | HybridSN-small | 5 | 99488.00 | 0.00 | 220.20 | 26.34 | 32.97 | 0.51 | 72.02 | 63.81 | 73.27 | False | True | 0 | 99488.00 |  |  |  | full HybridSN-small training and full patch inference |
| indian_pines | 10 | HybridSN-small | 5 | 99488.00 | 0.00 | 272.32 | 14.69 | 30.99 | 2.01 | 80.12 | 71.53 | 80.87 | False | True | 0 | 99488.00 |  |  |  | full HybridSN-small training and full patch inference |
| pavia_university | 10 | HybridSN-small | 5 | 99033.00 | 0.00 | 142.98 | 47.41 | 149.50 | 15.15 | 82.26 | 79.20 | 82.80 | False | True | 0 | 99033.00 |  |  |  | full HybridSN-small training and full patch inference |
| salinas | 10 | HybridSN-small | 5 | 99488.00 | 0.00 | 204.23 | 45.90 | 160.22 | 22.02 | 93.60 | 95.44 | 93.62 | False | True | 0 | 99488.00 |  |  |  | full HybridSN-small training and full patch inference |
| indian_pines | 5 | Spectral QNN + SupCon | 5 | 2176.00 | 0.00 | 26.89 | 12.32 | 18.90 | 1.49 | 72.06 | 64.04 | 72.91 | True | True | 99488 | 101664.00 | 6.00 | 1.00 | 18.00 | trained on cached frozen HybridSN-small features; encoder cost not included in train_time |
| indian_pines | 10 | Spectral QNN + SupCon | 5 | 2176.00 | 0.00 | 30.59 | 11.08 | 17.65 | 0.43 | 81.07 | 72.26 | 81.79 | True | True | 99488 | 101664.00 | 6.00 | 1.00 | 18.00 | trained on cached frozen HybridSN-small features; encoder cost not included in train_time |
| pavia_university | 10 | ConfidenceGuard-C + SupCon | 5 | 1477.00 | 0.00 | 12.06 | 1.39 | 67.53 | 0.55 | 85.77 | 85.93 | 86.49 | True | True | 99488 | 100965.00 | 6.00 | 1.00 | 18.00 | trained on cached frozen HybridSN-small features; encoder cost not included in train_time |
| salinas | 10 | ConfidenceGuard-C + SupCon | 5 | 2212.00 | 0.00 | 25.44 | 4.27 | 84.91 | 0.26 | 92.60 | 96.12 | 92.64 | True | True | 99488 | 101700.00 | 6.00 | 1.00 | 18.00 | trained on cached frozen HybridSN-small features; encoder cost not included in train_time |

## 按模型汇总

| model | is_qnn_branch | task_count | trainable_parameters_mean | estimated_deployment_parameters_mean | train_time_mean_s | test_time_mean_s | OA_mean | Macro_F1_mean | Weighted_F1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ConfidenceGuard-C + SupCon | True | 2 | 1844.50 | 101332.50 | 18.75 | 76.22 | 89.18 | 91.03 | 89.56 |
| HybridSN-small | False | 4 | 99374.25 | 99374.25 | 209.93 | 93.42 | 82.00 | 77.49 | 82.64 |
| Spectral QNN + SupCon | True | 2 | 2176.00 | 101664.00 | 28.74 | 18.28 | 76.56 | 68.15 | 77.35 |

## 量子线路资源

| qnn_variant | qubits | quantum_layers | encoding | entanglement | trainable_quantum_parameters | data_encoding_ry_gates | rot_gates_per_layer | cnot_gates_per_layer | measured_observables | qnode_evaluations_per_sample_forward | simulator_backend | diff_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard spectral QNN | 6 | 1 | angle encoding after tanh linear projection | linear CNOT chain | 18 | 6 | 6 | 5 | 6 | 1 | lightning.qubit | adjoint |

## 关键结论

1. QNN 分支的可训练参数量明显小于完整 HybridSN-small。Indian Pines 的 Spectral QNN + SupCon 分支为 2176 个可训练参数；Pavia/Salinas 的 ConfidenceGuard-C 分支分别为 1477 和 2212 个可训练参数，而 HybridSN-small 约为 9.9 万个参数。
2. 该参数优势不等于端到端部署更轻。QNN 分支需要 frozen HybridSN-small encoder，因此估计部署参数约为 encoder 参数 99488 加 QNN 分支参数。
3. 在经典模拟器上，QNN 不具备速度优势。虽然 QNN 分支训练时间较短，但这是因为其复用了缓存特征；测试阶段仍需逐样本量子线路模拟，且正式端到端推理还要叠加 encoder 特征提取成本。
4. 标准 spectral QNN 使用 6 qubits、1 层量子线路、18 个可训练量子参数、线性 CNOT entanglement，每个样本前向需要 1 次 QNode 评估并测量 6 个 Pauli-Z 期望值。
5. 因此，当前论文中 QNN 的定位应是少样本 spectral decision-boundary regularizer，而不是计算效率或 quantum speedup 方法。
