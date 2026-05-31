from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT = Path("result/resource_complexity_summary_20260531")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    rows.extend(
        summarize_runs(
            "HybridSN-small",
            "result/hybridsn_small_fewshot_3datasets/metrics/all_runs.csv",
            datasets=["indian_pines"],
            shots=[5, 10],
            qnn=False,
            notes="full HybridSN-small training and full patch inference",
        )
    )
    rows.extend(
        summarize_runs(
            "HybridSN-small",
            "result/hybridsn_small_fewshot_pavia_salinas_5_10shot/metrics/all_runs.csv",
            datasets=["pavia_university", "salinas"],
            shots=[10],
            qnn=False,
            notes="full HybridSN-small training and full patch inference",
        )
    )
    rows.extend(
        summarize_runs(
            "Spectral QNN + SupCon",
            "result/hybridsn_small_spectral_qnn_gated_supcon_indian_pines_5_10shot/metrics/all_runs_metric_qnn.csv",
            datasets=["indian_pines"],
            shots=[5, 10],
            qnn=True,
            notes="trained on cached frozen HybridSN-small features; encoder cost not included in train_time",
        )
    )
    rows.extend(
        summarize_runs(
            "ConfidenceGuard-C + SupCon",
            "result/qnn_confguardc_penalty01_supcon_phase1_salinas_pavia_10shot/metrics/all_runs_metric_qnn.csv",
            datasets=["pavia_university", "salinas"],
            shots=[10],
            qnn=True,
            notes="trained on cached frozen HybridSN-small features; encoder cost not included in train_time",
        )
    )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "resource_complexity_by_task.csv", index=False)

    model_summary = summarize_by_model(summary)
    model_summary.to_csv(OUT / "resource_complexity_by_model.csv", index=False)

    quantum_resource = pd.DataFrame(
        [
            {
                "qnn_variant": "standard spectral QNN",
                "qubits": 6,
                "quantum_layers": 1,
                "encoding": "angle encoding after tanh linear projection",
                "entanglement": "linear CNOT chain",
                "trainable_quantum_parameters": 18,
                "data_encoding_ry_gates": 6,
                "rot_gates_per_layer": 6,
                "cnot_gates_per_layer": 5,
                "measured_observables": 6,
                "qnode_evaluations_per_sample_forward": 1,
                "simulator_backend": "lightning.qubit",
                "diff_method": "adjoint",
            }
        ]
    )
    quantum_resource.to_csv(OUT / "quantum_resource_summary.csv", index=False)

    write_report(summary, model_summary, quantum_resource)


def summarize_runs(model_name: str, path: str, datasets: list[str], shots: list[int], qnn: bool, notes: str) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    df = df[df["dataset"].isin(datasets) & df["shot"].isin(shots)].copy()
    rows = []
    for (dataset, shot), group in df.groupby(["dataset", "shot"]):
        row = {
            "dataset": dataset,
            "shot": int(shot),
            "model": model_name,
            "runs": len(group),
            "trainable_parameters_mean": group["trainable_parameters"].mean(),
            "trainable_parameters_std": group["trainable_parameters"].std(ddof=0),
            "train_time_mean_s": group["train_time_seconds"].mean(),
            "train_time_std_s": group["train_time_seconds"].std(ddof=0),
            "test_time_mean_s": group["test_time_seconds"].mean(),
            "test_time_std_s": group["test_time_seconds"].std(ddof=0),
            "OA_mean": group["OA"].mean(),
            "Macro-F1_mean": group["Macro-F1"].mean(),
            "Weighted-F1_mean": group["Weighted-F1"].mean(),
            "is_qnn_branch": qnn,
            "deployment_requires_encoder": True,
            "frozen_encoder_parameters": 99488 if qnn else 0,
            "estimated_deployment_parameters": group["trainable_parameters"].mean() + (99488 if qnn else 0),
            "qubits": 6 if qnn else None,
            "quantum_layers": 1 if qnn else None,
            "trainable_quantum_parameters": 18 if qnn else None,
            "notes": notes,
        }
        rows.append(row)
    return rows


def summarize_by_model(summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "is_qnn_branch"]
    return (
        summary.groupby(group_cols)
        .agg(
            task_count=("dataset", "count"),
            trainable_parameters_mean=("trainable_parameters_mean", "mean"),
            estimated_deployment_parameters_mean=("estimated_deployment_parameters", "mean"),
            train_time_mean_s=("train_time_mean_s", "mean"),
            test_time_mean_s=("test_time_mean_s", "mean"),
            OA_mean=("OA_mean", "mean"),
            Macro_F1_mean=("Macro-F1_mean", "mean"),
            Weighted_F1_mean=("Weighted-F1_mean", "mean"),
        )
        .reset_index()
    )


def write_report(summary: pd.DataFrame, model_summary: pd.DataFrame, quantum_resource: pd.DataFrame) -> None:
    lines = [
        "# 资源复杂度分析汇总",
        "",
        "## 口径说明",
        "",
        "- HybridSN-small 的 `trainable_parameters` 是完整 encoder + classifier 的训练参数。",
        "- QNN 分支实验使用已经缓存的 HybridSN-small embedding 和中心像素光谱，因此 `train_time` 只统计 QNN/gate/head 阶段，不包含训练或提取 encoder 特征的成本。",
        "- 部署时 QNN 分支仍需要 HybridSN-small encoder 提供 embedding，因此报告同时给出 `estimated_deployment_parameters = frozen_encoder_parameters + QNN_branch_trainable_parameters`。",
        "- QNN 运行在 PennyLane `lightning.qubit` 经典模拟器上，当前结果不主张量子速度优势。",
        "",
        "## 按任务汇总",
        "",
        markdown_table(format_percent(summary.copy(), ["OA_mean", "Macro-F1_mean", "Weighted-F1_mean"])),
        "",
        "## 按模型汇总",
        "",
        markdown_table(format_percent(model_summary.copy(), ["OA_mean", "Macro_F1_mean", "Weighted_F1_mean"])),
        "",
        "## 量子线路资源",
        "",
        markdown_table(quantum_resource),
        "",
        "## 关键结论",
        "",
        "1. QNN 分支的可训练参数量明显小于完整 HybridSN-small。Indian Pines 的 Spectral QNN + SupCon 分支为 2176 个可训练参数；Pavia/Salinas 的 ConfidenceGuard-C 分支分别为 1477 和 2212 个可训练参数，而 HybridSN-small 约为 9.9 万个参数。",
        "2. 该参数优势不等于端到端部署更轻。QNN 分支需要 frozen HybridSN-small encoder，因此估计部署参数约为 encoder 参数 99488 加 QNN 分支参数。",
        "3. 在经典模拟器上，QNN 不具备速度优势。虽然 QNN 分支训练时间较短，但这是因为其复用了缓存特征；测试阶段仍需逐样本量子线路模拟，且正式端到端推理还要叠加 encoder 特征提取成本。",
        "4. 标准 spectral QNN 使用 6 qubits、1 层量子线路、18 个可训练量子参数、线性 CNOT entanglement，每个样本前向需要 1 次 QNode 评估并测量 6 个 Pauli-Z 期望值。",
        "5. 因此，当前论文中 QNN 的定位应是少样本 spectral decision-boundary regularizer，而不是计算效率或 quantum speedup 方法。",
        "",
    ]
    (OUT / "resource_complexity_report.md").write_text("\n".join(lines), encoding="utf-8")


def format_percent(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col] * 100
    return df


def markdown_table(df: pd.DataFrame) -> str:
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    header = "| " + " | ".join(show.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = ["| " + " | ".join(str(x) for x in row) + " |" for row in show.to_numpy()]
    return "\n".join([header, sep, *rows])


if __name__ == "__main__":
    main()
