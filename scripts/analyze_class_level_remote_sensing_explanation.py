from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["precision", "recall", "f1", "accuracy"]


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    shot: int
    qnn_label: str
    baseline_metrics_dir: Path
    baseline_confusion_dir: Path
    qnn_metrics_dir: Path
    qnn_confusion_dir: Path
    qnn_per_class_pattern: str
    qnn_confusion_pattern: str
    baseline_per_class_pattern: str = "{dataset}_shot{shot}_seed{seed}_per_class.csv"
    baseline_confusion_pattern: str = "{dataset}_shot{shot}_seed{seed}.csv"


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = [
        ExperimentSpec(
            dataset="salinas",
            shot=10,
            qnn_label="ConfidenceGuard-C + SupCon",
            baseline_metrics_dir=Path("result/hybridsn_small_fewshot_pavia_salinas_5_10shot/metrics"),
            baseline_confusion_dir=Path("result/hybridsn_small_fewshot_pavia_salinas_5_10shot/confusion_matrices"),
            qnn_metrics_dir=Path("result/qnn_confguardc_penalty01_supcon_phase1_salinas_pavia_10shot/metrics"),
            qnn_confusion_dir=Path("result/qnn_confguardc_penalty01_supcon_phase1_salinas_pavia_10shot/confusion_matrices"),
            qnn_per_class_pattern="{dataset}_spectral_qnn_confguard_gated_supcon_shot{shot}_seed{seed}_per_class.csv",
            qnn_confusion_pattern="{dataset}_spectral_qnn_confguard_gated_supcon_shot{shot}_seed{seed}.csv",
        ),
        ExperimentSpec(
            dataset="pavia_university",
            shot=10,
            qnn_label="ConfidenceGuard-C + SupCon",
            baseline_metrics_dir=Path("result/hybridsn_small_fewshot_pavia_salinas_5_10shot/metrics"),
            baseline_confusion_dir=Path("result/hybridsn_small_fewshot_pavia_salinas_5_10shot/confusion_matrices"),
            qnn_metrics_dir=Path("result/qnn_confguardc_penalty01_supcon_phase1_salinas_pavia_10shot/metrics"),
            qnn_confusion_dir=Path("result/qnn_confguardc_penalty01_supcon_phase1_salinas_pavia_10shot/confusion_matrices"),
            qnn_per_class_pattern="{dataset}_spectral_qnn_confguard_gated_supcon_shot{shot}_seed{seed}_per_class.csv",
            qnn_confusion_pattern="{dataset}_spectral_qnn_confguard_gated_supcon_shot{shot}_seed{seed}.csv",
        ),
        ExperimentSpec(
            dataset="indian_pines",
            shot=5,
            qnn_label="Spectral QNN + SupCon",
            baseline_metrics_dir=Path("result/hybridsn_small_fewshot_3datasets/metrics"),
            baseline_confusion_dir=Path("result/hybridsn_small_fewshot_3datasets/confusion_matrices"),
            qnn_metrics_dir=Path("result/hybridsn_small_spectral_qnn_gated_supcon_indian_pines_5_10shot/metrics"),
            qnn_confusion_dir=Path("result/hybridsn_small_spectral_qnn_gated_supcon_indian_pines_5_10shot/confusion_matrices"),
            qnn_per_class_pattern="{dataset}_spectral_gated_qnn_supcon_shot{shot}_seed{seed}_per_class.csv",
            qnn_confusion_pattern="{dataset}_spectral_gated_qnn_supcon_shot{shot}_seed{seed}.csv",
        ),
        ExperimentSpec(
            dataset="indian_pines",
            shot=10,
            qnn_label="Spectral QNN + SupCon",
            baseline_metrics_dir=Path("result/hybridsn_small_fewshot_3datasets/metrics"),
            baseline_confusion_dir=Path("result/hybridsn_small_fewshot_3datasets/confusion_matrices"),
            qnn_metrics_dir=Path("result/hybridsn_small_spectral_qnn_gated_supcon_indian_pines_5_10shot/metrics"),
            qnn_confusion_dir=Path("result/hybridsn_small_spectral_qnn_gated_supcon_indian_pines_5_10shot/confusion_matrices"),
            qnn_per_class_pattern="{dataset}_spectral_gated_qnn_supcon_shot{shot}_seed{seed}_per_class.csv",
            qnn_confusion_pattern="{dataset}_spectral_gated_qnn_supcon_shot{shot}_seed{seed}.csv",
        ),
    ]

    per_seed = []
    confusion_rows = []
    for spec in specs:
        for seed in args.seeds:
            per_seed.append(load_per_class_delta(spec, seed))
            confusion_rows.extend(load_confusion_delta(spec, seed))

    per_seed_df = pd.concat(per_seed, ignore_index=True)
    per_seed_df.to_csv(out / "per_class_seed_delta.csv", index=False)

    class_summary = summarize_per_class(per_seed_df)
    class_summary.to_csv(out / "per_class_delta_summary.csv", index=False)

    confusion_df = pd.DataFrame(confusion_rows)
    confusion_df.to_csv(out / "confusion_pair_seed_delta.csv", index=False)

    confusion_summary = summarize_confusion(confusion_df)
    confusion_summary.to_csv(out / "confusion_pair_delta_summary.csv", index=False)

    write_report(out, class_summary, confusion_summary, per_seed_df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="result/class_level_remote_sensing_explanation_20260531",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser


def load_per_class_delta(spec: ExperimentSpec, seed: int) -> pd.DataFrame:
    base_path = spec.baseline_metrics_dir / spec.baseline_per_class_pattern.format(
        dataset=spec.dataset, shot=spec.shot, seed=seed
    )
    qnn_path = spec.qnn_metrics_dir / spec.qnn_per_class_pattern.format(
        dataset=spec.dataset, shot=spec.shot, seed=seed
    )
    base = pd.read_csv(base_path)
    qnn = pd.read_csv(qnn_path)
    base = ensure_per_class_columns(base)
    qnn = ensure_per_class_columns(qnn)
    merged = base.merge(qnn, on=["class_id", "class_name", "support"], suffixes=("_baseline", "_qnn"))
    merged.insert(0, "qnn_label", spec.qnn_label)
    merged.insert(0, "seed", seed)
    merged.insert(0, "shot", spec.shot)
    merged.insert(0, "dataset", spec.dataset)
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_qnn"] - merged[f"{metric}_baseline"]
    merged["support_weighted_delta_f1"] = merged["support"] * merged["delta_f1"]
    merged["support_weighted_delta_recall"] = merged["support"] * merged["delta_recall"]
    return merged


def ensure_per_class_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "accuracy" not in frame.columns and "recall" in frame.columns:
        frame["accuracy"] = frame["recall"]
    return frame


def load_confusion_delta(spec: ExperimentSpec, seed: int) -> list[dict[str, object]]:
    base_path = spec.baseline_confusion_dir / spec.baseline_confusion_pattern.format(
        dataset=spec.dataset, shot=spec.shot, seed=seed
    )
    qnn_path = spec.qnn_confusion_dir / spec.qnn_confusion_pattern.format(
        dataset=spec.dataset, shot=spec.shot, seed=seed
    )
    base = read_confusion(base_path)
    qnn = read_confusion(qnn_path)
    names = load_class_names(spec, seed)
    rows = []
    for true_id in range(base.shape[0]):
        for pred_id in range(base.shape[1]):
            if true_id == pred_id:
                continue
            baseline_count = int(base[true_id, pred_id])
            qnn_count = int(qnn[true_id, pred_id])
            delta = qnn_count - baseline_count
            if baseline_count == 0 and qnn_count == 0:
                continue
            rows.append(
                {
                    "dataset": spec.dataset,
                    "shot": spec.shot,
                    "seed": seed,
                    "qnn_label": spec.qnn_label,
                    "true_class_id": true_id,
                    "true_class_name": names.get(str(true_id), str(true_id)),
                    "pred_class_id": pred_id,
                    "pred_class_name": names.get(str(pred_id), str(pred_id)),
                    "baseline_count": baseline_count,
                    "qnn_count": qnn_count,
                    "delta_count": delta,
                }
            )
    return rows


def read_confusion(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=int)


def load_class_names(spec: ExperimentSpec, seed: int) -> dict[str, str]:
    json_path = spec.baseline_metrics_dir / f"{spec.dataset}_shot{spec.shot}_seed{seed}.json"
    if json_path.exists():
        return json.loads(json_path.read_text())["class_names"]
    per_class = pd.read_csv(
        spec.baseline_metrics_dir
        / spec.baseline_per_class_pattern.format(dataset=spec.dataset, shot=spec.shot, seed=seed)
    )
    return {str(row.class_id): row.class_name for row in per_class.itertuples(index=False)}


def summarize_per_class(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "shot", "qnn_label", "class_id", "class_name"]
    agg = {
        "support": "mean",
        "precision_baseline": "mean",
        "precision_qnn": "mean",
        "recall_baseline": "mean",
        "recall_qnn": "mean",
        "f1_baseline": "mean",
        "f1_qnn": "mean",
        "delta_precision": ["mean", "std"],
        "delta_recall": ["mean", "std"],
        "delta_f1": ["mean", "std"],
        "delta_accuracy": ["mean", "std"],
    }
    summary = df.groupby(group_cols).agg(agg)
    summary.columns = ["_".join([part for part in col if part]) for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    for metric in ["precision", "recall", "f1", "accuracy"]:
        summary[f"positive_{metric}_seeds"] = (
            df.assign(pos=df[f"delta_{metric}"] > 0)
            .groupby(group_cols)["pos"]
            .sum()
            .reset_index(drop=True)
            .astype(int)
        )
        summary[f"negative_{metric}_seeds"] = (
            df.assign(neg=df[f"delta_{metric}"] < 0)
            .groupby(group_cols)["neg"]
            .sum()
            .reset_index(drop=True)
            .astype(int)
        )
    return summary.sort_values(["dataset", "shot", "delta_f1_mean"], ascending=[True, True, True])


def summarize_confusion(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = [
        "dataset",
        "shot",
        "qnn_label",
        "true_class_id",
        "true_class_name",
        "pred_class_id",
        "pred_class_name",
    ]
    summary = (
        df.groupby(group_cols)
        .agg(
            baseline_count_mean=("baseline_count", "mean"),
            qnn_count_mean=("qnn_count", "mean"),
            delta_count_mean=("delta_count", "mean"),
            delta_count_sum=("delta_count", "sum"),
            increased_seeds=("delta_count", lambda values: int((values > 0).sum())),
            decreased_seeds=("delta_count", lambda values: int((values < 0).sum())),
        )
        .reset_index()
    )
    return summary.sort_values(["dataset", "shot", "delta_count_mean"], ascending=[True, True, False])


def write_report(out: Path, class_summary: pd.DataFrame, confusion_summary: pd.DataFrame, per_seed: pd.DataFrame) -> None:
    lines = [
        "# 类别级遥感解释实验汇总",
        "",
        "## 实验范围",
        "",
        "- 主诊断：ConfidenceGuard-C + SupCon vs HybridSN-small，覆盖 Salinas 10-shot 与 Pavia University 10-shot。",
        "- 补充诊断：Spectral QNN + SupCon vs HybridSN-small，覆盖 Indian Pines 5-shot 与 10-shot。",
        "- 每个设置使用 seeds 0--4，按 class 对齐 precision、recall、F1、accuracy，并比较 confusion pair 的错误计数变化。",
        "",
        "## 类别级结论",
        "",
    ]
    for (dataset, shot), group in class_summary.groupby(["dataset", "shot"]):
        lines.extend(dataset_section(dataset, int(shot), group, confusion_summary, per_seed))
    lines.extend(
        [
            "## 论文表述建议",
            "",
            "1. Salinas 10-shot 的负迁移主要不是所有类别一起下降，而是由少数高支持度地物类别的 F1/recall 损失放大为 OA 与 Weighted-F1 下降。",
            "2. QNN 分支在部分光谱相近或 baseline 不稳定的类别上仍能改善 Macro-F1，因此不能简单写成 QNN 无效。",
            "3. 更准确的表述是：QNN spectral residual 的收益具有类别条件性；当强 baseline 已经稳定识别大类时，未校准 residual 会对这些类别造成扰动。",
            "4. 后续应把 validation-calibrated class mask 或 confusion-pair mask 作为主要改进方向。",
            "",
        ]
    )
    (out / "class_level_report.md").write_text("\n".join(lines), encoding="utf-8")


def dataset_section(
    dataset: str,
    shot: int,
    group: pd.DataFrame,
    confusion_summary: pd.DataFrame,
    per_seed: pd.DataFrame,
) -> list[str]:
    display_name = {
        "salinas": "Salinas",
        "pavia_university": "Pavia University",
        "indian_pines": "Indian Pines",
    }.get(dataset, dataset)
    worst = group.sort_values("delta_f1_mean").head(5)
    best = group.sort_values("delta_f1_mean", ascending=False).head(5)
    task_seed = per_seed[(per_seed["dataset"] == dataset) & (per_seed["shot"] == shot)]
    mean_delta = task_seed[["delta_recall", "delta_f1", "support_weighted_delta_f1"]].mean()
    weighted_delta_f1 = (
        task_seed.groupby("seed")
        .apply(lambda group: group["support_weighted_delta_f1"].sum() / group["support"].sum(), include_groups=False)
        .mean()
    )
    lines = [
        f"### {display_name} {shot}-shot",
        "",
        f"- mean per-class delta recall: {mean_delta['delta_recall']:+.4f}",
        f"- mean per-class delta F1: {mean_delta['delta_f1']:+.4f}",
        f"- support-weighted mean delta F1: {weighted_delta_f1:+.4f}",
        "",
        "**F1 下降最大的类别**",
        "",
        markdown_table(
            worst,
            [
                "class_name",
                "support_mean",
                "f1_baseline_mean",
                "f1_qnn_mean",
                "delta_f1_mean",
                "delta_recall_mean",
                "negative_f1_seeds",
            ],
        ),
        "",
        "**F1 提升最大的类别**",
        "",
        markdown_table(
            best,
            [
                "class_name",
                "support_mean",
                "f1_baseline_mean",
                "f1_qnn_mean",
                "delta_f1_mean",
                "delta_recall_mean",
                "positive_f1_seeds",
            ],
        ),
        "",
    ]
    task_conf = confusion_summary[(confusion_summary["dataset"] == dataset) & (confusion_summary["shot"] == shot)]
    if not task_conf.empty:
        lines.extend(
            [
                "**新增最多的混淆对（QNN 错误数 - HybridSN-small 错误数）**",
                "",
                markdown_table(
                    task_conf.sort_values("delta_count_mean", ascending=False).head(8),
                    [
                        "true_class_name",
                        "pred_class_name",
                        "baseline_count_mean",
                        "qnn_count_mean",
                        "delta_count_mean",
                        "increased_seeds",
                    ],
                ),
                "",
                "**减少最多的混淆对**",
                "",
                markdown_table(
                    task_conf.sort_values("delta_count_mean").head(8),
                    [
                        "true_class_name",
                        "pred_class_name",
                        "baseline_count_mean",
                        "qnn_count_mean",
                        "delta_count_mean",
                        "decreased_seeds",
                    ],
                ),
                "",
            ]
        )
    return lines


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    rounded = df.loc[:, columns].copy()
    for column in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in rounded.to_numpy()]
    return "\n".join([header, sep, *rows])


if __name__ == "__main__":
    main()
