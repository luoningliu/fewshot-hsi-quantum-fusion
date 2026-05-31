from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["oa", "macro_f1", "weighted_f1"]


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paired = pd.concat(
        [
            load_confguardc_pavia_salinas(),
            load_indian_pines_supcon(),
        ],
        ignore_index=True,
    )
    paired.to_csv(out / "paired_seed_deltas.csv", index=False)

    summary = significance_summary(paired, args.bootstrap_samples, args.seed)
    summary.to_csv(out / "paired_significance_summary.csv", index=False)
    write_report(out, paired, summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="result/paired_significance_fewshot_20260531")
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_confguardc_pavia_salinas() -> pd.DataFrame:
    path = Path("result/qnn_confguardc_penalty01_supcon_phase1_salinas_pavia_10shot/paired_seed_delta_vs_hybridsn_small.csv")
    df = pd.read_csv(path)
    df["comparison"] = "ConfidenceGuard-C + SupCon vs HybridSN-small"
    return df[
        [
            "dataset",
            "shot",
            "seed",
            "model",
            "comparison",
            "hybridsn_oa",
            "qnn_oa",
            "delta_oa",
            "hybridsn_macro_f1",
            "qnn_macro_f1",
            "delta_macro_f1",
            "hybridsn_weighted_f1",
            "qnn_weighted_f1",
            "delta_weighted_f1",
        ]
    ]


def load_indian_pines_supcon() -> pd.DataFrame:
    baseline = pd.read_csv("result/hybridsn_small_fewshot_3datasets/metrics/all_runs.csv")
    qnn = pd.read_csv("result/hybridsn_small_spectral_qnn_gated_supcon_indian_pines_5_10shot/metrics/all_runs_metric_qnn.csv")
    baseline = baseline[(baseline["dataset"] == "indian_pines") & (baseline["shot"].isin([5, 10]))]
    qnn = qnn[(qnn["dataset"] == "indian_pines") & (qnn["shot"].isin([5, 10]))]
    merged = baseline.merge(qnn, on=["dataset", "shot", "seed"], suffixes=("_hybridsn", "_qnn"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "shot": int(row["shot"]),
                "seed": int(row["seed"]),
                "model": "spectral_qnn_gated_supcon",
                "comparison": "Spectral QNN + SupCon vs HybridSN-small",
                "hybridsn_oa": row["OA_hybridsn"],
                "qnn_oa": row["OA_qnn"],
                "delta_oa": row["OA_qnn"] - row["OA_hybridsn"],
                "hybridsn_macro_f1": row["Macro-F1_hybridsn"],
                "qnn_macro_f1": row["Macro-F1_qnn"],
                "delta_macro_f1": row["Macro-F1_qnn"] - row["Macro-F1_hybridsn"],
                "hybridsn_weighted_f1": row["Weighted-F1_hybridsn"],
                "qnn_weighted_f1": row["Weighted-F1_qnn"],
                "delta_weighted_f1": row["Weighted-F1_qnn"] - row["Weighted-F1_hybridsn"],
            }
        )
    return pd.DataFrame(rows)


def significance_summary(paired: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (dataset, shot, model, comparison), group in paired.groupby(["dataset", "shot", "model", "comparison"]):
        for metric in METRICS:
            deltas = group[f"delta_{metric}"].to_numpy(dtype=float)
            rows.append(test_metric(dataset, int(shot), model, comparison, metric, deltas, bootstrap_samples, rng))
    return pd.DataFrame(rows).sort_values(["dataset", "shot", "metric"])


def test_metric(
    dataset: str,
    shot: int,
    model: str,
    comparison: str,
    metric: str,
    deltas: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    n = len(deltas)
    mean = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    median = float(np.median(deltas))
    positive = int(np.sum(deltas > 0))
    negative = int(np.sum(deltas < 0))

    if n > 1 and std > 0:
        t_stat, t_p_two = stats.ttest_1samp(deltas, popmean=0.0)
        t_p_greater = float(t_p_two / 2) if t_stat > 0 else float(1 - t_p_two / 2)
        cohen_dz = float(mean / std)
    else:
        t_stat, t_p_two, t_p_greater, cohen_dz = np.nan, np.nan, np.nan, np.nan

    try:
        wilcoxon_two = stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided").pvalue
        wilcoxon_greater = stats.wilcoxon(deltas, zero_method="wilcox", alternative="greater").pvalue
        wilcoxon_less = stats.wilcoxon(deltas, zero_method="wilcox", alternative="less").pvalue
    except ValueError:
        wilcoxon_two = wilcoxon_greater = wilcoxon_less = np.nan

    sign_p_greater = stats.binomtest(positive, n, 0.5, alternative="greater").pvalue
    sign_p_less = stats.binomtest(negative, n, 0.5, alternative="greater").pvalue

    boot_means = rng.choice(deltas, size=(bootstrap_samples, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    bootstrap_p_greater = float(np.mean(boot_means <= 0.0))
    bootstrap_p_less = float(np.mean(boot_means >= 0.0))

    return {
        "dataset": dataset,
        "shot": shot,
        "model": model,
        "comparison": comparison,
        "metric": metric,
        "n": n,
        "mean_delta": mean,
        "median_delta": median,
        "std_delta": std,
        "ci95_low_bootstrap": float(ci_low),
        "ci95_high_bootstrap": float(ci_high),
        "positive_seeds": positive,
        "negative_seeds": negative,
        "t_stat": float(t_stat),
        "paired_t_p_two_sided": float(t_p_two),
        "paired_t_p_greater": float(t_p_greater),
        "wilcoxon_p_two_sided": float(wilcoxon_two),
        "wilcoxon_p_greater": float(wilcoxon_greater),
        "wilcoxon_p_less": float(wilcoxon_less),
        "sign_p_greater": float(sign_p_greater),
        "sign_p_less": float(sign_p_less),
        "bootstrap_p_mean_greater": bootstrap_p_greater,
        "bootstrap_p_mean_less": bootstrap_p_less,
        "cohen_dz": cohen_dz,
        "significant_positive_0.05": bool(mean > 0 and ci_low > 0 and wilcoxon_greater < 0.05),
        "significant_negative_0.05": bool(mean < 0 and ci_high < 0 and wilcoxon_less < 0.05),
    }


def write_report(out: Path, paired: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Few-shot QNN paired significance test",
        "",
        "## Scope",
        "",
        "- Pavia University 10-shot and Salinas 10-shot: ConfidenceGuard-C + SupCon vs HybridSN-small.",
        "- Indian Pines 5-shot and 10-shot: Spectral QNN + SupCon vs HybridSN-small.",
        "- Paired by identical random seed, n=5 for each dataset-shot setting.",
        "- Tests: paired t-test on seed deltas, Wilcoxon signed-rank test, sign test, bootstrap 95% CI of mean delta, Cohen's dz.",
        "",
        "## Main Results",
        "",
    ]
    for (dataset, shot), group in summary.groupby(["dataset", "shot"]):
        lines.extend(task_section(dataset, int(shot), group, paired))
    lines.extend(
        [
            "## Interpretation",
            "",
            "- With only five seeds, Wilcoxon/sign tests are conservative; results should be reported as paired evidence rather than definitive population-level proof unless p-values and bootstrap CI both support the claim.",
            "- Pavia University 10-shot has consistent positive paired deltas across all seeds for the main metrics, supporting a robust positive QNN effect.",
            "- Salinas 10-shot shows mixed seed signs: Macro-F1 trends positive, but OA and Weighted-F1 remain negative on average and are not positive-significant.",
            "- Indian Pines 10-shot is positive on average but not statistically strong under n=5; Indian Pines 5-shot is essentially neutral.",
            "",
        ]
    )
    (out / "paired_significance_report.md").write_text("\n".join(lines), encoding="utf-8")


def task_section(dataset: str, shot: int, group: pd.DataFrame, paired: pd.DataFrame) -> list[str]:
    display = {
        "indian_pines": "Indian Pines",
        "pavia_university": "Pavia University",
        "salinas": "Salinas",
    }.get(dataset, dataset)
    task = paired[(paired["dataset"] == dataset) & (paired["shot"] == shot)]
    lines = [f"### {display} {shot}-shot", ""]
    table = group[
        [
            "metric",
            "mean_delta",
            "ci95_low_bootstrap",
            "ci95_high_bootstrap",
            "positive_seeds",
            "negative_seeds",
            "paired_t_p_greater",
            "wilcoxon_p_greater",
            "wilcoxon_p_less",
            "cohen_dz",
        ]
    ].copy()
    lines.append(markdown_table(table))
    lines.append("")
    deltas = task[["seed", "delta_oa", "delta_macro_f1", "delta_weighted_f1"]].sort_values("seed")
    lines.append("Seed deltas:")
    lines.append("")
    lines.append(markdown_table(deltas))
    lines.append("")
    return lines


def markdown_table(df: pd.DataFrame) -> str:
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in show.to_numpy()]
    return "\n".join([header, sep, *rows])


if __name__ == "__main__":
    main()
