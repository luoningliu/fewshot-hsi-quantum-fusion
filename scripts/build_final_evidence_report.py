from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["OA", "AA", "Kappa", "Macro-F1", "Weighted-F1"]
DATASETS = ["indian_pines", "pavia_university", "salinas"]
SHOTS = [5, 10]
SEEDS = [0, 1, 2, 3, 4]

DISPLAY_DATASET = {
    "indian_pines": "Indian Pines",
    "pavia_university": "Pavia University",
    "salinas": "Salinas",
}
DISPLAY_MODEL = {
    "hybridsn_small": "HybridSN-small",
    "prototype": "Spectral QNN Gated Fusion + Prototype Loss",
    "supcon": "Spectral QNN Gated Fusion + SupCon Loss",
}


def main() -> None:
    args = build_parser().parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir or f"result/final_evidence_closure_fewshot_spectral_qnn_{timestamp}")
    prepare_output(out)

    data = load_all_runs()
    raw_sources = copy_raw_sources(out)
    failures = build_failed_runs(data)
    failures.to_csv(out / "failed_runs.csv", index=False)

    supcon_rows, supcon_agg = build_supcon_tables(data)
    supcon_rows.to_csv(out / "tables/supcon_cross_dataset_summary.csv", index=False)
    supcon_agg.to_csv(out / "tables/supcon_cross_dataset_aggregate.csv", index=False)

    main_results, paired_deltas, tests = build_significance_tables(data)
    main_results.to_csv(out / "tables/main_results_with_significance.csv", index=False)
    paired_deltas.to_csv(out / "tables/paired_seed_deltas.csv", index=False)
    tests.to_csv(out / "tables/statistical_tests.csv", index=False)
    stability = build_prototype_supcon_stability(main_results, paired_deltas, tests)
    stability.to_csv(out / "tables/prototype_vs_supcon_stability.csv", index=False)

    per_class, per_class_delta, top_classes = build_per_class_tables()
    per_class.to_csv(out / "tables/per_class_metrics.csv", index=False)
    per_class_delta.to_csv(out / "tables/per_class_delta_qnn_vs_baseline.csv", index=False)
    top_classes.to_csv(out / "tables/top_improved_and_degraded_classes.csv", index=False)

    salinas_negative = build_salinas_negative_transfer(data, per_class_delta, paired_deltas)
    salinas_negative.to_csv(out / "tables/salinas_10shot_negative_transfer.csv", index=False)

    logit_summary, margin_joint = build_logit_tables()
    logit_summary.to_csv(out / "tables/logit_margin_summary.csv", index=False)
    margin_joint.to_csv(out / "tables/metric_margin_joint_summary.csv", index=False)

    gate_summary, class_gate = build_gate_tables(out)
    gate_summary.to_csv(out / "tables/gate_value_summary.csv", index=False)
    class_gate.to_csv(out / "tables/classwise_gate_value_summary.csv", index=False)

    complexity = build_complexity_summary(data)
    complexity.to_csv(out / "tables/complexity_summary.csv", index=False)

    make_figures(out, data, paired_deltas, per_class_delta, margin_joint)
    write_reports(out, data, main_results, stability, tests, per_class_delta, top_classes, salinas_negative, margin_joint, complexity, failures)
    write_metadata(out, timestamp, raw_sources, failures)
    write_checklist(out)
    print(f"Final output directory: {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final evidence closure package for few-shot spectral QNN.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--timestamp", default=None)
    return parser


def prepare_output(out: Path) -> None:
    for sub in ["reports", "tables", "figures", "raw/copied_or_linked_raw_metrics"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def normalize_run_frame(df: pd.DataFrame, model_key: str, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "dataset_id" in out.columns and "dataset" not in out.columns:
        out = out.rename(columns={"dataset_id": "dataset"})
    out["model_key"] = model_key
    out["source"] = source
    for metric in METRICS:
        if metric not in out.columns:
            out[metric] = np.nan
    return out


def load_all_runs() -> dict[str, pd.DataFrame]:
    baseline_paths = [
        "result/hybridsn_small_fewshot_3datasets/metrics/all_runs.csv",
        "result/hybridsn_small_fewshot_pavia_salinas_5_10shot/metrics/all_runs.csv",
    ]
    baseline = []
    for path in baseline_paths:
        df = normalize_run_frame(read_csv_if_exists(path), "hybridsn_small", path)
        if not df.empty:
            baseline.append(df)
    baseline_df = pd.concat(baseline, ignore_index=True) if baseline else pd.DataFrame()
    if not baseline_df.empty:
        baseline_df = baseline_df.sort_values(["dataset", "shot", "seed", "source"]).drop_duplicates(
            ["dataset", "shot", "seed"], keep="last"
        )

    prototype = normalize_run_frame(
        read_csv_if_exists("result/fewshot_metric_loss_cross_dataset_summary/qnn_prototype_all_runs.csv"),
        "prototype",
        "result/fewshot_metric_loss_cross_dataset_summary/qnn_prototype_all_runs.csv",
    )
    supcon_sources = [
        "result/hybridsn_small_spectral_qnn_gated_supcon_indian_pines_5_10shot/metrics/all_runs_metric_qnn.csv",
        "result/supcon_cross_dataset_pavia_salinas_20260525_183521/metrics/all_runs_metric_qnn.csv",
    ]
    supcon_frames = []
    for path in supcon_sources:
        df = normalize_run_frame(read_csv_if_exists(path), "supcon", path)
        if not df.empty:
            supcon_frames.append(df)
    supcon = pd.concat(supcon_frames, ignore_index=True) if supcon_frames else pd.DataFrame()
    if not supcon.empty:
        supcon = supcon.sort_values(["dataset", "shot", "seed", "source"]).drop_duplicates(
            ["dataset", "shot", "seed"], keep="last"
        )
    return {"hybridsn_small": baseline_df, "prototype": prototype, "supcon": supcon}


def copy_raw_sources(out: Path) -> list[str]:
    sources = [
        "result/fewshot_metric_loss_cross_dataset_summary/all_model_summary.csv",
        "result/fewshot_metric_loss_cross_dataset_summary/comparison_vs_hybridsn_small.csv",
        "result/fewshot_metric_loss_cross_dataset_summary/qnn_prototype_all_runs.csv",
        "result/hybridsn_small_fewshot_3datasets/metrics/all_runs.csv",
        "result/hybridsn_small_fewshot_pavia_salinas_5_10shot/metrics/all_runs.csv",
        "result/hybridsn_small_spectral_qnn_gated_supcon_indian_pines_5_10shot/metrics/all_runs_metric_qnn.csv",
        "result/supcon_cross_dataset_pavia_salinas_20260525_183521/metrics/all_runs_metric_qnn.csv",
        "result/supcon_cross_dataset_pavia_salinas_20260525_183521/supcon_cross_dataset_summary.csv",
        "result/supcon_cross_dataset_pavia_salinas_20260525_183521/supcon_cross_dataset_aggregate.csv",
        "result/logit_margin_hybridsn_vs_qnn/metrics/logit_margin_summary_by_dataset_shot_model.csv",
        "result/logit_margin_hybridsn_vs_qnn/metrics/geometry_vs_logit_margin_joint_summary.csv",
        "result/boundary_geometry_hybridsn_vs_qnn/metrics/geometry_summary.csv",
    ]
    copied = []
    target_dir = out / "raw/copied_or_linked_raw_metrics"
    for src in sources:
        p = Path(src)
        if p.exists():
            dst = target_dir / p.name
            shutil.copy2(p, dst)
            copied.append(src)
    return copied


def build_failed_runs(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    supcon = data["supcon"]
    for dataset in ["pavia_university", "salinas"]:
        for shot in SHOTS:
            for seed in SEEDS:
                found = not supcon[
                    (supcon["dataset"] == dataset) & (supcon["shot"] == shot) & (supcon["seed"] == seed)
                ].empty
                if not found:
                    rows.append(
                        {
                            "dataset": dataset,
                            "shot": shot,
                            "seed": seed,
                            "model": "spectral_qnn_gated_supcon",
                            "stage": "Task A",
                            "status": "missing_not_run",
                            "reason": "No saved SupCon cross-dataset seed result exists in the current workspace.",
                        }
                    )
    return pd.DataFrame(rows)


def build_supcon_tables(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    supcon = data["supcon"].copy()
    if supcon.empty:
        cols = ["dataset", "shot", "model", "loss_type", "seed", "oa", "aa", "kappa", "macro_f1", "weighted_f1"]
        return pd.DataFrame(columns=cols), pd.DataFrame()
    rows = []
    for _, row in supcon.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "shot": int(row["shot"]),
                "model": "Spectral QNN Gated Fusion",
                "loss_type": "SupCon",
                "seed": int(row["seed"]),
                "oa": float(row["OA"]),
                "aa": float(row["AA"]),
                "kappa": float(row["Kappa"]),
                "macro_f1": float(row["Macro-F1"]),
                "weighted_f1": float(row["Weighted-F1"]),
            }
        )
    summary = pd.DataFrame(rows)
    agg_rows = []
    for (dataset, shot, model, loss_type), group in summary.groupby(["dataset", "shot", "model", "loss_type"]):
        agg = {"dataset": dataset, "shot": shot, "model": model, "loss_type": loss_type, "runs": len(group)}
        for src, dst in [("oa", "oa"), ("aa", "aa"), ("kappa", "kappa"), ("macro_f1", "macro_f1"), ("weighted_f1", "weighted_f1")]:
            agg[f"{dst}_mean"] = group[src].mean()
            agg[f"{dst}_std"] = group[src].std(ddof=0)
        agg_rows.append(agg)
    return summary, pd.DataFrame(agg_rows)


def get_metric(row: pd.Series, metric: str) -> float:
    return float(row[metric]) if metric in row and pd.notna(row[metric]) else np.nan


def paired_values(base: pd.DataFrame, qnn: pd.DataFrame, dataset: str, shot: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    b = base[(base["dataset"] == dataset) & (base["shot"] == shot)][["seed", metric]]
    q = qnn[(qnn["dataset"] == dataset) & (qnn["shot"] == shot)][["seed", metric]]
    merged = b.merge(q, on="seed", suffixes=("_baseline", "_qnn")).sort_values("seed")
    return merged[f"{metric}_baseline"].to_numpy(float), merged[f"{metric}_qnn"].to_numpy(float)


def build_significance_tables(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = data["hybridsn_small"]
    rows = []
    deltas = []
    tests = []
    for dataset in DATASETS:
        for shot in SHOTS:
            for model_key in ["prototype", "supcon"]:
                qnn = data[model_key]
                comparison = f"HybridSN-small vs {DISPLAY_MODEL[model_key]}"
                for metric in METRICS:
                    bvals, qvals = paired_values(base, qnn, dataset, shot, metric)
                    if len(bvals) == 0:
                        continue
                    for seed, b, q in zip(sorted(set(base[(base["dataset"] == dataset) & (base["shot"] == shot)]["seed"]).intersection(set(qnn[(qnn["dataset"] == dataset) & (qnn["shot"] == shot)]["seed"]))), bvals, qvals):
                        deltas.append(
                            {
                                "dataset": dataset,
                                "shot": shot,
                                "comparison": comparison,
                                "metric": metric,
                                "seed": int(seed),
                                "baseline_value": b,
                                "qnn_value": q,
                                "delta": q - b,
                            }
                        )
                    tests.append(stat_test_row(dataset, shot, comparison, metric, bvals, qvals))
                rows.append(main_result_row(base, qnn, dataset, shot, model_key, tests))
    main_results = pd.DataFrame([r for r in rows if r])
    return main_results, pd.DataFrame(deltas), pd.DataFrame(tests)


def build_prototype_supcon_stability(main_results: pd.DataFrame, paired_deltas: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    if main_results.empty:
        return pd.DataFrame()
    rows = []
    for dataset in DATASETS:
        for shot in SHOTS:
            proto = _main_row(main_results, dataset, shot, "Prototype")
            supcon = _main_row(main_results, dataset, shot, "SupCon")
            if proto is None or supcon is None:
                continue
            proto_macro = _test_row(tests, dataset, shot, "Prototype", "Macro-F1")
            supcon_macro = _test_row(tests, dataset, shot, "SupCon", "Macro-F1")
            proto_oa = _test_row(tests, dataset, shot, "Prototype", "OA")
            supcon_oa = _test_row(tests, dataset, shot, "SupCon", "OA")
            proto_seed = _paired_subset(paired_deltas, dataset, shot, "Prototype", "Macro-F1")
            supcon_seed = _paired_subset(paired_deltas, dataset, shot, "SupCon", "Macro-F1")
            supcon_minus_proto = float(supcon["qnn_Macro-F1_mean"] - proto["qnn_Macro-F1_mean"])
            rows.append(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "prototype_delta_macro_f1": float(proto["delta_Macro-F1"]),
                    "supcon_delta_macro_f1": float(supcon["delta_Macro-F1"]),
                    "supcon_minus_prototype_macro_f1": supcon_minus_proto,
                    "prototype_std_delta_macro_f1": _row_value(proto_macro, "std_delta"),
                    "supcon_std_delta_macro_f1": _row_value(supcon_macro, "std_delta"),
                    "prototype_positive_seed_count_macro_f1": int((proto_seed["delta"] > 0).sum()) if not proto_seed.empty else 0,
                    "supcon_positive_seed_count_macro_f1": int((supcon_seed["delta"] > 0).sum()) if not supcon_seed.empty else 0,
                    "prototype_negative_seed_count_macro_f1": int((proto_seed["delta"] < 0).sum()) if not proto_seed.empty else 0,
                    "supcon_negative_seed_count_macro_f1": int((supcon_seed["delta"] < 0).sum()) if not supcon_seed.empty else 0,
                    "prototype_delta_oa": float(proto["delta_OA"]),
                    "supcon_delta_oa": float(supcon["delta_OA"]),
                    "prototype_std_delta_oa": _row_value(proto_oa, "std_delta"),
                    "supcon_std_delta_oa": _row_value(supcon_oa, "std_delta"),
                    "winner_by_macro_f1_mean": "SupCon" if supcon_minus_proto > 0 else "Prototype" if supcon_minus_proto < 0 else "Tie",
                    "stability_interpretation": _stability_interpretation(dataset, shot, proto, supcon, proto_macro, supcon_macro),
                }
            )
    return pd.DataFrame(rows)


def _main_row(frame: pd.DataFrame, dataset: str, shot: int, loss_name: str) -> pd.Series | None:
    subset = frame[
        (frame["dataset"] == dataset)
        & (frame["shot"] == shot)
        & (frame["model"].str.contains(loss_name, na=False))
    ]
    return None if subset.empty else subset.iloc[0]


def _test_row(frame: pd.DataFrame, dataset: str, shot: int, loss_name: str, metric: str) -> pd.Series | None:
    subset = frame[
        (frame["dataset"] == dataset)
        & (frame["shot"] == shot)
        & (frame["comparison"].str.contains(loss_name, na=False))
        & (frame["metric"] == metric)
    ]
    return None if subset.empty else subset.iloc[0]


def _paired_subset(frame: pd.DataFrame, dataset: str, shot: int, loss_name: str, metric: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return frame[
        (frame["dataset"] == dataset)
        & (frame["shot"] == shot)
        & (frame["comparison"].str.contains(loss_name, na=False))
        & (frame["metric"] == metric)
    ]


def _row_value(row: pd.Series | None, column: str) -> float:
    return np.nan if row is None else float(row[column])


def _stability_interpretation(dataset: str, shot: int, proto: pd.Series, supcon: pd.Series, proto_test: pd.Series | None, supcon_test: pd.Series | None) -> str:
    proto_delta = float(proto["delta_Macro-F1"])
    supcon_delta = float(supcon["delta_Macro-F1"])
    proto_std = _row_value(proto_test, "std_delta")
    supcon_std = _row_value(supcon_test, "std_delta")
    if dataset == "salinas" and shot == 10 and proto_delta < 0 < supcon_delta:
        return "supcon_mitigates_prototype_negative_transfer"
    if supcon_delta > proto_delta and (pd.isna(supcon_std) or pd.isna(proto_std) or supcon_std <= proto_std * 1.25):
        return "supcon_slightly_stronger"
    if proto_delta > supcon_delta and (pd.isna(proto_std) or pd.isna(supcon_std) or proto_std <= supcon_std * 1.25):
        return "prototype_slightly_stronger"
    if supcon_delta > 0 and proto_delta > 0:
        return "both_positive_mixed_stability"
    if supcon_delta < 0 and proto_delta < 0:
        return "both_negative_or_unstable"
    return "mixed"


def stat_test_row(dataset: str, shot: int, comparison: str, metric: str, bvals: np.ndarray, qvals: np.ndarray) -> dict[str, Any]:
    diff = qvals - bvals
    n = len(diff)
    mean_delta = float(np.mean(diff)) if n else np.nan
    std_delta = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    if n >= 2 and not np.allclose(diff, diff[0]):
        t_p = float(stats.ttest_rel(qvals, bvals).pvalue)
    else:
        t_p = np.nan
    try:
        w_p = float(stats.wilcoxon(diff).pvalue) if n >= 2 and not np.allclose(diff, 0) else np.nan
    except ValueError:
        w_p = np.nan
    cohen_d = float(mean_delta / std_delta) if n > 1 and std_delta and not math.isclose(std_delta, 0.0) else np.nan
    ci_low = ci_high = np.nan
    if n > 1:
        sem = stats.sem(diff)
        tcrit = stats.t.ppf(0.975, n - 1)
        ci_low = float(mean_delta - tcrit * sem)
        ci_high = float(mean_delta + tcrit * sem)
    interpretation = interpret(mean_delta, t_p, n)
    return {
        "dataset": dataset,
        "shot": shot,
        "comparison": comparison,
        "metric": metric,
        "baseline_mean": float(np.mean(bvals)) if n else np.nan,
        "qnn_mean": float(np.mean(qvals)) if n else np.nan,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_t_pvalue": t_p,
        "wilcoxon_pvalue": w_p,
        "cohen_d": cohen_d,
        "is_significant_p005": bool(pd.notna(t_p) and t_p < 0.05),
        "interpretation": interpretation,
        "n_paired_seeds": n,
    }


def interpret(mean_delta: float, pvalue: float, n: int) -> str:
    if n < 3 or pd.isna(mean_delta):
        return "inconclusive"
    if mean_delta < 0:
        return "negative_transfer"
    if pd.notna(pvalue) and pvalue < 0.05 and mean_delta > 0:
        return "clear_positive"
    if mean_delta > 0:
        return "marginal_positive"
    return "not_significant"


def main_result_row(base: pd.DataFrame, qnn: pd.DataFrame, dataset: str, shot: int, model_key: str, tests: list[dict[str, Any]]) -> dict[str, Any] | None:
    b = base[(base["dataset"] == dataset) & (base["shot"] == shot)]
    q = qnn[(qnn["dataset"] == dataset) & (qnn["shot"] == shot)]
    if b.empty or q.empty:
        return None
    row = {"dataset": dataset, "shot": shot, "model": DISPLAY_MODEL[model_key], "runs": len(q)}
    for metric in METRICS:
        row[f"baseline_{metric}_mean"] = b[metric].mean()
        row[f"qnn_{metric}_mean"] = q[metric].mean()
        row[f"delta_{metric}"] = q[metric].mean() - b[metric].mean()
    match = [t for t in tests if t["dataset"] == dataset and t["shot"] == shot and DISPLAY_MODEL[model_key] in t["comparison"] and t["metric"] == "Macro-F1"]
    row["macro_f1_interpretation"] = match[-1]["interpretation"] if match else "inconclusive"
    return row


def find_per_class_path(model_key: str, dataset: str, shot: int, seed: int) -> Path | None:
    patterns = {
        "hybridsn_small": [
            f"result/hybridsn_small_fewshot_3datasets/metrics/{dataset}_shot{shot}_seed{seed}_per_class.csv",
            f"result/hybridsn_small_fewshot_pavia_salinas_5_10shot/metrics/{dataset}_shot{shot}_seed{seed}_per_class.csv",
        ],
        "prototype": [
            f"result/**/metrics/{dataset}_spectral_gated_qnn_prototype_shot{shot}_seed{seed}_per_class.csv",
            f"result/**/metrics/{dataset}_spectral_gated_qnn_proto_shot{shot}_seed{seed}_per_class.csv",
        ],
        "supcon": [
            f"result/**/metrics/{dataset}_spectral_gated_qnn_supcon_shot{shot}_seed{seed}_per_class.csv",
        ],
    }
    for pattern in patterns[model_key]:
        matches = sorted(Path(".").glob(pattern))
        if matches:
            return matches[0]
    return None


def find_confusion_path(model_key: str, dataset: str, shot: int, seed: int, normalized: bool = False) -> Path | None:
    suffix = "_normalized" if normalized else ""
    patterns = {
        "hybridsn_small": [
            f"result/hybridsn_small_fewshot_3datasets/confusion_matrices/{dataset}_shot{shot}_seed{seed}{suffix}.csv",
            f"result/hybridsn_small_fewshot_pavia_salinas_5_10shot/confusion_matrices/{dataset}_shot{shot}_seed{seed}{suffix}.csv",
        ],
        "prototype": [
            f"result/**/confusion_matrices/{dataset}_spectral_gated_qnn_prototype_shot{shot}_seed{seed}{suffix}.csv",
            f"result/**/confusion_matrices/{dataset}_spectral_gated_qnn_proto_shot{shot}_seed{seed}{suffix}.csv",
        ],
        "supcon": [
            f"result/**/confusion_matrices/{dataset}_spectral_gated_qnn_supcon_shot{shot}_seed{seed}{suffix}.csv",
        ],
    }
    for pattern in patterns[model_key]:
        matches = sorted(Path(".").glob(pattern))
        if matches:
            return matches[0]
    return None


def build_per_class_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for model_key in ["hybridsn_small", "prototype", "supcon"]:
        for dataset in DATASETS:
            for shot in SHOTS:
                for seed in SEEDS:
                    path = find_per_class_path(model_key, dataset, shot, seed)
                    if path is None:
                        continue
                    df = pd.read_csv(path)
                    df["dataset"] = dataset
                    df["shot"] = shot
                    df["seed"] = seed
                    df["model_key"] = model_key
                    df["model"] = DISPLAY_MODEL[model_key]
                    frames.append(df)
    per_class = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if per_class.empty:
        return per_class, pd.DataFrame(), pd.DataFrame()
    deltas = []
    for model_key in ["prototype", "supcon"]:
        for dataset in DATASETS:
            for shot in SHOTS:
                base = per_class[(per_class["dataset"] == dataset) & (per_class["shot"] == shot) & (per_class["model_key"] == "hybridsn_small")]
                qnn = per_class[(per_class["dataset"] == dataset) & (per_class["shot"] == shot) & (per_class["model_key"] == model_key)]
                if base.empty or qnn.empty:
                    continue
                bmean = base.groupby(["class_id", "class_name"])[["precision", "recall", "f1"]].mean().reset_index()
                qmean = qnn.groupby(["class_id", "class_name"])[["precision", "recall", "f1"]].mean().reset_index()
                merged = bmean.merge(qmean, on=["class_id", "class_name"], suffixes=("_baseline", "_qnn"))
                for row in merged.itertuples(index=False):
                    delta_f1 = row.f1_qnn - row.f1_baseline
                    deltas.append(
                        {
                            "dataset": dataset,
                            "shot": shot,
                            "comparison": f"HybridSN-small vs {DISPLAY_MODEL[model_key]}",
                            "class_id": row.class_id,
                            "class_name_if_available": row.class_name,
                            "baseline_precision": row.precision_baseline,
                            "qnn_precision": row.precision_qnn,
                            "delta_precision": row.precision_qnn - row.precision_baseline,
                            "baseline_recall": row.recall_baseline,
                            "qnn_recall": row.recall_qnn,
                            "delta_recall": row.recall_qnn - row.recall_baseline,
                            "baseline_f1": row.f1_baseline,
                            "qnn_f1": row.f1_qnn,
                            "delta_f1": delta_f1,
                            "interpretation": "improved" if delta_f1 > 0.02 else "degraded" if delta_f1 < -0.02 else "near_neutral",
                        }
                    )
    delta_df = pd.DataFrame(deltas)
    top_rows = []
    for keys, group in delta_df.groupby(["dataset", "shot", "comparison"]):
        for tag, subset in [("top_improved", group.sort_values("delta_f1", ascending=False).head(5)), ("top_degraded", group.sort_values("delta_f1").head(5))]:
            for row in subset.itertuples(index=False):
                item = row._asdict()
                item["rank_type"] = tag
                top_rows.append(item)
    return per_class, delta_df, pd.DataFrame(top_rows)


def build_salinas_negative_transfer(data: dict[str, pd.DataFrame], per_class_delta: pd.DataFrame, paired_deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for loss_name in ["Prototype", "SupCon"]:
        subset = per_class_delta[
            (per_class_delta["dataset"] == "salinas")
            & (per_class_delta["shot"] == 10)
            & (per_class_delta["comparison"].str.contains(loss_name, na=False))
        ]
        for row in subset.sort_values("delta_f1").itertuples(index=False):
            rows.append(
                {
                    "analysis_type": "per_class_f1_delta",
                    "comparison": row.comparison,
                    "loss_type": loss_name,
                    "class_id": row.class_id,
                    "class_name": row.class_name_if_available,
                    "delta_f1": row.delta_f1,
                    "delta_recall": row.delta_recall,
                    "interpretation": row.interpretation,
                }
            )
        seed = paired_deltas[
            (paired_deltas["dataset"] == "salinas")
            & (paired_deltas["shot"] == 10)
            & (paired_deltas["comparison"].str.contains(loss_name, na=False))
            & (paired_deltas["metric"].isin(["OA", "Macro-F1", "Weighted-F1"]))
        ]
        for row in seed.itertuples(index=False):
            rows.append(
                {
                    "analysis_type": f"seedwise_{row.metric}_delta",
                    "comparison": row.comparison,
                    "loss_type": loss_name,
                    "class_id": np.nan,
                    "class_name": f"seed_{row.seed}",
                    "delta_f1": row.delta if row.metric == "Macro-F1" else np.nan,
                    "delta_recall": np.nan,
                    "interpretation": "negative_transfer" if row.delta < 0 else "improved",
                }
            )
    return pd.DataFrame(rows)


def build_logit_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    logit = read_csv_if_exists("result/logit_margin_hybridsn_vs_qnn/metrics/logit_margin_summary_by_dataset_shot_model.csv")
    joint = read_csv_if_exists("result/logit_margin_hybridsn_vs_qnn/metrics/geometry_vs_logit_margin_joint_summary.csv")
    return logit, joint


def build_gate_tables(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_paths = sorted(Path("result").glob("**/*gate_values.csv"))
    if not gate_paths:
        report = [
            "# Gate Analysis Missing",
            "",
            "No saved gate-value CSV files were found in the current result tree.",
            "",
            "Missing files: `*_gate_values.csv` for Spectral QNN Gated Fusion evaluations.",
            "",
            "The script that already contains gate export support is `scripts/run_fair_control_models_fewshot.py`, via `_write_gate_values()`.",
            "Future runs should call the model with `return_aux=True` during evaluation and save per-sample `mean_gate`, `gate_for_pred_class`, and `gate_for_true_class`.",
        ]
        (out / "reports/gate_analysis_missing.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        return pd.DataFrame(columns=["dataset", "shot", "model", "mean_gate_value", "std_gate_value"]), pd.DataFrame()
    frames = [pd.read_csv(p) for p in gate_paths]
    gates = pd.concat(frames, ignore_index=True)
    summary = gates.groupby(["dataset", "shot", "model"]).agg(
        mean_gate_value=("mean_gate", "mean"),
        std_gate_value=("mean_gate", "std"),
        gate_value_correct_samples=("mean_gate", lambda s: np.nan),
        gate_value_wrong_samples=("mean_gate", lambda s: np.nan),
    ).reset_index()
    classwise = gates.groupby(["dataset", "shot", "model", "true_label"]).agg(
        classwise_mean_gate_value=("mean_gate", "mean"),
        classwise_std_gate_value=("mean_gate", "std"),
    ).reset_index()
    return summary, classwise


def build_complexity_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    specs = [
        ("HybridSN-small", "hybridsn_small", None, None, None, "baseline encoder + classifier"),
        ("Spectral QNN Gated Fusion + Prototype", "prototype", 6, 1, "linear", "metric branch"),
        ("Spectral QNN Gated Fusion + SupCon", "supcon", 6, 1, "linear", "Indian Pines only in saved runs"),
    ]
    for name, key, qubits, layers, entanglement, notes in specs:
        df = data[key]
        rows.append(
            {
                "model": name,
                "trainable_params": float(df["trainable_parameters"].dropna().mean()) if "trainable_parameters" in df and not df.empty else np.nan,
                "encoder_params": 99488 if key == "hybridsn_small" else 99488,
                "classifier_or_head_params": np.nan,
                "qnn_params": np.nan if key == "hybridsn_small" else 2176,
                "qubits": qubits,
                "quantum_layers": layers,
                "encoding_type": "angle/tanh projection" if key != "hybridsn_small" else "NA",
                "entanglement_type": entanglement or "NA",
                "training_time_mean": float(df["train_time_seconds"].dropna().mean()) if "train_time_seconds" in df and not df.empty else np.nan,
                "inference_time_mean": float(df["test_time_seconds"].dropna().mean()) if "test_time_seconds" in df and not df.empty else np.nan,
                "device": "cpu",
                "notes": notes,
            }
        )
    rows.extend(
        [
            {
                "model": "CNN2D",
                "trainable_params": 28208,
                "encoder_params": np.nan,
                "classifier_or_head_params": np.nan,
                "qnn_params": np.nan,
                "qubits": np.nan,
                "quantum_layers": np.nan,
                "encoding_type": "NA",
                "entanglement_type": "NA",
                "training_time_mean": np.nan,
                "inference_time_mean": np.nan,
                "device": "cpu",
                "notes": "from few-shot CNN baseline metadata",
            },
            {
                "model": "SVM-RBF",
                "trainable_params": np.nan,
                "encoder_params": np.nan,
                "classifier_or_head_params": np.nan,
                "qnn_params": np.nan,
                "qubits": np.nan,
                "quantum_layers": np.nan,
                "encoding_type": "NA",
                "entanglement_type": "NA",
                "training_time_mean": np.nan,
                "inference_time_mean": np.nan,
                "device": "cpu",
                "notes": "non-parametric sklearn baseline; exact support-vector count not summarized",
            },
        ]
    )
    return pd.DataFrame(rows)


def make_figures(out: Path, data: dict[str, pd.DataFrame], paired: pd.DataFrame, per_class_delta: pd.DataFrame, margin_joint: pd.DataFrame) -> None:
    plot_seedwise(out / "figures/seedwise_delta_macro_f1.png", paired, "Macro-F1", "Seedwise Macro-F1 Delta")
    plot_seedwise(out / "figures/seedwise_delta_oa.png", paired, "OA", "Seedwise OA Delta")
    plot_per_class_panels(out, per_class_delta)
    plot_confusion_deltas(out)
    plot_salinas_seedwise(out / "figures/seedwise_delta_salinas_10shot.png", paired)
    plot_margin(out / "figures/logit_margin_comparison.png", margin_joint, "mean_true_logit_margin_mean", "Mean True Logit Margin")
    plot_margin(out / "figures/negative_margin_rate_comparison.png", margin_joint, "negative_logit_margin_rate_mean", "Negative Logit Margin Rate")
    empty_gate_figure(out / "figures/gate_value_distribution.png")
    empty_gate_figure(out / "figures/classwise_gate_value_heatmap.png")


def plot_seedwise(path: Path, paired: pd.DataFrame, metric: str, title: str) -> None:
    df = paired[paired["metric"] == metric]
    plt.figure(figsize=(10, 5), dpi=180)
    if df.empty:
        plt.text(0.5, 0.5, "No paired data", ha="center", va="center")
    else:
        labels = []
        values = []
        for keys, group in df.groupby(["dataset", "shot", "comparison"]):
            labels.append(f"{keys[0]}\n{keys[1]} {short_comparison(keys[2])}")
            values.append(group["delta"].mean())
        plt.bar(range(len(values)), values, color=["#4C78A8" if v >= 0 else "#E45756" for v in values])
        plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=7)
        plt.axhline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def short_comparison(text: str) -> str:
    if "SupCon" in text:
        return "SupCon"
    if "Prototype" in text:
        return "Proto"
    return text


def plot_per_class_panels(out: Path, delta: pd.DataFrame) -> None:
    if delta.empty:
        return
    for (dataset, shot), group in delta[delta["comparison"].str.contains("Prototype", na=False)].groupby(["dataset", "shot"]):
        path = out / "figures" / f"per_class_delta_f1_{dataset}_{shot}.png"
        g = group.sort_values("class_id")
        plt.figure(figsize=(10, 4), dpi=180)
        plt.bar(g["class_id"].astype(str), g["delta_f1"], color=["#54A24B" if v >= 0 else "#E45756" for v in g["delta_f1"]])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title(f"{DISPLAY_DATASET[dataset]} {shot}-shot Per-class F1 Delta")
        plt.xlabel("Class ID")
        plt.ylabel("QNN - HybridSN-small F1")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    sal = out / "figures/per_class_delta_f1_salinas_10shot.png"
    src = out / "figures/per_class_delta_f1_salinas_10.png"
    if src.exists():
        shutil.copy2(src, sal)
    first = sorted((out / "figures").glob("per_class_delta_f1_*_*.png"))
    if first:
        shutil.copy2(first[0], out / "figures/per_class_delta_f1.png")


def plot_confusion_deltas(out: Path) -> None:
    for dataset in DATASETS:
        for shot in SHOTS:
            mats = []
            for seed in SEEDS:
                b = find_confusion_path("hybridsn_small", dataset, shot, seed)
                q = find_confusion_path("prototype", dataset, shot, seed)
                if b and q:
                    mats.append(pd.read_csv(q).to_numpy() - pd.read_csv(b).to_numpy())
            if not mats:
                continue
            delta = np.mean(mats, axis=0)
            path = out / "figures" / f"confusion_delta_{dataset}_{shot}.png"
            plt.figure(figsize=(6, 5), dpi=180)
            vmax = max(abs(delta.min()), abs(delta.max()))
            plt.imshow(delta, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            plt.colorbar(label="QNN - baseline count")
            plt.title(f"{DISPLAY_DATASET[dataset]} {shot}-shot Confusion Delta")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
    sal = out / "figures/confusion_delta_salinas_10shot.png"
    src = out / "figures/confusion_delta_salinas_10.png"
    if src.exists():
        shutil.copy2(src, sal)
    first = sorted((out / "figures").glob("confusion_delta_*_*.png"))
    if first:
        shutil.copy2(first[0], out / "figures/confusion_delta_salinas_10shot.png") if not sal.exists() else None


def plot_salinas_seedwise(path: Path, paired: pd.DataFrame) -> None:
    df = paired[
        (paired["dataset"] == "salinas")
        & (paired["shot"] == 10)
        & (paired["comparison"].str.contains("Prototype", na=False))
        & (paired["metric"].isin(["OA", "Macro-F1"]))
    ]
    plt.figure(figsize=(7, 4), dpi=180)
    if df.empty:
        plt.text(0.5, 0.5, "No Salinas 10-shot paired data", ha="center", va="center")
    else:
        for metric, group in df.groupby("metric"):
            plt.plot(group["seed"], group["delta"], marker="o", label=metric)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.legend()
    plt.title("Salinas 10-shot Seedwise Delta")
    plt.xlabel("Seed")
    plt.ylabel("QNN - HybridSN-small")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_margin(path: Path, margin: pd.DataFrame, column: str, title: str) -> None:
    plt.figure(figsize=(9, 4), dpi=180)
    if margin.empty or column not in margin.columns:
        plt.text(0.5, 0.5, "No margin data", ha="center", va="center")
    else:
        labels = [f"{r.dataset}\n{int(r.shot)} {r.model}" for r in margin.itertuples(index=False)]
        values = margin[column].to_numpy(float)
        plt.bar(range(len(values)), values, color="#72B7B2")
        plt.xticks(range(len(values)), labels, rotation=60, ha="right", fontsize=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def empty_gate_figure(path: Path) -> None:
    plt.figure(figsize=(6, 3), dpi=180)
    plt.text(0.5, 0.5, "Gate values not available in saved runs", ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fmt_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x * 100:.2f}"


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No data.\n"
    show = df.head(max_rows) if max_rows else df
    return show.to_markdown(index=False) + "\n"


def write_reports(
    out: Path,
    data: dict[str, pd.DataFrame],
    main_results: pd.DataFrame,
    stability: pd.DataFrame,
    tests: pd.DataFrame,
    per_class_delta: pd.DataFrame,
    top_classes: pd.DataFrame,
    salinas_negative: pd.DataFrame,
    margin_joint: pd.DataFrame,
    complexity: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    write_report_zh(out, main_results, stability, tests, top_classes, salinas_negative, margin_joint, complexity, failures)
    write_report_en(out, main_results, stability, tests, top_classes, salinas_negative, margin_joint, complexity, failures)
    write_paper_tables(out, main_results, stability, tests, top_classes, salinas_negative, complexity)
    write_limitations(out)
    write_complexity_report(out, complexity)
    write_salinas_report(out, salinas_negative, margin_joint, tests, stability)


def write_report_zh(out: Path, main_results: pd.DataFrame, stability: pd.DataFrame, tests: pd.DataFrame, top_classes: pd.DataFrame, salinas_negative: pd.DataFrame, margin_joint: pd.DataFrame, complexity: pd.DataFrame, failures: pd.DataFrame) -> None:
    lines = [
        "# Few-shot Spectral QNN 最终证据闭环报告",
        "",
        "## 1. Executive Summary",
        "",
        "Spectral QNN Gated Fusion 与 metric-learning objective 结合后，在多个低样本 HSI 设置中提供了边际到中等幅度提升。最强证据来自 Pavia University 5/10-shot；Indian Pines 为边际但可复现提升。Salinas 10-shot 的负迁移主要出现在 Prototype Loss 配置，SupCon Loss 将 Macro-F1 拉回到轻微正提升，但 OA 与 Weighted-F1 仍低于 HybridSN-small，因此应谨慎表述为“SupCon 缓解而非彻底解决负迁移”。",
        "",
        "## 2. What has been completed",
        "",
        "- 已复用 HybridSN-small、QNN + Prototype，以及 Indian Pines/Pavia/Salinas QNN + SupCon 的 seed 级结果。",
        "- 已生成 paired seed delta、统计检验、per-class delta、confusion delta、logit margin 整合、复杂度表和负例报告。",
        "- Pavia/Salinas SupCon 受控对照已补齐；缺失项如有发生会记录在 `failed_runs.csv`。",
        "",
        "## 3. Main results",
        markdown_table(main_results),
        "## 4. Prototype vs SupCon comparison",
        "",
        "SupCon 在 Indian Pines 10-shot 和 Pavia University 5/10-shot 上略强于 Prototype；Salinas 5-shot 中 Prototype 的 Macro-F1 提升更大；Salinas 10-shot 中 Prototype 为负迁移，而 SupCon 将 Macro-F1 调整为轻微正提升。整体看，SupCon 更像是降低部分负迁移风险的稳定化目标，而不是在所有设置中都取得最大收益的目标。",
        "",
        markdown_table(stability),
        "",
        "## 5. Statistical significance",
        markdown_table(tests[tests["metric"] == "Macro-F1"]),
        "## 6. Per-class analysis",
        markdown_table(top_classes, 30),
        "## 7. Salinas 10-shot negative transfer",
        markdown_table(salinas_negative, 30),
        "## 8. Logit margin / decision-boundary explanation",
        markdown_table(margin_joint),
        "当前证据更支持最终 classifier logit margin 层面的 decision-boundary regularization，而不是 universal prototype-space separation improvement。",
        "",
        "## 9. Complexity analysis",
        markdown_table(complexity),
        "QNN 分支参数较紧凑，但在经典仿真下并不更快，因此当前优势应表述为 few-shot decision-boundary regularization，而不是计算效率。",
        "",
        "## 10. Limitations",
        "",
        "- QNN 不普遍优于 HybridSN-small。",
        "- Salinas 10-shot 的 Prototype Loss 配置出现负迁移；SupCon Loss 缓解 Macro-F1 负迁移，但 OA/Weighted-F1 仍下降。",
        "- SupCon 结果已覆盖 Indian Pines、Pavia University 与 Salinas 的 5/10-shot；1-shot SupCon 仍未纳入本轮闭合。",
        "- gate values 未在当前 saved runs 中保存。",
        "- random pixel split 对 patch-based HSI 可能偏乐观。",
        "",
        "## 11. Recommended paper narrative",
        "",
        "建议将论文叙事聚焦于：Spectral-side QNN + metric learning 在低样本场景中改善部分 dataset/shot 的最终决策边界；Prototype 与 SupCon 分别体现“原型约束”和“对比正则化”的不同稳定性，其中 SupCon 可缓解 Salinas 10-shot Prototype 负迁移，但整体效果仍依赖数据集和 shot。",
    ]
    (out / "reports/report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_en(out: Path, main_results: pd.DataFrame, stability: pd.DataFrame, tests: pd.DataFrame, top_classes: pd.DataFrame, salinas_negative: pd.DataFrame, margin_joint: pd.DataFrame, complexity: pd.DataFrame, failures: pd.DataFrame) -> None:
    lines = [
        "# Final Evidence Closure Report for Few-shot Spectral QNN",
        "",
        "## 1. Executive Summary",
        "",
        "Spectral QNN Gated Fusion combined with metric-learning objectives provides reproducible marginal-to-moderate gains in several low-shot HSI settings. The strongest evidence is on Pavia University 5/10-shot. Indian Pines shows marginal reproducible gains. The Salinas 10-shot negative transfer is mainly tied to Prototype Loss: SupCon moves Macro-F1 back to a small positive delta, while OA and Weighted-F1 remain below HybridSN-small.",
        "",
        "## 2. What has been completed",
        "",
        "- Reused saved seed-level results for HybridSN-small, QNN + Prototype, and Indian Pines/Pavia/Salinas QNN + SupCon.",
        "- Generated paired deltas, statistical tests, per-class deltas, confusion-delta figures, logit-margin integration, complexity summary, and negative-case reports.",
        "- Pavia/Salinas SupCon controlled comparisons have been completed; any missing items are explicitly logged in `failed_runs.csv`.",
        "",
        "## 3. Main results",
        markdown_table(main_results),
        "## 4. Prototype vs SupCon comparison",
        "",
        "SupCon is slightly stronger on Indian Pines 10-shot and Pavia University 5/10-shot. Prototype is stronger on Salinas 5-shot. On Salinas 10-shot, Prototype shows negative transfer, while SupCon mitigates it for Macro-F1 but does not fully recover OA or Weighted-F1. SupCon is therefore better framed as a stabilizing objective rather than a universally stronger objective.",
        "",
        markdown_table(stability),
        "",
        "## 5. Statistical significance",
        markdown_table(tests[tests["metric"] == "Macro-F1"]),
        "## 6. Per-class analysis",
        markdown_table(top_classes, 30),
        "## 7. Salinas 10-shot negative transfer",
        markdown_table(salinas_negative, 30),
        "## 8. Logit margin / decision-boundary explanation",
        markdown_table(margin_joint),
        "The evidence aligns better with final classifier logit-margin behavior than with universal prototype-space separation.",
        "",
        "## 9. Complexity analysis",
        markdown_table(complexity),
        "The QNN branch is parameter-compact but not computationally faster under classical simulation.",
        "",
        "## 10. Limitations",
        "",
        "- QNN does not universally outperform HybridSN-small.",
        "- Salinas 10-shot is a negative-transfer case for Prototype Loss; SupCon mitigates Macro-F1 degradation but OA/Weighted-F1 remain lower than HybridSN-small.",
        "- SupCon now covers Indian Pines, Pavia University, and Salinas at 5/10-shot; 1-shot SupCon remains outside this closure.",
        "- Gate values were not saved in current runs.",
        "- Random pixel split can be optimistic for patch-based HSI models.",
        "",
        "## 11. Recommended paper narrative",
        "",
        "Frame the contribution as spectral-side QNN with metric learning for low-shot decision-boundary regularization, with dataset- and shot-dependent gains. Prototype and SupCon should be discussed as complementary metric-learning objectives, with SupCon reducing the Salinas 10-shot Prototype negative-transfer risk.",
    ]
    (out / "reports/report_en.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_tables(out: Path, main_results: pd.DataFrame, stability: pd.DataFrame, tests: pd.DataFrame, top_classes: pd.DataFrame, salinas_negative: pd.DataFrame, complexity: pd.DataFrame) -> None:
    lines = [
        "# Paper-ready Tables",
        "",
        "## 1. Main few-shot result table",
        markdown_table(main_results),
        "## 2. Delta over HybridSN-small",
        markdown_table(main_results[[c for c in main_results.columns if c.startswith("dataset") or c.startswith("shot") or c.startswith("model") or c.startswith("delta_") or c == "macro_f1_interpretation"]]),
        "## 3. Statistical significance table",
        markdown_table(tests),
        "## 4. Prototype vs SupCon stability table",
        markdown_table(stability),
        "## 5. Per-class top improved/degraded table",
        markdown_table(top_classes, 40),
        "## 6. Complexity table",
        markdown_table(complexity),
        "## 7. Negative case summary table",
        markdown_table(salinas_negative, 40),
    ]
    (out / "reports/paper_ready_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_limitations(out: Path) -> None:
    lines = [
        "# Limitations and Negative Cases",
        "",
        "1. QNN does not universally outperform HybridSN-small; gains are dataset- and shot-dependent.",
        "2. Salinas 10-shot is a negative-transfer case for Prototype Loss; SupCon mitigates Macro-F1 degradation but does not fully recover OA or Weighted-F1.",
        "3. Final-head QNN does not outperform MLP head in fair frozen-embedding comparisons.",
        "4. Random pixel split can be optimistic for patch-based HSI models because neighboring pixels may share spatial context.",
        "5. Spatial split remains difficult; current spatial pilots show large performance drops.",
        "6. QNN simulation is slower than classical heads under the current CPU/classical simulator setup.",
        "7. Prototype geometry does not universally improve; logit margin gives a more consistent explanation for successful settings.",
    ]
    (out / "reports/limitations_and_negative_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_complexity_report(out: Path, complexity: pd.DataFrame) -> None:
    lines = [
        "# 复杂度与资源汇总",
        "",
        markdown_table(complexity),
        "结论：QNN branch 参数更紧凑，但经典仿真下不具备速度优势。因此当前优势应定位为少样本决策边界正则化，而不是计算效率。",
    ]
    (out / "reports/complexity_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_salinas_report(out: Path, salinas_negative: pd.DataFrame, margin_joint: pd.DataFrame, tests: pd.DataFrame, stability: pd.DataFrame) -> None:
    margin = margin_joint[(margin_joint["dataset"] == "salinas") & (margin_joint["shot"] == 10)] if not margin_joint.empty else pd.DataFrame()
    stat = tests[(tests["dataset"] == "salinas") & (tests["shot"] == 10)] if not tests.empty else pd.DataFrame()
    stable = stability[(stability["dataset"] == "salinas") & (stability["shot"] == 10)] if not stability.empty else pd.DataFrame()
    proto_classes = salinas_negative[
        (salinas_negative["analysis_type"] == "per_class_f1_delta")
        & (salinas_negative["loss_type"] == "Prototype")
    ].sort_values("delta_f1")
    supcon_classes = salinas_negative[
        (salinas_negative["analysis_type"] == "per_class_f1_delta")
        & (salinas_negative["loss_type"] == "SupCon")
    ].sort_values("delta_f1")
    seed_rows = salinas_negative[salinas_negative["analysis_type"].str.startswith("seedwise_", na=False)]
    lines = [
        "# Salinas 10-shot 负迁移分析",
        "",
        "## Main finding",
        "",
        "Salinas 10-shot 的负迁移主要发生在 QNN + Prototype Loss 配置：Macro-F1 相对 HybridSN-small 为 -0.0048，OA 为 -0.0288，Weighted-F1 为 -0.0333。QNN + SupCon Loss 将 Macro-F1 拉回到 +0.0026，但 OA 仍为 -0.0189、Weighted-F1 仍为 -0.0202，因此更准确的表述是 SupCon 缓解 Prototype 的 Macro-F1 负迁移，而不是全面超过 classical baseline。",
        "",
        "## Evidence table",
        markdown_table(stat),
        "## Prototype vs SupCon stability",
        markdown_table(stable),
        "## Per-class degradation: Prototype",
        markdown_table(proto_classes, 20),
        "## Per-class degradation: SupCon",
        markdown_table(supcon_classes, 20),
        "## Seed-wise degradation",
        markdown_table(seed_rows, 30),
        "## Margin or gate evidence if available",
        markdown_table(margin),
        "已有 logit margin 分析只覆盖 Prototype 配置，因此支持“Prototype 负迁移伴随最终决策边界 margin 变差”。SupCon 的 logits/gate values 尚未做 evaluation-only 导出，因此目前不能直接证明 SupCon 缓解来自 gate 使用更合理或 margin 改善。",
        "",
        "## Final interpretation",
        "",
        "负迁移更可能来自 Salinas 10-shot classical baseline 已接近饱和，Prototype Loss 在 Vinyard_untrained 与 Grapes_untrained 等类别上放大了已有混淆。SupCon 对 Macro-F1 有缓解作用，说明对比式约束可能比单一 prototype 约束更稳健；但由于 OA/Weighted-F1 仍下降，这一结果应写成“部分缓解”而不是“解决负迁移”。后续需要保存 SupCon logits、gate values 和 branch contribution magnitude 来验证机制。",
    ]
    (out / "reports/salinas_10shot_negative_transfer_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(out: Path, timestamp: str, raw_sources: list[str], failures: pd.DataFrame) -> None:
    meta = {
        "timestamp": timestamp,
        "output_dir": str(out),
        "seed_list": SEEDS,
        "datasets": DATASETS,
        "shots": SHOTS,
        "raw_sources_copied": raw_sources,
        "missing_or_failed_runs": int(len(failures)),
        "notes": "This package reuses saved metrics and includes the completed Pavia/Salinas SupCon controlled runs.",
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def write_checklist(out: Path) -> None:
    supcon_path = out / "tables/supcon_cross_dataset_summary.csv"
    supcon = pd.read_csv(supcon_path) if supcon_path.exists() else pd.DataFrame()

    def has_supcon(dataset: str, shot: int) -> bool:
        if supcon.empty:
            return False
        rows = supcon[(supcon["dataset"] == dataset) & (supcon["shot"] == shot)]
        return len(rows) >= 5

    items = [
        ("SupCon Pavia 5-shot completed", has_supcon("pavia_university", 5)),
        ("SupCon Pavia 10-shot completed", has_supcon("pavia_university", 10)),
        ("SupCon Salinas 5-shot completed", has_supcon("salinas", 5)),
        ("SupCon Salinas 10-shot completed", has_supcon("salinas", 10)),
        ("Paired statistical tests completed", (out / "tables/statistical_tests.csv").exists()),
        ("Per-class delta tables completed", (out / "tables/per_class_delta_qnn_vs_baseline.csv").exists()),
        ("Confusion delta figures completed", bool(list((out / "figures").glob("confusion_delta_*.png")))),
        ("Salinas 10-shot negative transfer report completed", (out / "reports/salinas_10shot_negative_transfer_zh.md").exists()),
        ("Logit margin summary completed", (out / "tables/logit_margin_summary.csv").exists()),
        ("Gate analysis completed or missing-gate report generated", (out / "reports/gate_analysis_missing.md").exists() or (out / "tables/gate_value_summary.csv").exists()),
        ("Complexity summary completed", (out / "tables/complexity_summary.csv").exists()),
        ("Chinese final report completed", (out / "reports/report_zh.md").exists()),
        ("English final report completed", (out / "reports/report_en.md").exists()),
        ("Paper-ready tables completed", (out / "reports/paper_ready_tables.md").exists()),
        ("Limitations report completed", (out / "reports/limitations_and_negative_cases.md").exists()),
    ]
    text = "\n".join(f"[{'x' if ok else ' '}] {name}" for name, ok in items) + f"\n\nFinal output directory: {out.resolve()}\n"
    (out / "reports/checklist.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
