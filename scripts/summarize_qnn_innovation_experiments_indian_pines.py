from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("result/qnn_innovation_experiments_indian_pines")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    spectral = pd.read_csv("result/hybridsn_spectral_qnn_branch_indian_pines/summary_mean_std.csv")
    spectral_tests = pd.read_csv("result/hybridsn_spectral_qnn_branch_indian_pines/per_model_test_metrics.csv")
    param = pd.read_csv("result/parameter_efficiency_hybridsn_heads_indian_pines/summary_mean_std.csv")
    spatial_qnn = pd.read_csv("result/spatial_split_qnn_heads_indian_pines/all_runs.csv")

    spectral_out = spectral[
        [
            "model",
            "runs",
            "parameters_mean",
            "time_seconds_mean",
            "best_val_oa_mean",
            "best_val_aa_mean",
            "best_val_macro_f1_mean",
        ]
    ].copy()
    spectral_out.insert(0, "experiment", "spectral_branch_random_split")
    spectral_out = _percent_cols(spectral_out, ["best_val_oa_mean", "best_val_aa_mean", "best_val_macro_f1_mean"])

    param_out = param[
        [
            "model",
            "runs",
            "params_mean",
            "training_time_seconds_mean",
            "best_val_oa_mean",
            "best_val_aa_mean",
            "best_val_macro_f1_mean",
        ]
    ].copy()
    param_out = param_out.rename(columns={"params_mean": "parameters_mean", "training_time_seconds_mean": "time_seconds_mean"})
    param_out.insert(0, "experiment", "parameter_matched_random_split")
    param_out = _percent_cols(param_out, ["best_val_oa_mean", "best_val_aa_mean", "best_val_macro_f1_mean"])

    spatial_out = spatial_qnn.rename(
        columns={
            "best_val_oa": "best_val_oa_mean",
            "best_val_aa": "best_val_aa_mean",
            "best_val_macro_f1": "best_val_macro_f1_mean",
            "training_time_seconds": "time_seconds_mean",
            "run_id": "model",
        }
    )
    spatial_out["runs"] = 1
    spatial_out["parameters_mean"] = None
    spatial_out = spatial_out[
        [
            "experiment",
            "model",
            "runs",
            "parameters_mean",
            "time_seconds_mean",
            "best_val_oa_mean",
            "best_val_aa_mean",
            "best_val_macro_f1_mean",
        ]
    ] if "experiment" in spatial_out.columns else spatial_out[
        [
            "model",
            "runs",
            "parameters_mean",
            "time_seconds_mean",
            "best_val_oa_mean",
            "best_val_aa_mean",
            "best_val_macro_f1_mean",
        ]
    ]
    if "experiment" not in spatial_out.columns:
        spatial_out.insert(0, "experiment", "spatial_split_head_comparison")
    spatial_out = _percent_cols(spatial_out, ["best_val_oa_mean", "best_val_aa_mean", "best_val_macro_f1_mean"])

    all_tables = pd.concat([spectral_out, param_out, spatial_out], ignore_index=True)
    all_tables.to_csv(ROOT / "summary_tables.csv", index=False)

    best_metrics = {
        "spectral_branch": _load_json("result/hybridsn_spectral_qnn_branch_indian_pines/best_metrics.json"),
        "parameter_efficiency": _load_json("result/parameter_efficiency_hybridsn_heads_indian_pines/best_metrics.json"),
        "spatial_qnn_heads": _load_json("result/spatial_split_qnn_heads_indian_pines/best_metrics.json"),
        "spatial_hybridsn": _load_json("result/spatial_split_hybridsn_indian_pines/best_metrics.json"),
        "tuned_hybridsn": _load_json("result/hybridsn_tuning_indian_pines/best_metrics.json"),
    }
    with (ROOT / "best_metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)
    _write_report(ROOT / "innovation_experiments_summary.md", spectral_out, spectral_tests, param_out, spatial_out, best_metrics)


def _percent_cols(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = (out[column] * 100).round(2)
    out["time_seconds_mean"] = out["time_seconds_mean"].astype(float).round(2)
    return out


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_row(name: str, metrics: dict) -> str:
    return (
        f"| {name} | {metrics['OA']*100:.2f} | {metrics['AA']*100:.2f} | "
        f"{metrics['Kappa']*100:.2f} | {metrics['Macro-F1']*100:.2f} | {metrics['Weighted-F1']*100:.2f} |"
    )


def _write_report(
    path: Path,
    spectral: pd.DataFrame,
    spectral_tests: pd.DataFrame,
    param: pd.DataFrame,
    spatial: pd.DataFrame,
    metrics: dict,
) -> None:
    spectral_tests_display = spectral_tests.copy()
    for col in ("best_val_macro_f1", "best_val_oa", "best_val_aa", "OA", "AA", "Kappa", "Macro-F1", "Weighted-F1"):
        spectral_tests_display[col] = (spectral_tests_display[col] * 100).round(2)
    lines = [
        "# QNN Innovation Experiments: Indian Pines",
        "",
        "This summary consolidates three targeted experiments: moving QNN into a center spectral branch, evaluating heads under spatially isolated split, and comparing QNN heads under a parameter-efficiency protocol.",
        "",
        "## Experiment 1: HybridSN + Spectral Branch",
        "",
        spectral.to_markdown(index=False),
        "",
        "### Per-Model Test Metrics",
        "",
        spectral_tests_display.to_markdown(index=False),
        "",
        "## Experiment 2: Spatial Split Head Comparison",
        "",
        spatial.to_markdown(index=False),
        "",
        "## Experiment 3: Parameter Efficiency",
        "",
        param.to_markdown(index=False),
        "",
        "## Selected Test Metrics",
        "",
        "| Model / Protocol | OA | AA | Kappa | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|",
        _metric_row("Tuned HybridSN random split", metrics["tuned_hybridsn"]),
        _metric_row("Best spectral branch random split", metrics["spectral_branch"]),
        _metric_row("Spectral QNN branch random split", spectral_tests[spectral_tests["model"] == "spectral_qnn_fusion"].iloc[0].to_dict()),
        _metric_row("Best parameter-efficiency head", metrics["parameter_efficiency"]),
        _metric_row("HybridSN spatial split", metrics["spatial_hybridsn"]),
        _metric_row("Best spatial-split head", metrics["spatial_qnn_heads"]),
        "",
        "## Conclusion",
        "",
        "The spectral QNN branch is the most promising QNN placement so far: when selected within its own model family by validation Macro-F1, it gives the best test Macro-F1 in the spectral-branch pilot. However, its validation mean is still below the embedding MLP probe, so this should be treated as a positive signal that needs more seeds and a stricter confirmation run, not yet as a settled superiority claim. Under spatial split, QNN heads still drop strongly and do not beat the classical MLP head. The parameter-efficiency experiment shows that QNN heads can approach MLP performance with fewer trainable parameters, but they are much slower.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
