from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ("OA", "AA", "Kappa", "Macro-F1", "Weighted-F1")
DATASET_LABELS = {
    "indian_pines": "Indian Pines",
    "pavia_university": "Pavia University",
    "salinas": "Salinas",
}
MODEL_ORDER = {
    "svm_rbf": 0,
    "random_forest": 1,
    "knn": 2,
    "cnn1d": 3,
    "cnn2d": 4,
    "cnn3d": 5,
    "HybridSN-small": 6,
    "Spectral QNN Gated Fusion": 7,
    "Spectral QNN Gated Fusion + Prototype Loss": 8,
    "Spectral QNN Gated Fusion + SupCon Loss": 9,
}
MODEL_LABELS = {
    "svm_rbf": "SVM-RBF",
    "random_forest": "Random Forest",
    "knn": "kNN",
    "cnn1d": "CNN1D",
    "cnn2d": "CNN2D",
    "cnn3d": "CNN3D",
    "hybridsn_small_spectral_qnn_gated_fusion": "Spectral QNN Gated Fusion",
}


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.concat(
        [
            _other_baselines(Path(args.other_summary)),
            _hybridsn_rows(Path(args.hybridsn_summary), Path(args.hybridsn_pavia_salinas_summary)),
            _qnn_rows(Path(args.qnn_metric_summary), Path(args.qnn_indian_gated_summary)),
        ],
        ignore_index=True,
    )
    summary["model_order"] = summary["model"].map(MODEL_ORDER).fillna(99)
    summary = summary.sort_values(["dataset", "shot", "model_order", "model"]).reset_index(drop=True)
    summary.drop(columns=["model_order"]).to_csv(output_dir / "summary_all_models.csv", index=False)
    report = _render_report(summary)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {len(summary)} summarized rows to {output_dir}")


def _other_baselines(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["model"] = frame["model"].replace(MODEL_LABELS)
    return _keep_columns(frame)


def _hybridsn_rows(main_path: Path, pavia_salinas_path: Path) -> pd.DataFrame:
    main = pd.read_csv(main_path)
    pavia_salinas = pd.read_csv(pavia_salinas_path)
    main = main.loc[
        (main["dataset"] == "indian_pines")
        | ((main["dataset"] == "pavia_university") & (main["shot"] == 1))
    ].copy()
    frame = pd.concat([main, pavia_salinas], ignore_index=True)
    frame["model"] = "HybridSN-small"
    return _keep_columns(frame)


def _qnn_rows(metric_summary_path: Path, indian_gated_path: Path) -> pd.DataFrame:
    metric = pd.read_csv(metric_summary_path)
    metric = metric.loc[metric["model"] != "HybridSN-small"].copy()

    gated = pd.read_csv(indian_gated_path)
    gated = gated.loc[gated["shot"] == 1].copy()
    gated["model"] = gated["model"].replace(MODEL_LABELS)
    return _keep_columns(pd.concat([metric, gated], ignore_index=True))


def _keep_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "shot", "model", "runs"]
    for metric in METRICS:
        columns.extend([f"mean_{metric}", f"std_{metric}"])
    return frame[columns].copy()


def _render_report(summary: pd.DataFrame) -> str:
    lines = [
        "# Few-shot HSI Model Summary",
        "",
        "This report combines the current 1/5/10-shot non-HybridSN baselines with the retained HybridSN-small and QNN comparison rows.",
        "",
        "## Scope",
        "",
        "- All reported values are mean +/- population standard deviation over the saved seeds.",
        "- Classical and CNN baseline tables cover 1/5/10-shot for all three datasets.",
        "- HybridSN-small covers Indian Pines 1/5/10-shot, Pavia University 1/5/10-shot, and Salinas 5/10-shot in the saved runs.",
        "- Current cross-dataset QNN comparison rows use Spectral QNN Gated Fusion + Prototype Loss; Indian Pines also keeps the available 1-shot gated-fusion row and 5/10-shot SupCon rows.",
        "- Empty model-task combinations are intentionally omitted when no saved run exists.",
        "",
    ]
    for dataset in ("indian_pines", "pavia_university", "salinas"):
        lines.extend([f"## {DATASET_LABELS[dataset]}", ""])
        for shot in (1, 5, 10):
            group = summary.loc[(summary["dataset"] == dataset) & (summary["shot"] == shot)].copy()
            if group.empty:
                continue
            display = pd.DataFrame(
                {
                    "Model": group["model"],
                    "Runs": group["runs"].astype(int),
                    "OA": _metric_text(group, "OA"),
                    "AA": _metric_text(group, "AA"),
                    "Kappa": _metric_text(group, "Kappa"),
                    "Macro-F1": _metric_text(group, "Macro-F1"),
                    "Weighted-F1": _metric_text(group, "Weighted-F1"),
                }
            )
            lines.extend([f"### {DATASET_LABELS[dataset]} {shot}-shot", "", display.to_markdown(index=False), ""])
    return "\n".join(lines).rstrip() + "\n"


def _metric_text(frame: pd.DataFrame, metric: str) -> pd.Series:
    mean = (frame[f"mean_{metric}"] * 100).round(2).map(lambda value: f"{value:.2f}")
    std = (frame[f"std_{metric}"] * 100).round(2).map(lambda value: f"{value:.2f}")
    return mean + " +/- " + std


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize current few-shot classical, HybridSN-small, and QNN results.")
    parser.add_argument(
        "--other_summary",
        default="result/other_baselines_fewshot_summary/summary_by_dataset_model_shot.csv",
    )
    parser.add_argument(
        "--hybridsn_summary",
        default="result/hybridsn_small_fewshot_3datasets/metrics/summary_by_dataset_shot.csv",
    )
    parser.add_argument(
        "--hybridsn_pavia_salinas_summary",
        default="result/hybridsn_small_fewshot_pavia_salinas_5_10shot/metrics/summary_by_dataset_shot.csv",
    )
    parser.add_argument(
        "--qnn_metric_summary",
        default="result/fewshot_metric_loss_cross_dataset_summary/all_model_summary.csv",
    )
    parser.add_argument(
        "--qnn_indian_gated_summary",
        default="result/hybridsn_small_spectral_qnn_gated_fewshot_indian_pines/metrics/summary_by_shot_spectral_gated_qnn.csv",
    )
    parser.add_argument("--output_dir", default="result/all_fewshot_model_summary")
    return parser


if __name__ == "__main__":
    main()
