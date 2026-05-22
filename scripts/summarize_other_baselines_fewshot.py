from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ("OA", "AA", "Kappa", "Macro-F1", "Weighted-F1")


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [_read_runs(Path(path)) for path in args.inputs]
    runs = pd.concat(frames, ignore_index=True)
    runs = runs.drop_duplicates(subset=["dataset", "model", "shot", "seed"], keep="last")
    runs = runs.sort_values(["dataset", "shot", "model", "seed"]).reset_index(drop=True)
    summary = _summarize(runs)
    coverage = _coverage(summary, args.expected_seeds)

    runs.to_csv(output_dir / "all_runs.csv", index=False)
    summary.to_csv(output_dir / "summary_by_dataset_model_shot.csv", index=False)
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    (output_dir / "summary_by_dataset_model_shot.md").write_text(
        _display_summary(summary).to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _report(summary, coverage, args.inputs, args.expected_seeds),
        encoding="utf-8",
    )
    print(f"Combined {len(runs)} runs into {output_dir}")


def _read_runs(path: Path) -> pd.DataFrame:
    runs = path / "metrics" / "all_runs_other_baselines.csv"
    if not runs.exists():
        raise FileNotFoundError(f"missing all-runs table: {runs}")
    frame = pd.read_csv(runs)
    frame["source_result_dir"] = str(path)
    return frame


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, shot), group in runs.groupby(["dataset", "model", "shot"], sort=True):
        row = {
            "dataset": dataset,
            "model": model,
            "shot": int(shot),
            "runs": int(len(group)),
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        row["mean_best_epoch"] = float(group["best_epoch"].mean())
        params = group["trainable_parameters"].dropna()
        row["trainable_parameters"] = params.iloc[0] if not params.empty else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "shot", "model"]).reset_index(drop=True)


def _coverage(summary: pd.DataFrame, expected_seeds: int) -> pd.DataFrame:
    coverage = summary[["dataset", "model", "shot", "runs"]].copy()
    coverage["expected_runs"] = int(expected_seeds)
    coverage["complete"] = coverage["runs"] == int(expected_seeds)
    return coverage


def _display_summary(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.copy()
    for column in display.columns:
        if column.startswith("mean_") or column.startswith("std_"):
            if column == "mean_best_epoch":
                display[column] = display[column].round(2)
            else:
                display[column] = (display[column] * 100).round(2)
    return display


def _report(summary: pd.DataFrame, coverage: pd.DataFrame, inputs: list[str], expected_seeds: int) -> str:
    display = _display_summary(summary).to_markdown(index=False)
    incomplete = coverage.loc[~coverage["complete"]]
    incomplete_text = incomplete.to_markdown(index=False) if not incomplete.empty else "None."
    sources = "\n".join(f"- `{path}`" for path in inputs)
    return f"""# Other Few-shot Baseline Summary

This report combines non-HybridSN baselines under the current all-way HSI few-shot protocol.

## Protocol

- Shots: 1, 5, 10 per class.
- Seeds expected per setting: {expected_seeds}.
- Metrics are mean and population standard deviation over completed seeds.
- Traditional baselines use center PCA spectral vectors.
- CNN1D uses center PCA spectral vectors.
- CNN2D and CNN3D use PCA spatial-spectral patches.
- The source runs match the current HybridSN-small few-shot preprocessing protocol: PCA is fit on the full image without labels.

## Source Result Directories

{sources}

## Summary

{display}

## Coverage Gaps

{incomplete_text}
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine non-HybridSN few-shot baseline result directories.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Result directories containing metrics/all_runs_other_baselines.csv.")
    parser.add_argument("--output_dir", default="result/other_baselines_fewshot_summary")
    parser.add_argument("--expected_seeds", type=int, default=5)
    return parser


if __name__ == "__main__":
    main()
