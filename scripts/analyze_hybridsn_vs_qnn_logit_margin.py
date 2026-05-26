from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.run_hybridsn_small_fewshot import OnTheFlyPatchDataset, _load_dataset_config, _preprocess_full_image
from scripts.run_hybridsn_small_spectral_qnn_gated_metric_fewshot import SpectralQNNGatedMetricFusion
from src.analysis.metrics import classification_metrics, per_class_metrics
from src.datasets.hsi_dataset import load_hsi_mat
from src.models.classical import HybridSNSmall

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HYBRIDSN_MODEL = "hybridsn_small"
QNN_MODEL = "spectral_qnn_gated_proto"
HIGHER_IS_BETTER = (
    "mean_true_logit_margin",
    "median_true_logit_margin",
    "safe_logit_margin_rate",
    "mean_top1_top2_margin",
    "mean_true_prob_margin",
    "median_true_prob_margin",
    "safe_prob_margin_rate",
    "mean_top1_top2_prob_margin",
    "OA",
    "Macro-F1",
)
LOWER_IS_BETTER = (
    "negative_logit_margin_rate",
    "low_logit_margin_rate",
    "negative_prob_margin_rate",
    "low_prob_margin_rate",
)
PAIR_METRICS = (
    "mean_true_logit_margin",
    "negative_logit_margin_rate",
    "low_logit_margin_rate",
    "mean_true_prob_margin",
    "negative_prob_margin_rate",
    "safe_prob_margin_rate",
    "Macro-F1",
    "OA",
)
REPORT_PAIR_METRICS = ("mean_true_logit_margin", "negative_logit_margin_rate", "mean_true_prob_margin", "Macro-F1")


@dataclass
class DatasetBundle:
    padded_cube: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    labels: np.ndarray
    class_names: dict[int, str]
    num_classes: int


@dataclass
class FeatureBundle:
    z: np.ndarray
    spectra: np.ndarray
    labels: np.ndarray
    source: str


def main() -> None:
    args = _build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    for subdir in (
        "metrics",
        "margins",
        "plots/logit_margin_distribution",
        "plots/prob_margin_distribution",
        "plots/per_class_logit_margin_delta",
    ):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    data_cache = DatasetCache(args)
    qnn_configs = {root: _read_json(root / "config.json") for root in args.qnn_result_dirs}
    failures: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    samples: list[pd.DataFrame] = []
    per_class: list[pd.DataFrame] = []

    for dataset in args.datasets:
        for shot in args.shots:
            for seed in args.seeds:
                split_path = _find_split(args.split_dirs, dataset, int(shot), int(seed))
                try:
                    split = _load_split(split_path)
                    bundle = data_cache.get(dataset)
                    _validate_split(split, bundle.labels, split_path)
                except Exception as exc:
                    _record_failure(failures, dataset, shot, seed, "shared_inputs", exc)
                    continue
                for model in (HYBRIDSN_MODEL, QNN_MODEL):
                    try:
                        if model == HYBRIDSN_MODEL:
                            logits, y_true, sample_idx, source = _hybrid_logits(args, bundle, split, dataset, int(shot), int(seed))
                        else:
                            logits, y_true, sample_idx, source = _qnn_logits(
                                args,
                                qnn_configs,
                                data_cache,
                                bundle,
                                split,
                                dataset,
                                int(shot),
                                int(seed),
                            )
                        margin_df = _margin_frame(args, dataset, model, int(shot), int(seed), sample_idx, y_true, logits)
                        run_row = _run_summary(margin_df, split, bundle.num_classes)
                        run_row["logit_source"] = source
                        class_df = _per_class_summary(bundle, margin_df, split)
                        runs.append(run_row)
                        samples.append(margin_df)
                        per_class.append(class_df)
                        margin_df.to_csv(out / "margins" / f"{dataset}_{model}_shot{shot}_seed{seed}_logit_margins.csv", index=False)
                    except Exception as exc:
                        _record_failure(failures, dataset, shot, seed, model, exc)

    run_df = pd.DataFrame(runs)
    sample_df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    per_class_df = pd.concat(per_class, ignore_index=True) if per_class else pd.DataFrame()
    _write_json(out / "failed_analysis.json", {"failed_analysis": failures})
    _write_outputs(args, out, run_df, sample_df, per_class_df)
    print(f"Output directory: {out}")


class DatasetCache:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._cache: dict[str, DatasetBundle] = {}

    def get(self, dataset: str) -> DatasetBundle:
        if dataset not in self._cache:
            cfg = _load_dataset_config(dataset, self.args.data_root)
            raw = load_hsi_mat(cfg)
            rows, cols = np.nonzero(raw.gt != raw.background_label)
            labels = raw.gt[rows, cols].astype(np.int64) - 1
            cube_pca, _ = _preprocess_full_image(raw.cube, self.args.pca_bands, self.args.seed)
            radius = self.args.patch_size // 2
            padded_cube = np.pad(cube_pca, ((radius, radius), (radius, radius), (0, 0)), mode="reflect").astype(np.float32)
            class_names = {int(label) - 1: str(name) for label, name in cfg["class_names"].items()}
            self._cache[dataset] = DatasetBundle(padded_cube, rows, cols, labels, class_names, int(cfg["num_classes"]))
        return self._cache[dataset]


def _hybrid_logits(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    split: dict[str, Any],
    dataset: str,
    shot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    ckpt = _find_hybrid_checkpoint(args.hybridsn_result_dirs, dataset, shot, seed)
    if ckpt is None:
        raise FileNotFoundError(f"Missing HybridSN checkpoint for {dataset} shot{shot} seed{seed}")
    model = HybridSNSmall(
        pca_channels=args.pca_bands,
        num_classes=bundle.num_classes,
        patch_size=args.patch_size,
        conv3d_channels=tuple(args.conv3d_channels),
        conv2d_channels=args.conv2d_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(args.device)
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    model.eval()
    indices = np.asarray(split["test"], dtype=np.int64)
    ds = OnTheFlyPatchDataset(bundle.padded_cube, bundle.rows, bundle.cols, bundle.labels, indices.tolist(), args.patch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logits, labels = [], []
    with torch.no_grad():
        for patch, y in loader:
            logits.append(model(patch.to(args.device)).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels), indices, f"checkpoint:{ckpt}"


def _qnn_logits(
    args: argparse.Namespace,
    qnn_configs: dict[Path, dict[str, Any]],
    data_cache: DatasetCache,
    bundle: DatasetBundle,
    split: dict[str, Any],
    dataset: str,
    shot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    checkpoint = _find_qnn_checkpoint(args.qnn_result_dirs, dataset, shot, seed)
    if checkpoint is None:
        raise FileNotFoundError(f"Missing QNN checkpoint for {dataset} shot{shot} seed{seed}")
    features = _load_or_extract_features(args, data_cache, dataset, shot, seed)
    if len(features.labels) != len(bundle.labels) or not np.array_equal(features.labels, bundle.labels):
        raise ValueError(f"QNN feature labels do not match dataset labels: {features.source}")
    config = qnn_configs.get(checkpoint.parents[1], {})
    model = SpectralQNNGatedMetricFusion(
        embedding_dim=features.z.shape[1],
        spectral_dim=features.spectra.shape[1],
        num_classes=bundle.num_classes,
        gate_mode=_config_value(args, config, "gate_mode"),
        qubits=int(_config_value(args, config, "qubits")),
        layers=int(_config_value(args, config, "qnn_layers")),
        entanglement=_config_value(args, config, "entanglement"),
        backend=_config_value(args, config, "backend"),
        diff_method=_config_value(args, config, "diff_method"),
        normalize_input=True,
        angle_scale=float(_config_value(args, config, "angle_scale")),
    ).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    model.eval()
    indices = np.asarray(split["test"], dtype=np.int64)
    logits = []
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            idx = indices[start : start + args.batch_size]
            z = torch.from_numpy(features.z[idx]).float().to(args.device)
            spectra = torch.from_numpy(features.spectra[idx]).float().to(args.device)
            logits.append(model(z, spectra).cpu().numpy())
    return np.concatenate(logits), bundle.labels[indices], indices, f"checkpoint:{checkpoint};features:{features.source}"


def _load_or_extract_features(args: argparse.Namespace, data_cache: DatasetCache, dataset: str, shot: int, seed: int) -> FeatureBundle:
    for path in _feature_candidates(args, dataset, shot, seed):
        if not path.exists():
            continue
        with np.load(path) as data:
            z = _first_array(data, ("z", "z_hybrid", "hybrid_feature", "encoder_feature"))
            spectra = _first_array(data, ("spectra", "spectrum"))
            labels = _first_array(data, ("y", "labels", "label"))
        if z is not None and spectra is not None and labels is not None:
            return FeatureBundle(z.astype(np.float32), spectra.astype(np.float32), labels.astype(np.int64), f"npz:{path}")
    bundle = data_cache.get(dataset)
    ckpt = _find_hybrid_checkpoint(args.hybridsn_result_dirs, dataset, shot, seed)
    if ckpt is None:
        raise FileNotFoundError(f"QNN feature extraction needs HybridSN checkpoint for {dataset} shot{shot} seed{seed}")
    model = HybridSNSmall(
        pca_channels=args.pca_bands,
        num_classes=bundle.num_classes,
        patch_size=args.patch_size,
        conv3d_channels=tuple(args.conv3d_channels),
        conv2d_channels=args.conv2d_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(args.device)
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    model.eval()
    all_idx = list(range(len(bundle.labels)))
    loader = DataLoader(OnTheFlyPatchDataset(bundle.padded_cube, bundle.rows, bundle.cols, bundle.labels, all_idx, args.patch_size), batch_size=args.batch_size, shuffle=False, num_workers=0)
    center = args.patch_size // 2
    z_parts, spectra_parts = [], []
    with torch.no_grad():
        for patch, _ in loader:
            spectra_parts.append(patch[:, center, center, :].numpy())
            x = model._prepare_input(patch.to(args.device))
            x = model.conv3d(x)
            batch, channels, spectral, height, width = x.shape
            x = x.reshape(batch, channels * spectral, height, width)
            z_parts.append(model.conv2d(x).flatten(1).cpu().numpy())
    return FeatureBundle(np.concatenate(z_parts).astype(np.float32), np.concatenate(spectra_parts).astype(np.float32), bundle.labels, f"extracted:{ckpt}")


def _margin_frame(
    args: argparse.Namespace,
    dataset: str,
    model: str,
    shot: int,
    seed: int,
    sample_idx: np.ndarray,
    y_true: np.ndarray,
    logits: np.ndarray,
) -> pd.DataFrame:
    logits = logits.astype(np.float64)
    row_idx = np.arange(len(logits))
    true_logit = logits[row_idx, y_true]
    wrong = logits.copy()
    wrong[row_idx, y_true] = -np.inf
    highest_wrong_logit = wrong.max(axis=1)
    true_logit_margin = true_logit - highest_wrong_logit
    top_logits = np.sort(logits, axis=1)[:, -2:]
    top1_logit = top_logits[:, 1]
    top2_logit = top_logits[:, 0]
    probs = _softmax(logits)
    true_prob = probs[row_idx, y_true]
    wrong_prob = probs.copy()
    wrong_prob[row_idx, y_true] = -np.inf
    highest_wrong_prob = wrong_prob.max(axis=1)
    true_prob_margin = true_prob - highest_wrong_prob
    top_probs = np.sort(probs, axis=1)[:, -2:]
    pred = logits.argmax(axis=1).astype(np.int64)
    return pd.DataFrame(
        {
            "dataset": dataset,
            "model": model,
            "shot": shot,
            "seed": seed,
            "sample_index": sample_idx,
            "true_label": y_true.astype(np.int64),
            "pred_label": pred,
            "correct": pred == y_true,
            "true_logit": true_logit,
            "highest_wrong_logit": highest_wrong_logit,
            "true_logit_margin": true_logit_margin,
            "top1_logit": top1_logit,
            "top2_logit": top2_logit,
            "top1_top2_margin": top1_logit - top2_logit,
            "true_prob": true_prob,
            "highest_wrong_prob": highest_wrong_prob,
            "true_prob_margin": true_prob_margin,
            "top1_prob": top_probs[:, 1],
            "top2_prob": top_probs[:, 0],
            "top1_top2_prob_margin": top_probs[:, 1] - top_probs[:, 0],
            "negative_logit_margin": true_logit_margin < 0,
            "low_logit_margin": (true_logit_margin >= 0) & (true_logit_margin < args.low_logit_margin_threshold),
            "negative_prob_margin": true_prob_margin < 0,
            "low_prob_margin": (true_prob_margin >= 0) & (true_prob_margin < args.low_prob_margin_threshold),
        }
    )


def _run_summary(frame: pd.DataFrame, split: dict[str, Any], num_classes: int) -> dict[str, Any]:
    metrics = classification_metrics(frame["true_label"].to_numpy(), frame["pred_label"].to_numpy(), labels=list(range(num_classes)))
    row = {
        "dataset": frame["dataset"].iloc[0],
        "shot": int(frame["shot"].iloc[0]),
        "seed": int(frame["seed"].iloc[0]),
        "model": frame["model"].iloc[0],
        "OA": metrics["OA"],
        "Macro-F1": metrics["Macro-F1"],
        "mean_true_logit_margin": float(frame["true_logit_margin"].mean()),
        "median_true_logit_margin": float(frame["true_logit_margin"].median()),
        "std_true_logit_margin": float(frame["true_logit_margin"].std(ddof=0)),
        "negative_logit_margin_rate": float(frame["negative_logit_margin"].mean()),
        "low_logit_margin_rate": float(frame["low_logit_margin"].mean()),
        "safe_logit_margin_rate": float((~frame["negative_logit_margin"] & ~frame["low_logit_margin"]).mean()),
        "mean_top1_top2_margin": float(frame["top1_top2_margin"].mean()),
        "median_top1_top2_margin": float(frame["top1_top2_margin"].median()),
        "mean_true_prob_margin": float(frame["true_prob_margin"].mean()),
        "median_true_prob_margin": float(frame["true_prob_margin"].median()),
        "negative_prob_margin_rate": float(frame["negative_prob_margin"].mean()),
        "low_prob_margin_rate": float(frame["low_prob_margin"].mean()),
        "safe_prob_margin_rate": float((~frame["negative_prob_margin"] & ~frame["low_prob_margin"]).mean()),
        "mean_top1_top2_prob_margin": float(frame["top1_top2_prob_margin"].mean()),
        "train_size": len(split["train"]),
        "validation_size": len(split["validation"]),
        "test_size": len(split["test"]),
    }
    return row


def _per_class_summary(bundle: DatasetBundle, margin_df: pd.DataFrame, split: dict[str, Any]) -> pd.DataFrame:
    metric_df = per_class_metrics(margin_df["true_label"].to_numpy(), margin_df["pred_label"].to_numpy(), bundle.class_names)
    rows = []
    train_y = bundle.labels[np.asarray(split["train"], dtype=np.int64)]
    for metric in metric_df.itertuples(index=False):
        group = margin_df[margin_df["true_label"] == metric.class_id]
        rows.append(
            {
                "dataset": margin_df["dataset"].iloc[0],
                "shot": int(margin_df["shot"].iloc[0]),
                "seed": int(margin_df["seed"].iloc[0]),
                "model": margin_df["model"].iloc[0],
                "class_id": int(metric.class_id),
                "class_name": metric.class_name,
                "support": int(np.sum(train_y == metric.class_id)),
                "OA_or_recall": float(metric.recall),
                "F1": float(metric.f1),
                "mean_true_logit_margin": float(group["true_logit_margin"].mean()),
                "negative_logit_margin_rate": float(group["negative_logit_margin"].mean()),
                "mean_true_prob_margin": float(group["true_prob_margin"].mean()),
                "negative_prob_margin_rate": float(group["negative_prob_margin"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_outputs(args: argparse.Namespace, out: Path, runs: pd.DataFrame, samples: pd.DataFrame, per_class: pd.DataFrame) -> None:
    metrics = out / "metrics"
    runs.to_csv(metrics / "all_logit_margin_runs.csv", index=False)
    summary = _summary(runs)
    summary.to_csv(metrics / "logit_margin_summary_by_dataset_shot_model.csv", index=False)
    _write_markdown(summary, metrics / "logit_margin_summary_by_dataset_shot_model.md")
    delta = _delta(runs)
    delta.to_csv(metrics / "qnn_vs_hybridsn_logit_margin_delta.csv", index=False)
    _write_markdown(delta, metrics / "qnn_vs_hybridsn_logit_margin_delta.md")
    paired = _paired(runs)
    paired.to_csv(metrics / "paired_seed_logit_margin_delta.csv", index=False)
    class_summary = _per_class_aggregate(per_class)
    class_summary.to_csv(metrics / "per_class_logit_margin_summary.csv", index=False)
    class_delta = _per_class_delta(class_summary)
    class_delta.to_csv(metrics / "per_class_logit_margin_delta.csv", index=False)
    joint = _joint_geometry(args, summary)
    if not joint.empty:
        joint.to_csv(metrics / "geometry_vs_logit_margin_joint_summary.csv", index=False)
        _write_markdown(joint, metrics / "geometry_vs_logit_margin_joint_summary.md")
    _plot_margin_distributions(out, samples)
    _plot_per_class_delta(out, class_delta)
    _write_report(out, summary, delta, paired, class_delta, joint)


def _summary(runs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "shot",
        "model",
        "runs",
        "OA_mean",
        "OA_std",
        "Macro-F1_mean",
        "Macro-F1_std",
        "mean_true_logit_margin_mean",
        "mean_true_logit_margin_std",
        "negative_logit_margin_rate_mean",
        "negative_logit_margin_rate_std",
        "low_logit_margin_rate_mean",
        "low_logit_margin_rate_std",
        "safe_logit_margin_rate_mean",
        "safe_logit_margin_rate_std",
        "mean_true_prob_margin_mean",
        "mean_true_prob_margin_std",
        "negative_prob_margin_rate_mean",
        "negative_prob_margin_rate_std",
        "low_prob_margin_rate_mean",
        "low_prob_margin_rate_std",
        "safe_prob_margin_rate_mean",
        "safe_prob_margin_rate_std",
        "mean_top1_top2_margin_mean",
        "mean_top1_top2_prob_margin_mean",
    ]
    if runs.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    std_metrics = columns[4:-2]
    for keys, group in runs.groupby(["dataset", "shot", "model"], sort=True):
        row = {"dataset": keys[0], "shot": int(keys[1]), "model": keys[2], "runs": len(group)}
        for metric in ("OA", "Macro-F1", "mean_true_logit_margin", "negative_logit_margin_rate", "low_logit_margin_rate", "safe_logit_margin_rate", "mean_true_prob_margin", "negative_prob_margin_rate", "low_prob_margin_rate", "safe_prob_margin_rate"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        row["mean_top1_top2_margin_mean"] = float(group["mean_top1_top2_margin"].mean())
        row["mean_top1_top2_prob_margin_mean"] = float(group["mean_top1_top2_prob_margin"].mean())
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def _delta(runs: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "shot", "metric", "hybridsn_mean", "qnn_mean", "delta", "better_model"]
    if runs.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in runs.groupby(["dataset", "shot"], sort=True):
        for metric in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER):
            values = group.groupby("model")[metric].mean()
            if HYBRIDSN_MODEL not in values or QNN_MODEL not in values:
                continue
            hybrid = float(values[HYBRIDSN_MODEL])
            qnn = float(values[QNN_MODEL])
            higher = metric in HIGHER_IS_BETTER
            rows.append(
                {
                    "dataset": keys[0],
                    "shot": int(keys[1]),
                    "metric": metric,
                    "hybridsn_mean": hybrid,
                    "qnn_mean": qnn,
                    "delta": qnn - hybrid,
                    "better_model": _better_model(hybrid, qnn, higher),
                }
            )
    return pd.DataFrame(rows).reindex(columns=columns)


def _paired(runs: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "shot", "seed", "metric", "hybridsn_value", "qnn_value", "delta", "qnn_better"]
    if runs.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in runs.groupby(["dataset", "shot", "seed"], sort=True):
        by_model = group.set_index("model")
        if HYBRIDSN_MODEL not in by_model.index or QNN_MODEL not in by_model.index:
            continue
        for metric in PAIR_METRICS:
            hybrid = float(by_model.loc[HYBRIDSN_MODEL, metric])
            qnn = float(by_model.loc[QNN_MODEL, metric])
            rows.append(
                {
                    "dataset": keys[0],
                    "shot": int(keys[1]),
                    "seed": int(keys[2]),
                    "metric": metric,
                    "hybridsn_value": hybrid,
                    "qnn_value": qnn,
                    "delta": qnn - hybrid,
                    "qnn_better": bool(qnn > hybrid) if metric in HIGHER_IS_BETTER else bool(qnn < hybrid),
                }
            )
    return pd.DataFrame(rows).reindex(columns=columns)


def _per_class_aggregate(per_class: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "shot",
        "model",
        "class_id",
        "class_name",
        "support",
        "OA_or_recall",
        "F1",
        "mean_true_logit_margin",
        "negative_logit_margin_rate",
        "mean_true_prob_margin",
        "negative_prob_margin_rate",
    ]
    if per_class.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in per_class.groupby(["dataset", "shot", "model", "class_id", "class_name"], sort=True):
        row = {"dataset": keys[0], "shot": int(keys[1]), "model": keys[2], "class_id": int(keys[3]), "class_name": keys[4]}
        for metric in columns[5:]:
            row[metric] = float(group[metric].mean())
        row["support"] = int(round(row["support"]))
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def _per_class_delta(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "shot",
        "class_id",
        "class_name",
        "hybridsn_F1",
        "qnn_F1",
        "delta_F1",
        "hybridsn_negative_logit_margin_rate",
        "qnn_negative_logit_margin_rate",
        "delta_negative_logit_margin_rate",
        "hybridsn_mean_true_logit_margin",
        "qnn_mean_true_logit_margin",
        "delta_mean_true_logit_margin",
    ]
    rows = []
    for keys, group in summary.groupby(["dataset", "shot", "class_id", "class_name"], sort=True):
        by_model = group.set_index("model")
        if HYBRIDSN_MODEL not in by_model.index or QNN_MODEL not in by_model.index:
            continue
        rows.append(
            {
                "dataset": keys[0],
                "shot": int(keys[1]),
                "class_id": int(keys[2]),
                "class_name": keys[3],
                "hybridsn_F1": float(by_model.loc[HYBRIDSN_MODEL, "F1"]),
                "qnn_F1": float(by_model.loc[QNN_MODEL, "F1"]),
                "delta_F1": float(by_model.loc[QNN_MODEL, "F1"] - by_model.loc[HYBRIDSN_MODEL, "F1"]),
                "hybridsn_negative_logit_margin_rate": float(by_model.loc[HYBRIDSN_MODEL, "negative_logit_margin_rate"]),
                "qnn_negative_logit_margin_rate": float(by_model.loc[QNN_MODEL, "negative_logit_margin_rate"]),
                "delta_negative_logit_margin_rate": float(by_model.loc[QNN_MODEL, "negative_logit_margin_rate"] - by_model.loc[HYBRIDSN_MODEL, "negative_logit_margin_rate"]),
                "hybridsn_mean_true_logit_margin": float(by_model.loc[HYBRIDSN_MODEL, "mean_true_logit_margin"]),
                "qnn_mean_true_logit_margin": float(by_model.loc[QNN_MODEL, "mean_true_logit_margin"]),
                "delta_mean_true_logit_margin": float(by_model.loc[QNN_MODEL, "mean_true_logit_margin"] - by_model.loc[HYBRIDSN_MODEL, "mean_true_logit_margin"]),
            }
        )
    return pd.DataFrame(rows).reindex(columns=columns)


def _joint_geometry(args: argparse.Namespace, logit_summary: pd.DataFrame) -> pd.DataFrame:
    geometry_path = Path(args.prototype_geometry_summary)
    columns = ["dataset", "shot", "model", "Macro-F1_mean", "separation_ratio_mean", "prototype_negative_margin_rate_mean", "mean_true_logit_margin_mean", "negative_logit_margin_rate_mean"]
    if not geometry_path.exists() or logit_summary.empty:
        return pd.DataFrame(columns=columns)
    geometry = pd.read_csv(geometry_path).rename(columns={"negative_margin_rate_mean": "prototype_negative_margin_rate_mean"})
    cols = ["dataset", "shot", "model", "separation_ratio_mean", "prototype_negative_margin_rate_mean"]
    joint = logit_summary.merge(geometry[cols], on=["dataset", "shot", "model"], how="inner")
    return joint.reindex(columns=columns)


def _plot_margin_distributions(out: Path, samples: pd.DataFrame) -> None:
    if samples.empty:
        return
    for keys, group in samples.groupby(["dataset", "shot"], sort=True):
        _distribution_plot(out / "plots/logit_margin_distribution" / f"{keys[0]}_shot{int(keys[1])}_true_logit_margin_distribution.png", group, "true_logit_margin", f"{keys[0]} {int(keys[1])}-shot true logit margin", "True-class logit margin")
        _distribution_plot(out / "plots/prob_margin_distribution" / f"{keys[0]}_shot{int(keys[1])}_true_prob_margin_distribution.png", group, "true_prob_margin", f"{keys[0]} {int(keys[1])}-shot true probability margin", "True-class probability margin")


def _distribution_plot(path: Path, group: pd.DataFrame, column: str, title: str, xlabel: str) -> None:
    colors = {HYBRIDSN_MODEL: "#2b6cb0", QNN_MODEL: "#c05621"}
    names = {HYBRIDSN_MODEL: "HybridSN-small", QNN_MODEL: "Spectral QNN Gated Fusion + Prototype"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    bins = np.histogram_bin_edges(group[column].to_numpy(), bins="auto")
    if len(bins) < 8:
        bins = 30
    for model in (HYBRIDSN_MODEL, QNN_MODEL):
        part = group[group["model"] == model]
        if part.empty:
            continue
        ax.hist(part[column], bins=bins, density=True, alpha=0.38, edgecolor=colors[model], color=colors[model], linewidth=1.0, label=names[model])
    ax.axvline(0.0, color="#1a202c", linestyle="--", linewidth=1.2, label="margin = 0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_per_class_delta(out: Path, delta: pd.DataFrame) -> None:
    if delta.empty:
        return
    for keys, group in delta.groupby(["dataset", "shot"], sort=True):
        ordered = group.sort_values("class_id")
        x = np.arange(len(ordered))
        fig, axes = plt.subplots(2, 1, figsize=(max(8.2, 0.55 * len(ordered)), 7.0), dpi=180, sharex=True)
        axes[0].bar(x, ordered["delta_mean_true_logit_margin"], color="#2f855a")
        axes[0].axhline(0, color="#1a202c", linewidth=1)
        axes[0].set_ylabel("Delta mean true logit margin")
        axes[0].grid(axis="y", alpha=0.22)
        axes[1].bar(x, ordered["delta_F1"], color="#c05621")
        axes[1].axhline(0, color="#1a202c", linewidth=1)
        axes[1].set_ylabel("Delta F1")
        axes[1].grid(axis="y", alpha=0.22)
        axes[1].set_xticks(x, [f"{row.class_id}: {row.class_name}" for row in ordered.itertuples()], rotation=45, ha="right")
        fig.suptitle(f"{keys[0]} {int(keys[1])}-shot per-class QNN minus HybridSN")
        fig.tight_layout()
        fig.savefig(out / "plots/per_class_logit_margin_delta" / f"{keys[0]}_shot{int(keys[1])}_per_class_delta_margin.png")
        plt.close(fig)


def _write_report(out: Path, summary: pd.DataFrame, delta: pd.DataFrame, paired: pd.DataFrame, class_delta: pd.DataFrame, joint: pd.DataFrame) -> None:
    lines = [
        "# HybridSN-small vs Spectral QNN Logit Margin Analysis",
        "",
        "## 1. 分析目的",
        "",
        "Prototype geometry 不能完全代表最终分类边界，因此本实验直接分析 classifier logits，判断 QNN 融合是否改善真正用于预测的 decision boundary。",
        "",
        "## 2. 方法",
        "",
        "- true-class logit margin = 真实类别 logit 与最高错误类别 logit 的差。",
        "- top1-top2 margin = 预测第一候选与第二候选 logit 的差。",
        "- 同时计算 softmax probability margin，缓解不同模型 logits scale 不一致的问题。",
        "- negative / low / safe margin rate 分别描述错误侧、靠近边界和远离阈值边界的样本比例。",
        "- 所有 margin 只在已有 split 的 test set 上计算，并做 paired seed 与 per-class 分析。",
        "",
        "## 3. 主结果表",
        "",
        _markdown(summary),
        "",
        "## 4. QNN vs HybridSN 差值",
        "",
        _markdown(delta),
        "",
        "## 5. Paired seed 分析",
        "",
        *_paired_lines(paired),
        "",
        "## 6. 与 prototype geometry 结果的关系",
        "",
        *_relation_lines(delta, joint),
        "",
        "## 7. Per-class 分析",
        "",
        *_per_class_lines(class_delta),
        "",
        "## 8. 初步结论",
        "",
        *_conclusion_lines(delta),
    ]
    if joint.empty:
        lines.extend(["", "Prototype geometry summary 未找到或没有可联结行，未生成 geometry-vs-logit 联合表。"])
    else:
        lines.extend(["", "Prototype geometry 与 logit margin 联合表已写入 `metrics/geometry_vs_logit_margin_joint_summary.md`。"])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paired_lines(paired: pd.DataFrame) -> list[str]:
    if paired.empty:
        return ["没有形成 paired seed。"]
    lines = []
    for keys, group in paired[paired["metric"].isin(REPORT_PAIR_METRICS)].groupby(["dataset", "shot"], sort=True):
        parts = []
        for metric in REPORT_PAIR_METRICS:
            metric_df = group[group["metric"] == metric]
            parts.append(f"{metric}: QNN better in {int(metric_df['qnn_better'].sum())}/{len(metric_df)} seeds")
        lines.append(f"- {keys[0]} {int(keys[1])}-shot: " + "; ".join(parts))
    return lines


def _relation_lines(delta: pd.DataFrame, joint: pd.DataFrame) -> list[str]:
    if delta.empty:
        return ["缺少可比较的 logit delta。"]
    lines = []
    for keys, group in delta.groupby(["dataset", "shot"], sort=True):
        rows = {row.metric: row for row in group.itertuples()}
        perf_up = rows.get("Macro-F1") is not None and rows["Macro-F1"].delta > 0
        logit_up = rows.get("mean_true_prob_margin") is not None and rows["mean_true_prob_margin"].delta > 0
        text = "分类性能未提升。"
        if perf_up and logit_up:
            text = "分类性能和 probability logit margin 同步提升，支持 classifier decision boundary 改善。"
        elif perf_up:
            text = "分类性能提升但平均 probability margin 未提升，提升可能集中于少数类别或特定样本。"
        if keys[0] == "salinas" and int(keys[1]) == 10 and not logit_up:
            text += " Salinas 10-shot 的 logit margin 下降支持 baseline 较强时可能出现负迁移。"
        if not joint.empty:
            geom = joint[(joint["dataset"] == keys[0]) & (joint["shot"] == int(keys[1]))]
            if len(geom) == 2:
                separation_delta = float(geom[geom["model"] == QNN_MODEL]["separation_ratio_mean"].iloc[0] - geom[geom["model"] == HYBRIDSN_MODEL]["separation_ratio_mean"].iloc[0])
                if perf_up and logit_up and separation_delta <= 0:
                    text += " 同时 prototype separation 未提升，说明改善更接近最终 classifier 边界而非 prototype geometry。"
        lines.append(f"- {keys[0]} {int(keys[1])}-shot: {text}")
    return lines


def _per_class_lines(delta: pd.DataFrame) -> list[str]:
    if delta.empty:
        return ["没有 per-class delta。"]
    lines = []
    for keys, group in delta.groupby(["dataset", "shot"], sort=True):
        f1 = int((group["delta_F1"] > 0).sum())
        less_negative = int((group["delta_negative_logit_margin_rate"] < 0).sum())
        margin = int((group["delta_mean_true_logit_margin"] > 0).sum())
        lines.append(f"- {keys[0]} {int(keys[1])}-shot: F1 提升 {f1}/{len(group)} 类；negative logit margin rate 下降 {less_negative}/{len(group)} 类；mean true logit margin 提升 {margin}/{len(group)} 类。")
    return lines


def _conclusion_lines(delta: pd.DataFrame) -> list[str]:
    if delta.empty:
        return ["没有形成结论。"]
    positive = []
    for keys, group in delta.groupby(["dataset", "shot"], sort=True):
        rows = {row.metric: row for row in group.itertuples()}
        if rows["mean_true_prob_margin"].delta > 0 and rows["negative_logit_margin_rate"].delta < 0:
            positive.append(f"{keys[0]} {int(keys[1])}-shot")
    if not positive:
        return ["当前设置中 QNN 没有稳定提升最终 logits margin，不能把分类差异解释为 classifier decision boundary 的整体改善。"]
    return [
        "在某些 dataset / shot 设置中，QNN 提升了最终分类 logits 或 probability margin，说明其可能改善 classifier decision boundary；该判断仍以对应设置为限。",
        "同时满足 mean probability margin 提升且 negative logit margin rate 降低的设置为：" + "、".join(positive) + "。",
    ]


def _feature_candidates(args: argparse.Namespace, dataset: str, shot: int, seed: int) -> list[Path]:
    roots = [Path(path) for path in args.qnn_feature_dirs]
    roots.extend(root / "features" for root in args.qnn_result_dirs)
    candidates = []
    for root in roots:
        candidates.extend([root / f"{dataset}_shot{shot}_seed{seed}_features.npz", root / f"{dataset}_shot{shot}_seed{seed}_embeddings.npz"])
    return list(dict.fromkeys(candidates))


def _find_qnn_checkpoint(roots: list[Path], dataset: str, shot: int, seed: int) -> Path | None:
    names = (
        f"{dataset}_spectral_gated_qnn_prototype_shot{shot}_seed{seed}.pt",
        f"{dataset}_spectral_gated_qnn_proto_shot{shot}_seed{seed}.pt",
        f"{dataset}_shot{shot}_seed{seed}.pt",
    )
    for root in roots:
        for name in names:
            path = root / "checkpoints" / name
            if path.exists():
                return path
        matches = sorted((root / "checkpoints").glob(f"{dataset}*shot{shot}_seed{seed}.pt"))
        if matches:
            return matches[0]
    return None


def _find_hybrid_checkpoint(roots: list[Path], dataset: str, shot: int, seed: int) -> Path | None:
    for root in roots:
        path = root / "checkpoints" / f"{dataset}_shot{shot}_seed{seed}.pt"
        if path.exists():
            return path
    return None


def _find_split(roots: list[Path], dataset: str, shot: int, seed: int) -> Path:
    for root in roots:
        path = root / f"{dataset}_seed{seed}_{shot}shot.json"
        if path.exists():
            return path
    return roots[0] / f"{dataset}_seed{seed}_{shot}shot.json"


def _load_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_split(split: dict[str, Any], labels: np.ndarray, path: Path) -> None:
    if set(split["train"]) & set(split["test"]):
        raise ValueError(f"Train/test overlap in split: {path}")
    indices = np.asarray([*split["train"], *split["test"]], dtype=np.int64)
    if not len(indices) or indices.min() < 0 or indices.max() >= len(labels):
        raise IndexError(f"Split index out of range: {path}")


def _config_value(args: argparse.Namespace, config: dict[str, Any], name: str) -> Any:
    return config.get(name, getattr(args, name))


def _first_array(data: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in data:
            return data[name]
    return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _better_model(hybrid: float, qnn: float, higher: bool) -> str:
    if np.isclose(hybrid, qnn):
        return "tie"
    if higher:
        return QNN_MODEL if qnn > hybrid else HYBRIDSN_MODEL
    return QNN_MODEL if qnn < hybrid else HYBRIDSN_MODEL


def _write_markdown(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(_markdown(frame) + "\n", encoding="utf-8")


def _markdown(frame: pd.DataFrame) -> str:
    return "No completed runs." if frame.empty else frame.round(6).to_markdown(index=False)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _record_failure(failures: list[dict[str, Any]], dataset: str, shot: int, seed: int, model: str, exc: Exception) -> None:
    failure = {"dataset": dataset, "shot": int(shot), "seed": int(seed), "model": model, "error": repr(exc)}
    failures.append(failure)
    print(f"[WARN] skipped {failure}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze final classifier logit margins for HybridSN-small versus prototype QNN fusion.")
    parser.add_argument("--datasets", nargs="+", default=["indian_pines", "pavia_university", "salinas"])
    parser.add_argument("--shots", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--hybridsn_result_dir", default="result/hybridsn_small_fewshot_3datasets")
    parser.add_argument("--hybridsn_result_dirs", nargs="+", default=None)
    parser.add_argument("--qnn_result_dir", default="result/hybridsn_small_spectral_qnn_gated_proto_tuning_3datasets")
    parser.add_argument("--qnn_result_dirs", nargs="+", default=None)
    parser.add_argument("--qnn_feature_dir", default=None)
    parser.add_argument("--qnn_feature_dirs", nargs="+", default=None)
    parser.add_argument("--split_dir", default="result/hybridsn_small_fewshot_3datasets/split_indices")
    parser.add_argument("--split_dirs", nargs="+", default=None)
    parser.add_argument("--output_dir", default="result/logit_margin_hybridsn_vs_qnn")
    parser.add_argument("--prototype_geometry_summary", default="result/boundary_geometry_hybridsn_vs_qnn/metrics/geometry_summary.csv")
    parser.add_argument("--low_logit_margin_threshold", type=float, default=0.1)
    parser.add_argument("--low_prob_margin_threshold", type=float, default=0.05)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--patch_size", type=int, default=19)
    parser.add_argument("--pca_bands", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--conv3d_channels", nargs=3, type=int, default=[8, 16, 16])
    parser.add_argument("--conv2d_channels", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--qnn_layers", type=int, default=1)
    parser.add_argument("--entanglement", default="linear")
    parser.add_argument("--gate_mode", choices=["scalar", "classwise"], default="classwise")
    parser.add_argument("--backend", default="lightning.qubit")
    parser.add_argument("--diff_method", default="adjoint")
    parser.add_argument("--angle_scale", type=float, default=float(np.pi))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    original_parse_args = parser.parse_args

    def parse_args(*parse_args: Any, **parse_kwargs: Any) -> argparse.Namespace:
        args = original_parse_args(*parse_args, **parse_kwargs)
        args.hybridsn_result_dirs = [Path(path) for path in (args.hybridsn_result_dirs or [args.hybridsn_result_dir])]
        args.qnn_result_dirs = [Path(path) for path in (args.qnn_result_dirs or [args.qnn_result_dir])]
        feature_dirs = args.qnn_feature_dirs or ([args.qnn_feature_dir] if args.qnn_feature_dir else [])
        args.qnn_feature_dirs = [Path(path) for path in feature_dirs]
        args.split_dirs = [Path(path) for path in (args.split_dirs or [args.split_dir])]
        return args

    parser.parse_args = parse_args  # type: ignore[method-assign]
    return parser


if __name__ == "__main__":
    main()
