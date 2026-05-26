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
from src.datasets.hsi_dataset import load_hsi_mat
from src.models.classical import HybridSNSmall

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HYBRIDSN_MODEL = "hybridsn_small"
QNN_MODEL = "spectral_qnn_gated_proto"
METRIC_DIRECTIONS = {
    "mean_inter_distance": "higher",
    "separation_ratio": "higher",
    "mean_margin": "higher",
    "median_margin": "higher",
    "safe_margin_rate": "higher",
    "mean_intra_distance": "lower",
    "negative_margin_rate": "lower",
    "low_margin_rate": "lower",
}
PAIR_REPORT_METRICS = ("mean_margin", "negative_margin_rate", "separation_ratio")


@dataclass
class FeatureBundle:
    z: np.ndarray
    spectra: np.ndarray | None
    labels: np.ndarray
    source: str


def main() -> None:
    args = _build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    for subdir in ("metrics", "margins", "plots/margin_distribution"):
        (out / subdir).mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    margin_frames: list[pd.DataFrame] = []
    extractor = HybridFeatureExtractor(args)
    qnn_configs = {root: _read_config(root / "config.json") for root in args.qnn_result_dirs}

    for dataset in args.datasets:
        for shot in args.shots:
            for seed in args.seeds:
                split_path = _find_split(args.split_dirs, dataset, int(shot), int(seed))
                try:
                    split = _load_split(split_path)
                    bundle = _load_hybrid_bundle(args, extractor, dataset, int(shot), int(seed))
                    _validate_indices(bundle.labels, split, split_path)
                except Exception as exc:
                    _record_failure(failures, dataset, shot, seed, "shared_inputs", exc)
                    continue

                try:
                    row, margins = _analyze_one(
                        dataset,
                        int(shot),
                        int(seed),
                        HYBRIDSN_MODEL,
                        bundle.z,
                        bundle.labels,
                        split,
                        args.low_margin_threshold,
                    )
                    row["feature_source"] = bundle.source
                    geometry_rows.append(row)
                    margin_frames.append(margins)
                    _write_margins(out, margins, dataset, HYBRIDSN_MODEL, int(shot), int(seed))
                except Exception as exc:
                    _record_failure(failures, dataset, shot, seed, HYBRIDSN_MODEL, exc)

                try:
                    qnn_features, qnn_source = _load_qnn_features(
                        args,
                        qnn_configs,
                        dataset,
                        int(shot),
                        int(seed),
                        bundle,
                    )
                    row, margins = _analyze_one(
                        dataset,
                        int(shot),
                        int(seed),
                        QNN_MODEL,
                        qnn_features,
                        bundle.labels,
                        split,
                        args.low_margin_threshold,
                    )
                    row["feature_source"] = qnn_source
                    geometry_rows.append(row)
                    margin_frames.append(margins)
                    _write_margins(out, margins, dataset, QNN_MODEL, int(shot), int(seed))
                except Exception as exc:
                    _record_failure(failures, dataset, shot, seed, QNN_MODEL, exc)

    geometry = pd.DataFrame(geometry_rows)
    all_margins = pd.concat(margin_frames, ignore_index=True) if margin_frames else pd.DataFrame()
    _write_json(out / "failed_analysis.json", {"failed_analysis": failures})
    _write_outputs(args, out, geometry, all_margins)
    print(f"Output directory: {out}")


class HybridFeatureExtractor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}

    def extract(self, dataset: str, shot: int, seed: int) -> FeatureBundle:
        padded_cube, rows, cols, labels, num_classes = self._dataset(dataset)
        ckpt = _find_hybrid_checkpoint(self.args.hybridsn_result_dirs, dataset, shot, seed)
        if ckpt is None:
            raise FileNotFoundError(f"Missing HybridSN-small checkpoint for {dataset} shot{shot} seed{seed}")
        model = HybridSNSmall(
            pca_channels=self.args.pca_bands,
            num_classes=num_classes,
            patch_size=self.args.patch_size,
            conv3d_channels=tuple(self.args.conv3d_channels),
            conv2d_channels=self.args.conv2d_channels,
            hidden_dim=self.args.hidden_dim,
            dropout=self.args.dropout,
        ).to(self.args.device)
        model.load_state_dict(torch.load(ckpt, map_location=self.args.device))
        model.eval()
        ds = OnTheFlyPatchDataset(padded_cube, rows, cols, labels, list(range(len(labels))), self.args.patch_size)
        loader = DataLoader(ds, batch_size=self.args.feature_batch_size, shuffle=False, num_workers=0)
        center = self.args.patch_size // 2
        z_parts, spectra_parts = [], []
        with torch.no_grad():
            for patch, _ in loader:
                spectra_parts.append(patch[:, center, center, :].numpy())
                patch = patch.to(self.args.device)
                patch = model._prepare_input(patch)
                patch = model.conv3d(patch)
                batch, channels, spectral, height, width = patch.shape
                patch = patch.reshape(batch, channels * spectral, height, width)
                z_parts.append(model.conv2d(patch).flatten(1).cpu().numpy())
        return FeatureBundle(
            z=np.concatenate(z_parts).astype(np.float32),
            spectra=np.concatenate(spectra_parts).astype(np.float32),
            labels=labels.astype(np.int64),
            source=f"extracted:{ckpt}",
        )

    def _dataset(self, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        if dataset not in self.cache:
            cfg = _load_dataset_config(dataset, self.args.data_root)
            raw = load_hsi_mat(cfg)
            rows, cols = np.nonzero(raw.gt != raw.background_label)
            labels = raw.gt[rows, cols].astype(np.int64) - 1
            cube_pca, _ = _preprocess_full_image(raw.cube, self.args.pca_bands, self.args.seed)
            radius = self.args.patch_size // 2
            padded_cube = np.pad(cube_pca, ((radius, radius), (radius, radius), (0, 0)), mode="reflect").astype(np.float32)
            self.cache[dataset] = (padded_cube, rows, cols, labels, int(cfg["num_classes"]))
        return self.cache[dataset]


def _load_hybrid_bundle(args: argparse.Namespace, extractor: HybridFeatureExtractor, dataset: str, shot: int, seed: int) -> FeatureBundle:
    for path in _feature_candidates(args, dataset, shot, seed):
        if not path.exists():
            continue
        with np.load(path) as data:
            labels = _first_array(data, ("y", "labels", "label"))
            z = _first_array(data, ("z_hybrid", "z", "hybridsn_feature", "hybrid_feature", "encoder_feature"))
            spectra = _first_array(data, ("spectra", "spectrum"), required=False)
        if labels is not None and z is not None:
            return FeatureBundle(
                z=z.astype(np.float32),
                spectra=spectra.astype(np.float32) if spectra is not None else None,
                labels=labels.astype(np.int64),
                source=f"npz:{path}",
            )
    return extractor.extract(dataset, shot, seed)


def _load_qnn_features(
    args: argparse.Namespace,
    qnn_configs: dict[Path, dict[str, Any]],
    dataset: str,
    shot: int,
    seed: int,
    bundle: FeatureBundle,
) -> tuple[np.ndarray, str]:
    feature_path = next((path for path in _feature_candidates(args, dataset, shot, seed) if path.exists()), None)
    if feature_path:
        with np.load(feature_path) as data:
            fused = _first_array(
                data,
                ("z_qnn", "fused_feature", "fused_features", "fused", "prototype_feature"),
                required=False,
            )
        if fused is not None:
            return fused.astype(np.float32), f"npz-fused:{feature_path}"
    if bundle.spectra is None:
        raise FileNotFoundError("QNN reconstruction needs saved or extracted spectra.")
    checkpoint = _find_qnn_checkpoint(args.qnn_result_dirs, dataset, shot, seed)
    if checkpoint is None:
        raise FileNotFoundError(f"Missing QNN prototype checkpoint for {dataset} shot{shot} seed{seed}")
    fused = _reconstruct_qnn_features(args, qnn_configs.get(checkpoint.parents[1], {}), checkpoint, bundle.z, bundle.spectra, bundle.labels)
    return fused, f"checkpoint:{checkpoint}"


def _reconstruct_qnn_features(
    args: argparse.Namespace,
    qnn_config: dict[str, Any],
    checkpoint: Path,
    z: np.ndarray,
    spectra: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    # This class has the same state layout as the older Indian Pines prototype script.
    from scripts.run_hybridsn_small_spectral_qnn_gated_metric_fewshot import SpectralQNNGatedMetricFusion

    num_classes = int(labels.max()) + 1
    model = SpectralQNNGatedMetricFusion(
        embedding_dim=z.shape[1],
        spectral_dim=spectra.shape[1],
        num_classes=num_classes,
        gate_mode=_config_value(args, qnn_config, "gate_mode"),
        qubits=int(_config_value(args, qnn_config, "qubits")),
        layers=int(_config_value(args, qnn_config, "qnn_layers")),
        entanglement=_config_value(args, qnn_config, "entanglement"),
        backend=_config_value(args, qnn_config, "backend"),
        diff_method=_config_value(args, qnn_config, "diff_method"),
        normalize_input=True,
        angle_scale=float(_config_value(args, qnn_config, "angle_scale")),
    ).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(z), args.qnn_feature_batch_size):
            z_batch = torch.from_numpy(z[start : start + args.qnn_feature_batch_size]).float().to(args.device)
            s_batch = torch.from_numpy(spectra[start : start + args.qnn_feature_batch_size]).float().to(args.device)
            q_batch = model.quantum_features(s_batch)
            parts.append(torch.cat([z_batch, q_batch], dim=1).cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def _analyze_one(
    dataset: str,
    shot: int,
    seed: int,
    model: str,
    features: np.ndarray,
    labels: np.ndarray,
    split: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    support_idx = np.asarray(split["train"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)
    normalized = _l2_normalize(features.astype(np.float64))
    support, support_y = normalized[support_idx], labels[support_idx]
    query, query_y = normalized[test_idx], labels[test_idx]
    classes = np.unique(support_y)
    prototypes = np.stack([support[support_y == cls].mean(axis=0) for cls in classes])
    prototypes = _l2_normalize(prototypes)
    distances = _pairwise_euclidean(query, prototypes)
    class_to_col = {int(cls): col for col, cls in enumerate(classes)}
    correct_cols = np.asarray([class_to_col[int(label)] for label in query_y])
    d_correct = distances[np.arange(len(query)), correct_cols]
    wrong = distances.copy()
    wrong[np.arange(len(query)), correct_cols] = np.inf
    wrong_cols = wrong.argmin(axis=1)
    d_wrong = wrong[np.arange(len(query)), wrong_cols]
    margins = d_wrong - d_correct
    pred_cols = distances.argmin(axis=1)
    pred = classes[pred_cols].astype(np.int64)
    wrong_class = classes[wrong_cols].astype(np.int64)

    proto_distances = _pairwise_euclidean(prototypes, prototypes)
    inter = proto_distances[np.triu_indices(len(prototypes), k=1)]
    row = {
        "dataset": dataset,
        "shot": shot,
        "seed": seed,
        "model": model,
        "support_samples": len(support_idx),
        "test_samples": len(test_idx),
        "mean_intra_distance": _stat_mean(d_correct),
        "median_intra_distance": _stat_median(d_correct),
        "std_intra_distance": _stat_std(d_correct),
        "mean_inter_distance": _stat_mean(inter),
        "min_inter_distance": float(np.min(inter)) if len(inter) else np.nan,
        "median_inter_distance": _stat_median(inter),
        "separation_ratio": _safe_ratio(_stat_mean(inter), _stat_mean(d_correct)),
        "mean_margin": _stat_mean(margins),
        "median_margin": _stat_median(margins),
        "std_margin": _stat_std(margins),
        "negative_margin_rate": float(np.mean(margins < 0)),
        "low_margin_rate": float(np.mean((margins >= 0) & (margins < threshold))),
        "safe_margin_rate": float(np.mean(margins >= threshold)),
    }
    margin_df = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model,
            "shot": shot,
            "seed": seed,
            "sample_index": test_idx,
            "true_label": query_y.astype(np.int64),
            "pred_label": pred,
            "correct": pred == query_y,
            "d_correct": d_correct,
            "d_wrong_nearest": d_wrong,
            "nearest_wrong_class": wrong_class,
            "margin": margins,
            "is_negative_margin": margins < 0,
            "is_low_margin": (margins >= 0) & (margins < threshold),
        }
    )
    return row, margin_df


def _write_outputs(args: argparse.Namespace, out: Path, geometry: pd.DataFrame, margins: pd.DataFrame) -> None:
    metrics_dir = out / "metrics"
    run_path = metrics_dir / "geometry_by_run.csv"
    geometry.to_csv(run_path, index=False)
    classification = _load_classification_metrics(args)
    summary = _geometry_summary(geometry, classification)
    summary.to_csv(metrics_dir / "geometry_summary.csv", index=False)
    _write_markdown(summary, metrics_dir / "geometry_summary.md")

    margin_summary = _margin_summary(margins)
    margin_summary.to_csv(metrics_dir / "margin_distribution_summary.csv", index=False)
    _write_markdown(margin_summary, metrics_dir / "margin_distribution_summary.md")

    delta = _mean_delta(geometry)
    delta.to_csv(metrics_dir / "qnn_vs_hybridsn_geometry_delta.csv", index=False)
    _write_markdown(delta, metrics_dir / "qnn_vs_hybridsn_geometry_delta.md")

    paired = _paired_delta(geometry)
    paired.to_csv(metrics_dir / "paired_seed_geometry_delta.csv", index=False)
    _plot_margin_distributions(out, margins)
    _write_report(out, summary, delta, paired, margin_summary, classification.empty)


def _geometry_summary(geometry: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "shot",
        "model",
        "runs",
        "OA_mean",
        "Macro-F1_mean",
        "mean_intra_distance_mean",
        "mean_intra_distance_std",
        "mean_inter_distance_mean",
        "mean_inter_distance_std",
        "separation_ratio_mean",
        "separation_ratio_std",
        "mean_margin_mean",
        "mean_margin_std",
        "negative_margin_rate_mean",
        "negative_margin_rate_std",
        "low_margin_rate_mean",
        "low_margin_rate_std",
    ]
    if geometry.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in geometry.groupby(["dataset", "shot", "model"], sort=True):
        row = {"dataset": keys[0], "shot": int(keys[1]), "model": keys[2], "runs": len(group)}
        for metric in ("mean_intra_distance", "mean_inter_distance", "separation_ratio", "mean_margin", "negative_margin_rate", "low_margin_rate"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        metrics = classification[
            (classification["dataset"] == keys[0])
            & (classification["shot"] == int(keys[1]))
            & (classification["model"] == keys[2])
        ]
        row["OA_mean"] = float(metrics["OA"].mean()) if not metrics.empty else np.nan
        row["Macro-F1_mean"] = float(metrics["Macro-F1"].mean()) if not metrics.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def _margin_summary(margins: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "shot", "model", "samples", "mean_margin", "median_margin", "std_margin", "negative_margin_rate", "low_margin_rate", "safe_margin_rate"]
    if margins.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in margins.groupby(["dataset", "shot", "model"], sort=True):
        rows.append(
            {
                "dataset": keys[0],
                "shot": int(keys[1]),
                "model": keys[2],
                "samples": len(group),
                "mean_margin": float(group["margin"].mean()),
                "median_margin": float(group["margin"].median()),
                "std_margin": float(group["margin"].std(ddof=0)),
                "negative_margin_rate": float(group["is_negative_margin"].mean()),
                "low_margin_rate": float(group["is_low_margin"].mean()),
                "safe_margin_rate": float((~group["is_negative_margin"] & ~group["is_low_margin"]).mean()),
            }
        )
    return pd.DataFrame(rows).reindex(columns=columns)


def _mean_delta(geometry: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "shot", "metric", "hybridsn_mean", "qnn_mean", "delta", "better_model"]
    rows = []
    if geometry.empty:
        return pd.DataFrame(columns=columns)
    for keys, group in geometry.groupby(["dataset", "shot"], sort=True):
        for metric, direction in METRIC_DIRECTIONS.items():
            values = group.groupby("model")[metric].mean()
            if HYBRIDSN_MODEL not in values or QNN_MODEL not in values:
                continue
            hybrid, qnn = float(values[HYBRIDSN_MODEL]), float(values[QNN_MODEL])
            rows.append(
                {
                    "dataset": keys[0],
                    "shot": int(keys[1]),
                    "metric": metric,
                    "hybridsn_mean": hybrid,
                    "qnn_mean": qnn,
                    "delta": qnn - hybrid,
                    "better_model": _better_name(hybrid, qnn, direction),
                }
            )
    return pd.DataFrame(rows).reindex(columns=columns)


def _paired_delta(geometry: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "shot", "seed", "metric", "hybridsn_value", "qnn_value", "delta", "qnn_better"]
    rows = []
    if geometry.empty:
        return pd.DataFrame(columns=columns)
    for keys, group in geometry.groupby(["dataset", "shot", "seed"], sort=True):
        by_model = group.set_index("model")
        if HYBRIDSN_MODEL not in by_model.index or QNN_MODEL not in by_model.index:
            continue
        for metric, direction in METRIC_DIRECTIONS.items():
            hybrid, qnn = float(by_model.loc[HYBRIDSN_MODEL, metric]), float(by_model.loc[QNN_MODEL, metric])
            rows.append(
                {
                    "dataset": keys[0],
                    "shot": int(keys[1]),
                    "seed": int(keys[2]),
                    "metric": metric,
                    "hybridsn_value": hybrid,
                    "qnn_value": qnn,
                    "delta": qnn - hybrid,
                    "qnn_better": bool(qnn > hybrid) if direction == "higher" else bool(qnn < hybrid),
                }
            )
    return pd.DataFrame(rows).reindex(columns=columns)


def _load_classification_metrics(args: argparse.Namespace) -> pd.DataFrame:
    frames = [
        *(_classification_frame(root, HYBRIDSN_MODEL) for root in args.hybridsn_result_dirs),
        *(_classification_frame(root, QNN_MODEL) for root in args.qnn_result_dirs),
    ]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["dataset", "shot", "seed", "model", "OA", "Macro-F1"])


def _classification_frame(root: Path, model: str) -> pd.DataFrame:
    all_run_paths = sorted((root / "metrics").glob("all_runs*.csv"))
    for path in all_run_paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if {"dataset", "shot", "seed", "OA", "Macro-F1"}.issubset(frame.columns):
            frame = frame[["dataset", "shot", "seed", "OA", "Macro-F1"]].copy()
            frame["model"] = model
            return frame
    rows = []
    for path in sorted((root / "metrics").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if {"dataset", "shot", "seed", "OA", "Macro-F1"}.issubset(payload):
            rows.append({key: payload[key] for key in ("dataset", "shot", "seed", "OA", "Macro-F1")})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["model"] = model
    return frame


def _write_report(
    out: Path,
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    paired: pd.DataFrame,
    margin_summary: pd.DataFrame,
    missing_classification: bool,
) -> None:
    pair_lines = _paired_report_lines(paired)
    distribution_lines = _distribution_report_lines(delta, margin_summary)
    conclusion_lines = _conclusion_lines(delta)
    warning = (
        "Warning: classification metrics were not found; geometry analysis was generated only from features and splits."
        if missing_classification
        else "OA / Macro-F1 were read from available result metrics and merged into the geometry summary."
    )
    lines = [
        "# HybridSN-small vs Spectral QNN Boundary Geometry Analysis",
        "",
        "## 1. 分析目的",
        "",
        "本分析用于验证：QNN 主模型是否在部分 few-shot 设置中改善类别边界。比较对象仅为 HybridSN-small 与 Spectral QNN Gated Fusion + Prototype。",
        "",
        "## 2. 方法",
        "",
        "- 使用相同 split 的 support/train set 计算 class prototypes，test set 不参与 prototype 计算。",
        "- 所有 feature 先做 L2 normalize；prototype 由 support feature 的 normalized mean 得到并再次 normalize。",
        "- 使用 test query 到 prototype 的欧氏距离计算 intra-class distance、inter-class prototype distance、separation ratio 和 prototype margin。",
        "- margin 小于 0 表示 query 更靠近错误 prototype；pred_label/correct 按最近 prototype 判定，不是分类头输出。",
        "",
        warning,
        "",
        "## 3. 主结果表",
        "",
        _markdown_text(summary),
        "",
        "## 4. QNN vs HybridSN 差值",
        "",
        _markdown_text(delta),
        "",
        "## 5. Paired seed 分析",
        "",
        *pair_lines,
        "",
        "## 6. Margin distribution 分析",
        "",
        *distribution_lines,
        "",
        "margin distribution 图保存在 `plots/margin_distribution/`，逐个 dataset / shot 比较两模型的 prototype margin。",
        "",
        "## 7. 初步结论",
        "",
        *conclusion_lines,
        "",
        "本报告不把 geometry 指标解释为普遍证明。实验结果支持时，只能说明在相应 dataset / shot 设置中，Spectral QNN Gated Fusion + Prototype 相比 HybridSN-small 表现出更好的 prototype margin 和 separation ratio，因此可以认为它改善了特征空间中的类别边界。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paired_report_lines(paired: pd.DataFrame) -> list[str]:
    if paired.empty:
        return ["没有形成同时含两个模型的 paired seed。"]
    lines = []
    selected = paired[paired["metric"].isin(PAIR_REPORT_METRICS)]
    for keys, group in selected.groupby(["dataset", "shot"], sort=True):
        items = []
        for metric in PAIR_REPORT_METRICS:
            metric_group = group[group["metric"] == metric]
            items.append(f"{metric}: QNN better in {int(metric_group['qnn_better'].sum())}/{len(metric_group)} seeds")
        lines.append(f"- {keys[0]} {int(keys[1])}-shot: " + "; ".join(items))
    return lines


def _distribution_report_lines(delta: pd.DataFrame, margin_summary: pd.DataFrame) -> list[str]:
    if delta.empty or margin_summary.empty:
        return ["缺少可比较的 margin 明细，未判断分布是否右移。"]
    lines = []
    for keys, group in delta.groupby(["dataset", "shot"], sort=True):
        mean_row = group[group["metric"] == "mean_margin"]
        negative_row = group[group["metric"] == "negative_margin_rate"]
        if mean_row.empty or negative_row.empty:
            continue
        right = float(mean_row["delta"].iloc[0]) > 0
        less_negative = float(negative_row["delta"].iloc[0]) < 0
        if right and less_negative:
            status = "QNN margin 均值右移且 negative margin rate 降低。"
        elif right:
            status = "QNN margin 均值右移，但负 margin 比例没有同步降低。"
        else:
            status = "QNN margin 均值未右移，需结合 paired seed 和分类指标判断。"
        lines.append(f"- {keys[0]} {int(keys[1])}-shot: {status}")
    return lines or ["缺少可比较的 margin delta。"]


def _conclusion_lines(delta: pd.DataFrame) -> list[str]:
    if delta.empty:
        return ["没有同时完成两个模型的设置，暂不形成结论。"]
    lines = []
    for keys, group in delta.groupby(["dataset", "shot"], sort=True):
        metric = {row.metric: row for row in group.itertuples()}
        improved = (
            "separation_ratio" in metric
            and "mean_margin" in metric
            and "negative_margin_rate" in metric
            and metric["separation_ratio"].delta > 0
            and metric["mean_margin"].delta > 0
            and metric["negative_margin_rate"].delta < 0
        )
        if improved:
            text = "separation_ratio 与 mean_margin 提高，negative_margin_rate 降低，geometry 指标支持类别边界改善。"
        else:
            text = "geometry 指标未同时满足 separation、margin 和 negative margin rate 改善，性能变化不应直接归因于更清晰的 prototype geometry。"
        if keys[0] == "salinas" and int(keys[1]) == 10 and not improved:
            text += " 该设置也符合 classical baseline 接近饱和时可能出现负迁移的谨慎解释。"
        lines.append(f"- {keys[0]} {int(keys[1])}-shot: {text}")
    return lines


def _plot_margin_distributions(out: Path, margins: pd.DataFrame) -> None:
    if margins.empty:
        return
    labels = {HYBRIDSN_MODEL: "HybridSN-small", QNN_MODEL: "Spectral QNN Gated Fusion + Prototype"}
    colors = {HYBRIDSN_MODEL: "#2b6cb0", QNN_MODEL: "#c05621"}
    plot_dir = out / "plots" / "margin_distribution"
    for keys, group in margins.groupby(["dataset", "shot"], sort=True):
        fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
        values = [part["margin"].to_numpy() for _, part in group.groupby("model")]
        flat = np.concatenate(values)
        bins = np.histogram_bin_edges(flat, bins="auto")
        if len(bins) < 8:
            bins = 30
        for model in (HYBRIDSN_MODEL, QNN_MODEL):
            part = group[group["model"] == model]
            if part.empty:
                continue
            ax.hist(part["margin"], bins=bins, density=True, alpha=0.38, linewidth=1.0, edgecolor=colors[model], color=colors[model], label=labels[model])
        ax.axvline(0.0, color="#1a202c", linestyle="--", linewidth=1.2, label="margin = 0")
        ax.set_title(f"{keys[0]} {int(keys[1])}-shot prototype margin distribution")
        ax.set_xlabel("Prototype margin")
        ax.set_ylabel("Density")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{keys[0]}_shot{int(keys[1])}_margin_distribution.png")
        plt.close(fig)


def _feature_candidates(args: argparse.Namespace, dataset: str, shot: int, seed: int) -> list[Path]:
    roots = []
    if args.qnn_feature_dir:
        roots.append(Path(args.qnn_feature_dir))
    roots.extend(root / "features" for root in args.qnn_result_dirs)
    patterns = (
        f"{dataset}_shot{shot}_seed{seed}_features.npz",
        f"{dataset}_shot{shot}_seed{seed}_embeddings.npz",
    )
    candidates = []
    for root in roots:
        candidates.extend(root / pattern for pattern in patterns)
    return list(dict.fromkeys(candidates))


def _find_qnn_checkpoint(roots: list[Path], dataset: str, shot: int, seed: int) -> Path | None:
    for root in roots:
        ckpt_dir = root / "checkpoints"
        direct = [
            ckpt_dir / f"{dataset}_spectral_gated_qnn_prototype_shot{shot}_seed{seed}.pt",
            ckpt_dir / f"{dataset}_spectral_gated_qnn_proto_shot{shot}_seed{seed}.pt",
        ]
        for path in direct:
            if path.exists():
                return path
        matches = sorted(ckpt_dir.glob(f"{dataset}*shot{shot}_seed{seed}.pt"))
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


def _config_value(args: argparse.Namespace, config: dict[str, Any], name: str) -> Any:
    return config.get(name, getattr(args, name))


def _load_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_indices(labels: np.ndarray, split: dict[str, Any], split_path: Path) -> None:
    train = set(split["train"])
    test = set(split["test"])
    if train & test:
        raise ValueError(f"Train and test overlap in {split_path}")
    all_indices = np.asarray([*split["train"], *split["test"]], dtype=np.int64)
    if not len(all_indices) or int(all_indices.max()) >= len(labels) or int(all_indices.min()) < 0:
        raise IndexError(f"Split indices do not fit feature labels in {split_path}")


def _first_array(data: Any, names: tuple[str, ...], required: bool = True) -> np.ndarray | None:
    for name in names:
        if name in data:
            return data[name]
    if required:
        return None
    return None


def _pairwise_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    squared = np.maximum(np.sum(left * left, axis=1, keepdims=True) + np.sum(right * right, axis=1)[None, :] - 2 * left @ right.T, 0)
    return np.sqrt(squared)


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def _stat_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _stat_median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else np.nan


def _stat_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if len(values) else np.nan


def _better_name(hybrid: float, qnn: float, direction: str) -> str:
    if np.isclose(hybrid, qnn):
        return "tie"
    if direction == "higher":
        return QNN_MODEL if qnn > hybrid else HYBRIDSN_MODEL
    return QNN_MODEL if qnn < hybrid else HYBRIDSN_MODEL


def _write_margins(out: Path, margins: pd.DataFrame, dataset: str, model: str, shot: int, seed: int) -> None:
    margins.to_csv(out / "margins" / f"{dataset}_{model}_shot{shot}_seed{seed}_margins.csv", index=False)


def _write_markdown(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(_markdown_text(frame) + "\n", encoding="utf-8")


def _markdown_text(frame: pd.DataFrame) -> str:
    return "No completed runs." if frame.empty else frame.round(6).to_markdown(index=False)


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _record_failure(failures: list[dict[str, Any]], dataset: str, shot: int, seed: int, model: str, exc: Exception) -> None:
    failure = {"dataset": dataset, "shot": int(shot), "seed": int(seed), "model": model, "error": repr(exc)}
    failures.append(failure)
    print(f"[WARN] skipped {failure}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze prototype boundary geometry for HybridSN-small versus prototype QNN fusion.")
    parser.add_argument("--datasets", nargs="+", default=["indian_pines", "pavia_university", "salinas"])
    parser.add_argument("--shots", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--hybridsn_result_dir", default="result/hybridsn_small_fewshot_3datasets")
    parser.add_argument("--hybridsn_result_dirs", nargs="+", default=None)
    parser.add_argument("--qnn_result_dir", default="result/hybridsn_small_spectral_qnn_gated_proto_tuning_3datasets")
    parser.add_argument("--qnn_result_dirs", nargs="+", default=None)
    parser.add_argument("--qnn_feature_dir", default=None)
    parser.add_argument("--split_dir", default="result/hybridsn_small_fewshot_3datasets/split_indices")
    parser.add_argument("--split_dirs", nargs="+", default=None)
    parser.add_argument("--output_dir", default="result/boundary_geometry_hybridsn_vs_qnn")
    parser.add_argument("--low_margin_threshold", type=float, default=0.05)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--patch_size", type=int, default=19)
    parser.add_argument("--pca_bands", type=int, default=30)
    parser.add_argument("--conv3d_channels", nargs=3, type=int, default=[8, 16, 16])
    parser.add_argument("--conv2d_channels", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--qnn_feature_batch_size", type=int, default=128)
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--qnn_layers", type=int, default=1)
    parser.add_argument("--entanglement", default="linear")
    parser.add_argument("--gate_mode", choices=["scalar", "classwise"], default="classwise")
    parser.add_argument("--backend", default="lightning.qubit")
    parser.add_argument("--diff_method", default="adjoint")
    parser.add_argument("--angle_scale", type=float, default=float(np.pi))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args_parser = parser
    original_parse_args = args_parser.parse_args

    def parse_args(*parse_args: Any, **parse_kwargs: Any) -> argparse.Namespace:
        args = original_parse_args(*parse_args, **parse_kwargs)
        args.hybridsn_result_dirs = [Path(path) for path in (args.hybridsn_result_dirs or [args.hybridsn_result_dir])]
        args.qnn_result_dirs = [Path(path) for path in (args.qnn_result_dirs or [args.qnn_result_dir])]
        args.split_dirs = [Path(path) for path in (args.split_dirs or [args.split_dir])]
        return args

    args_parser.parse_args = parse_args  # type: ignore[method-assign]
    return args_parser


if __name__ == "__main__":
    main()
