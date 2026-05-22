from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.analysis.metrics import class_distribution, write_json
from src.datasets.hsi_dataset import load_hsi_mat
from src.datasets.patch_extraction import extract_center_patches
from src.datasets.split import assert_split_integrity, make_seed_split, write_split
from src.utils.config import load_yaml


def run_stage1(config_path: str | Path) -> None:
    experiment_config = load_yaml(config_path)
    output_root = Path(experiment_config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_root / "config.yaml")

    summaries = []
    for dataset_config_path in experiment_config["datasets"]:
        dataset_config = load_yaml(dataset_config_path)
        summary = process_hsi_dataset(dataset_config, output_root)
        summaries.append(summary)

    _write_stage_report(output_root / "report.md", summaries)


def process_hsi_dataset(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    raw = load_hsi_mat(config)
    cube = raw.cube
    gt = raw.gt
    background_label = raw.background_label
    mask = gt != background_label
    rows, cols = np.nonzero(mask)
    original_labels = gt[rows, cols].astype(np.int64)
    labels = original_labels - 1
    if labels.min() < 0:
        raise ValueError(f"{raw.dataset_id}: labels must be positive after removing background.")
    if labels.max() + 1 != int(config["num_classes"]):
        raise ValueError(
            f"{raw.dataset_id}: expected {config['num_classes']} classes, observed labels 0..{labels.max()}."
        )

    flat_cube = cube.reshape(-1, cube.shape[-1]).astype(np.float32)
    flat_mask = mask.reshape(-1)
    fit_pixels = flat_cube[flat_mask]
    band_mean = fit_pixels.mean(axis=0, dtype=np.float64)
    band_std = fit_pixels.std(axis=0, dtype=np.float64)
    band_std = np.where(band_std == 0, 1.0, band_std)
    cube_norm = ((cube - band_mean) / band_std).astype(np.float32)

    pca_components = int(config["pca_components"])
    pca = PCA(n_components=pca_components, whiten=False, random_state=int(config["split"]["seed"]))
    pca.fit(cube_norm.reshape(-1, cube_norm.shape[-1])[flat_mask])
    reduced_flat = pca.transform(cube_norm.reshape(-1, cube_norm.shape[-1])).astype(np.float32)
    cube_pca = reduced_flat.reshape(cube.shape[0], cube.shape[1], pca_components)

    patch_size = int(config["patch_size"])
    patches = extract_center_patches(cube_pca, rows, cols, patch_size)
    spectral_vectors = cube_pca[rows, cols, :].astype(np.float32)

    split = make_seed_split(labels, seed=int(config["split"]["seed"]), split_config=config["split"])
    assert_split_integrity(split, labels)

    processed_path = Path(config["output"]["processed_npz"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_path,
        x=patches,
        spectral=spectral_vectors,
        y=labels.astype(np.int64),
        y_original=original_labels.astype(np.int64),
        rows=rows.astype(np.int64),
        cols=cols.astype(np.int64),
        gt=gt.astype(np.int64),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        band_mean=band_mean.astype(np.float32),
        band_std=band_std.astype(np.float32),
        class_names=np.asarray([raw.class_names[i] for i in range(1, int(config["num_classes"]) + 1)]),
    )

    split_path = Path(config["output"]["split_json"])
    write_split(split_path, split)
    result_split_path = output_root / f"{raw.dataset_id}_split_seed{config['split']['seed']}.json"
    write_split(result_split_path, split)

    class_names_zero_based = {label - 1: name for label, name in raw.class_names.items()}
    dist = class_distribution(labels, class_names_zero_based)
    dist_path = output_root / f"{raw.dataset_id}_class_distribution.csv"
    dist.to_csv(dist_path, index=False)

    band_stats = pd.DataFrame(
        {
            "band": np.arange(cube.shape[-1]),
            "mean": band_mean,
            "std": band_std,
        }
    )
    band_stats_path = output_root / f"{raw.dataset_id}_band_statistics.csv"
    band_stats.to_csv(band_stats_path, index=False)

    write_json(output_root / f"{raw.dataset_id}_band_mean.json", {str(i): float(v) for i, v in enumerate(band_mean)})
    write_json(output_root / f"{raw.dataset_id}_band_std.json", {str(i): float(v) for i, v in enumerate(band_std)})

    summary = {
        "dataset_id": raw.dataset_id,
        "display_name": raw.display_name,
        "raw_cube_shape": list(cube.shape),
        "gt_shape": list(gt.shape),
        "num_labeled_samples": int(len(labels)),
        "num_classes": int(config["num_classes"]),
        "patch_shape": list(patches.shape[1:]),
        "spectral_shape": list(spectral_vectors.shape[1:]),
        "pca_components": pca_components,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "split_counts": {
            "train": len(split["train"]),
            "validation": len(split["validation"]),
            "test": len(split["test"]),
        },
        "processed_npz": str(processed_path),
        "split_json": str(split_path),
        "result_split_json": str(result_split_path),
        "class_distribution_csv": str(dist_path),
        "band_statistics_csv": str(band_stats_path),
    }
    write_json(output_root / f"{raw.dataset_id}_statistics.json", summary)
    return summary


def _write_stage_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 1 Data Protocol Report",
        "",
        "| Dataset | Raw shape | Samples | Patch shape | PCA | EVR sum | Train | Val | Test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {display_name} | {raw_cube_shape} | {num_labeled_samples} | {patch_shape} | "
            "{pca_components} | {pca_explained_variance_ratio_sum:.6f} | {train} | {validation} | {test} |".format(
                **item,
                train=item["split_counts"]["train"],
                validation=item["split_counts"]["validation"],
                test=item["split_counts"]["test"],
            )
        )
    lines.extend(
        [
            "",
            "## Fixed Protocol",
            "",
            "- Background label is ignored.",
            "- Labels are stored as zero-based class ids in processed files.",
            "- Per-band normalization and PCA are fit on all non-background pixels.",
            "- Patch extraction uses reflect padding.",
            "- Main split is stratified train/validation/test = 10%/10%/80% with seed 42.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 HSI data protocol.")
    parser.add_argument("--config", required=True, help="Experiment YAML path.")
    args = parser.parse_args()
    run_stage1(args.config)


if __name__ == "__main__":
    main()

