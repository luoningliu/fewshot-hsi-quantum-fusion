from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class HSIRawDataset:
    dataset_id: str
    display_name: str
    cube: np.ndarray
    gt: np.ndarray
    class_names: dict[int, str]
    background_label: int


def load_hsi_mat(config: dict[str, Any]) -> HSIRawDataset:
    data_path = Path(config["data_path"])
    gt_path = Path(config["gt_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"HSI data file not found: {data_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"HSI ground-truth file not found: {gt_path}")

    data_mat = sio.loadmat(data_path)
    gt_mat = sio.loadmat(gt_path)
    data_key = config["data_key"]
    gt_key = config["gt_key"]
    if data_key not in data_mat:
        raise KeyError(f"Key {data_key!r} not found in {data_path}. Available: {_public_keys(data_mat)}")
    if gt_key not in gt_mat:
        raise KeyError(f"Key {gt_key!r} not found in {gt_path}. Available: {_public_keys(gt_mat)}")

    cube = np.asarray(data_mat[data_key], dtype=np.float32)
    gt = np.asarray(gt_mat[gt_key])
    if cube.ndim != 3:
        raise ValueError(f"Expected cube [H, W, B], got shape {cube.shape}")
    if gt.shape != cube.shape[:2]:
        raise ValueError(f"GT shape {gt.shape} does not match cube spatial shape {cube.shape[:2]}")

    class_names = {int(k): str(v) for k, v in config["class_names"].items()}
    return HSIRawDataset(
        dataset_id=str(config["dataset_id"]),
        display_name=str(config["display_name"]),
        cube=cube,
        gt=gt.astype(np.int64),
        class_names=class_names,
        background_label=int(config.get("background_label", 0)),
    )


def _public_keys(mat: dict[str, Any]) -> list[str]:
    return [key for key in mat.keys() if not key.startswith("__")]

