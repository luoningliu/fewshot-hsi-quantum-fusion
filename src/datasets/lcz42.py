from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LCZ42Raw:
    x: np.ndarray
    y: np.ndarray
    city: np.ndarray
    bands: list[str] | None = None


def load_lcz42(input_config: dict[str, Any]) -> LCZ42Raw:
    input_type = input_config["type"]
    if input_type == "npz":
        return _load_npz_dataset(Path(input_config["path"]))
    if input_type == "manifest":
        return _load_manifest_dataset(
            manifest_path=Path(input_config["manifest_path"]),
            base_dir=Path(input_config.get("base_dir", ".")),
        )
    raise ValueError(f"Unsupported input.type: {input_type!r}. Expected 'npz' or 'manifest'.")


def _load_npz_dataset(path: Path) -> LCZ42Raw:
    if not path.exists():
        raise FileNotFoundError(
            f"Input NPZ not found: {path}. Place LCZ42 data there or edit configs/experiments/stage1_data_protocol.yaml."
        )
    data = np.load(path, allow_pickle=True)
    required = {"x", "y", "city"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
    bands = data["bands"].astype(str).tolist() if "bands" in data.files else None
    return LCZ42Raw(x=data["x"], y=data["y"], city=data["city"].astype(str), bands=bands)


def _load_manifest_dataset(manifest_path: Path, base_dir: Path) -> LCZ42Raw:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"path", "city", "label"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing required columns: {sorted(missing)}")

    patches = []
    for raw_path in manifest["path"]:
        patch_path = Path(raw_path)
        if not patch_path.is_absolute():
            patch_path = base_dir / patch_path
        patches.append(_load_patch(patch_path))

    return LCZ42Raw(
        x=np.stack(patches, axis=0),
        y=manifest["label"].to_numpy(),
        city=manifest["city"].astype(str).to_numpy(),
        bands=None,
    )


def _load_patch(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Patch file not found: {path}")
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        for key in ("x", "arr_0", "patch"):
            if key in data.files:
                return data[key]
        raise ValueError(f"No patch array found in {path}; expected key x, arr_0, or patch.")
    raise ValueError(f"Unsupported patch file type: {path.suffix}. Use .npy or .npz.")

