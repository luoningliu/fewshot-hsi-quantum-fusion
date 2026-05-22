from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_holdout_split(y: np.ndarray, seed: int, ratios: dict[str, float]) -> dict[str, list[int]]:
    indices = np.arange(len(y))
    train_ratio = float(ratios["train"])
    validation_ratio = float(ratios["validation"])
    test_ratio = float(ratios["test"])
    total = train_ratio + validation_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"holdout_split ratios must sum to 1.0, got {total}")

    train_idx, temp_idx, _, y_temp = train_test_split(
        indices,
        y,
        train_size=train_ratio,
        random_state=seed,
        stratify=y,
    )
    validation_fraction_of_temp = validation_ratio / (validation_ratio + test_ratio)
    validation_idx, test_idx = train_test_split(
        temp_idx,
        train_size=validation_fraction_of_temp,
        random_state=seed,
        stratify=y_temp,
    )
    return {
        "train": sorted(train_idx.astype(int).tolist()),
        "validation": sorted(validation_idx.astype(int).tolist()),
        "test": sorted(test_idx.astype(int).tolist()),
    }


def make_seed_split(y: np.ndarray, seed: int, split_config: dict[str, float]) -> dict[str, object]:
    split = make_holdout_split(
        y,
        seed=seed,
        ratios={
            "train": float(split_config["train"]),
            "validation": float(split_config["validation"]),
            "test": float(split_config["test"]),
        },
    )
    split["seed"] = int(seed)
    split["ratios"] = {
        "train": float(split_config["train"]),
        "validation": float(split_config["validation"]),
        "test": float(split_config["test"]),
    }
    return split


def make_kfold_splits(y: np.ndarray, seed: int, num_folds: int) -> list[dict[str, list[int]]]:
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y):
        folds.append(
            {
                "train": sorted(train_idx.astype(int).tolist()),
                "test": sorted(test_idx.astype(int).tolist()),
            }
        )
    return folds


def assert_split_integrity(split: dict[str, object], y: np.ndarray) -> None:
    train = set(split["train"])
    validation = set(split["validation"])
    test = set(split["test"])
    if train & validation or train & test or validation & test:
        raise ValueError("Split leakage detected: train/validation/test overlap.")
    merged = train | validation | test
    if merged != set(range(len(y))):
        raise ValueError("Split does not cover every sample exactly once.")
    for name in ("train", "validation", "test"):
        labels = set(int(v) for v in np.unique(y[np.asarray(split[name], dtype=np.int64)]))
        expected = set(int(v) for v in np.unique(y))
        if labels != expected:
            raise ValueError(f"Split {name!r} is missing labels: {sorted(expected - labels)}")


def write_split(path: str | Path, split: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

