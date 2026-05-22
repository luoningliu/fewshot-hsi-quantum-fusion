from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def make_fewshot_split(
    labels: np.ndarray,
    shot: int,
    seed: int,
    num_classes: int,
    max_val_per_class: int = 10,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    skipped: list[dict[str, int]] = []
    per_class: dict[int, dict[str, int]] = {}
    for cls in range(int(num_classes)):
        class_indices = np.flatnonzero(labels == cls)
        rng.shuffle(class_indices)
        if len(class_indices) <= int(shot):
            skipped.append({"class_id": cls, "available": int(len(class_indices)), "required_train": int(shot)})
            continue
        train_cls = class_indices[: int(shot)]
        remaining = class_indices[int(shot) :]
        val_count = min(int(max_val_per_class), max(1, int(np.floor(len(remaining) * float(val_fraction)))))
        if len(remaining) <= val_count:
            val_count = max(1, len(remaining) // 2)
        val_cls = remaining[:val_count]
        test_cls = remaining[val_count:]
        train.extend(train_cls.tolist())
        validation.extend(val_cls.tolist())
        test.extend(test_cls.tolist())
        per_class[cls] = {
            "train": int(len(train_cls)),
            "validation": int(len(val_cls)),
            "test": int(len(test_cls)),
            "available": int(len(class_indices)),
        }
    split = {
        "train": sorted(int(i) for i in train),
        "validation": sorted(int(i) for i in validation),
        "test": sorted(int(i) for i in test),
        "shot": int(shot),
        "seed": int(seed),
        "num_classes": int(num_classes),
        "per_class": per_class,
        "skipped": skipped,
    }
    assert_disjoint(split)
    return split


def assert_disjoint(split: dict[str, Any]) -> None:
    train = set(split["train"])
    validation = set(split["validation"])
    test = set(split["test"])
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def save_split(path: str | Path, split: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
