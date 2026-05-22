from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(**kwargs) -> RandomForestClassifier:
    params = {"n_estimators": 200, "random_state": 42, "n_jobs": -1, "class_weight": None}
    params.update(kwargs)
    return RandomForestClassifier(**params)

