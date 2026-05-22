from __future__ import annotations

from sklearn.svm import SVC


def build_svm_rbf(**kwargs) -> SVC:
    params = {"kernel": "rbf", "C": 100.0, "gamma": "scale", "class_weight": None}
    params.update(kwargs)
    return SVC(**params)

