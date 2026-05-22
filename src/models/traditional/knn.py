from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier


def build_knn(**kwargs) -> KNeighborsClassifier:
    params = {"n_neighbors": 5, "weights": "distance"}
    params.update(kwargs)
    return KNeighborsClassifier(**params)

