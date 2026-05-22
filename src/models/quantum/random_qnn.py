from __future__ import annotations

from src.models.quantum.qnn_classifier import QNNClassifier


class RandomQNNClassifier(QNNClassifier):
    def __init__(self, *args, **kwargs):
        kwargs["trainable_qnn"] = False
        super().__init__(*args, **kwargs)

