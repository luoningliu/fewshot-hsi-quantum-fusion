from __future__ import annotations

from src.models.quantum.qnn_classifier import QNNClassifier


class NoEntanglementQNNClassifier(QNNClassifier):
    def __init__(self, *args, **kwargs):
        kwargs["entanglement"] = "none"
        super().__init__(*args, **kwargs)

