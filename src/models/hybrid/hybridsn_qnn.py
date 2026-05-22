from __future__ import annotations

import torch
from torch import nn

from src.models.classical.hybridsn import HybridSNEncoder
from src.models.quantum.qnn_classifier import QNNClassifier


class HybridSNQNN(nn.Module):
    def __init__(
        self,
        pca_channels: int,
        num_classes: int,
        embedding_dim: int = 128,
        qubits: int = 6,
        backend: str = "lightning.qubit",
        diff_method: str = "adjoint",
        normalize_input: bool = True,
        angle_scale: float = 3.141592653589793,
    ):
        super().__init__()
        self.encoder = HybridSNEncoder(pca_channels=pca_channels, embedding_dim=embedding_dim)
        self.classifier = QNNClassifier(
            input_dim=embedding_dim,
            num_classes=num_classes,
            qubits=qubits,
            backend=backend,
            diff_method=diff_method,
            normalize_input=normalize_input,
            angle_scale=angle_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
