from __future__ import annotations

import math

import torch
from torch import nn

from src.models.quantum.qnn_layer import PennyLaneQNNLayer


class QNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        qubits: int = 6,
        layers: int = 2,
        entanglement: str = "ring",
        trainable_qnn: bool = True,
        backend: str = "lightning.qubit",
        diff_method: str = "adjoint",
        normalize_input: bool = True,
        angle_scale: float = math.pi,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if normalize_input else nn.Identity()
        self.project = nn.Linear(input_dim, qubits)
        self.angle_scale = float(angle_scale)
        self.qnn = PennyLaneQNNLayer(
            qubits=qubits,
            layers=layers,
            entanglement=entanglement,
            trainable=trainable_qnn,
            backend=backend,
            diff_method=diff_method,
        )
        self.feature_dim = qubits
        self.classifier = nn.Linear(qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.angle_scale * torch.tanh(self.project(x))
        return self.qnn(x)
