from __future__ import annotations

import math

import torch
from torch import nn

from src.models.quantum.advanced_qnn import DataReuploadingMultiObservableQNNLayer


class DataReuploadingQNNClassifier(nn.Module):
    """QNN classifier with explicit data re-uploading and multi-observable readout.

    The interface mirrors QNNClassifier so it can be swapped into the existing
    spectral gated fusion branch. Each variational layer re-encodes the projected
    spectral angles, then reads both single-qubit Z and neighboring ZZ
    observables. The wider readout gives the downstream gated fusion head a
    richer spectral quantum feature without increasing the input qubit count.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        qubits: int = 6,
        layers: int = 2,
        entanglement: str = "linear",
        backend: str = "lightning.qubit",
        diff_method: str = "adjoint",
        normalize_input: bool = True,
        angle_scale: float = math.pi,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if normalize_input else nn.Identity()
        self.project = nn.Linear(input_dim, qubits)
        self.angle_scale = float(angle_scale)
        self.qnn = DataReuploadingMultiObservableQNNLayer(
            qubits=qubits,
            layers=layers,
            entanglement=entanglement,
            backend=backend,
            diff_method=diff_method,
        )
        self.feature_dim = self.qnn.output_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.angle_scale * torch.tanh(self.project(x))
        return self.qnn(x)
