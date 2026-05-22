from __future__ import annotations

import torch
from torch import nn

from src.models.quantum.qnn_classifier import QNNClassifier


class GatedResidualQNNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, gate_mode: str = "scalar", **qnn_kwargs):
        super().__init__()
        self.qnn_head = QNNClassifier(input_dim, num_classes, **qnn_kwargs)
        self.linear_head = nn.Linear(input_dim, num_classes)
        self.gate_mode = gate_mode
        if gate_mode == "scalar":
            self.gate = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1), nn.Sigmoid())
        elif gate_mode == "classwise":
            self.gate = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, num_classes), nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_logits = self.linear_head(x)
        qnn_logits = self.qnn_head(x)
        return linear_logits + self.gate(x) * qnn_logits
