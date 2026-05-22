from __future__ import annotations

import torch
from torch import nn

from src.models.quantum.qnn_classifier import QNNClassifier


class ResidualQNNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, **qnn_kwargs):
        super().__init__()
        self.qnn_head = QNNClassifier(input_dim, num_classes, **qnn_kwargs)
        self.linear_head = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_head(x) + self.qnn_head(x)
