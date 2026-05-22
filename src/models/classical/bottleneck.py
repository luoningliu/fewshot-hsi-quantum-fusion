from __future__ import annotations

import torch
from torch import nn


class BottleneckClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, bottleneck_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

