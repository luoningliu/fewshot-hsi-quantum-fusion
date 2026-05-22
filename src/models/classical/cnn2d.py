from __future__ import annotations

import torch
from torch import nn


class CNN2D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.input_channels = input_channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] != self.input_channels and x.shape[-1] == self.input_channels:
            x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        return self.classifier(x.flatten(1))
