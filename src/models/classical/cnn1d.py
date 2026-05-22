from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        assert x.ndim == 3, f"1D-CNN expected [B, 1, L], got {tuple(x.shape)}"
        assert x.shape[1] == 1, f"1D-CNN expected channel=1, got {tuple(x.shape)}"
        assert x.shape[2] == self.input_channels, (
            f"1D-CNN expected spectral length={self.input_channels}, got {tuple(x.shape)}"
        )
        x = self.features(x)
        return self.classifier(x.flatten(1))
