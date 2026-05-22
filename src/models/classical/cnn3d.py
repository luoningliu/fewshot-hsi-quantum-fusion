from __future__ import annotations

import torch
from torch import nn


class CNN3D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
        assert x.ndim == 5, f"3D-CNN expected [B, 1, D, H, W], got {tuple(x.shape)}"
        assert x.shape[1] == 1, f"3D-CNN expected channel=1, got {tuple(x.shape)}"
        assert x.shape[2] == self.input_channels, (
            f"3D-CNN expected spectral depth={self.input_channels}, got {tuple(x.shape)}"
        )
        assert x.shape[-1] == x.shape[-2], f"3D-CNN expected square spatial patch, got {tuple(x.shape)}"
        x = self.features(x)
        return self.classifier(x.flatten(1))
