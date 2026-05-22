from __future__ import annotations

import torch
from torch import nn


class HybridSNEncoder(nn.Module):
    def __init__(
        self,
        pca_channels: int = 30,
        embedding_dim: int = 128,
        patch_size: int = 9,
        architecture: str = "hybridsn_base",
    ):
        super().__init__()
        self.pca_channels = int(pca_channels)
        self.patch_size = int(patch_size)
        self.architecture = str(architecture)
        if self.architecture == "hybridsn_base":
            self.conv3d = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
            )
        elif self.architecture == "hybridsn_small":
            self.conv3d = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=1),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported HybridSN architecture: {architecture}")

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.pca_channels, self.patch_size, self.patch_size)
            features = self._forward_3d(dummy)
            conv2d_in_channels = features.shape[1] * features.shape[2]

        self.conv2d = nn.Sequential(
            nn.Conv2d(conv2d_in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(64, embedding_dim)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
        assert x.dim() == 5, f"HybridSN expected [B, 1, D, H, W], got {tuple(x.shape)}"
        assert x.shape[1] == 1, f"HybridSN expected channel=1, got {tuple(x.shape)}"
        assert x.shape[2] in [15, 30, 50], f"Unexpected spectral/PCA depth: {tuple(x.shape)}"
        assert x.shape[3] == self.patch_size and x.shape[4] == self.patch_size, (
            f"HybridSN expected patch_size={self.patch_size}, got {tuple(x.shape)}"
        )
        return x

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3d(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        x = self._forward_3d(x)
        batch, channels, spectral, height, width = x.shape
        x = x.reshape(batch, channels * spectral, height, width)
        x = self.conv2d(x).flatten(1)
        return self.projection(x)


class HybridSN(nn.Module):
    def __init__(
        self,
        pca_channels: int,
        num_classes: int,
        embedding_dim: int = 128,
        patch_size: int = 9,
        architecture: str = "hybridsn_base",
        dropout: float = 0.4,
    ):
        super().__init__()
        self.encoder = HybridSNEncoder(
            pca_channels=pca_channels,
            embedding_dim=embedding_dim,
            patch_size=patch_size,
            architecture=architecture,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


class HybridSNSmall(nn.Module):
    """Lightweight HybridSN preserving 3D spectral-spatial then 2D spatial hierarchy."""

    def __init__(
        self,
        pca_channels: int,
        num_classes: int,
        patch_size: int = 19,
        conv3d_channels: tuple[int, int, int] = (8, 16, 16),
        conv2d_channels: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.pca_channels = int(pca_channels)
        self.patch_size = int(patch_size)
        c1, c2, c3 = [int(c) for c in conv3d_channels]
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.Conv3d(c1, c2, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
            nn.Conv3d(c2, c3, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.pca_channels, self.patch_size, self.patch_size)
            out = self.conv3d(dummy)
            conv2d_in_channels = out.shape[1] * out.shape[2]
        self.conv2d = nn.Sequential(
            nn.Conv2d(conv2d_in_channels, int(conv2d_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(conv2d_channels)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(conv2d_channels), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
        assert x.dim() == 5, f"HybridSNSmall expected [B, 1, D, H, W], got {tuple(x.shape)}"
        assert x.shape[1] == 1, f"HybridSNSmall expected channel=1, got {tuple(x.shape)}"
        assert x.shape[2] == self.pca_channels, (
            f"HybridSNSmall expected pca_channels={self.pca_channels}, got {tuple(x.shape)}"
        )
        assert x.shape[3] == self.patch_size and x.shape[4] == self.patch_size, (
            f"HybridSNSmall expected patch_size={self.patch_size}, got {tuple(x.shape)}"
        )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        x = self.conv3d(x)
        batch, channels, spectral, height, width = x.shape
        x = x.reshape(batch, channels * spectral, height, width)
        x = self.conv2d(x).flatten(1)
        return self.classifier(x)
