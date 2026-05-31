from __future__ import annotations

import torch
from torch import nn


class _SqueezeExcite2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(int(channels) // int(reduction), 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class _Residual3DBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class SSRNLite(nn.Module):
    """Compact spectral-spatial residual baseline inspired by SSRN-style HSI classifiers."""

    def __init__(self, pca_channels: int, num_classes: int, patch_size: int = 19, width: int = 16):
        super().__init__()
        self.pca_channels = int(pca_channels)
        self.patch_size = int(patch_size)
        self.stem = nn.Sequential(
            nn.Conv3d(1, width, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True),
        )
        self.spectral_residual = _Residual3DBlock(width)
        self.spatial_residual = _Residual3DBlock(width)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(width, num_classes),
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
        assert x.ndim == 5, f"SSRNLite expected [B, 1, D, H, W], got {tuple(x.shape)}"
        assert x.shape[1] == 1, f"SSRNLite expected channel=1, got {tuple(x.shape)}"
        assert x.shape[2] == self.pca_channels, (
            f"SSRNLite expected pca_channels={self.pca_channels}, got {tuple(x.shape)}"
        )
        assert x.shape[3] == self.patch_size and x.shape[4] == self.patch_size, (
            f"SSRNLite expected patch_size={self.patch_size}, got {tuple(x.shape)}"
        )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        x = self.stem(x)
        x = self.spectral_residual(x)
        x = self.spatial_residual(x)
        return self.head(x)


class SpectralFormerLite(nn.Module):
    """Spectral-token Transformer baseline using the center pixel plus local spatial context."""

    def __init__(
        self,
        pca_channels: int,
        num_classes: int,
        patch_size: int = 19,
        d_model: int = 48,
        num_heads: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        self.pca_channels = int(pca_channels)
        self.patch_size = int(patch_size)
        self.band_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.pca_channels + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.spatial_pool = nn.Sequential(
            nn.Conv2d(self.pca_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(0.35),
            nn.Linear(d_model * 2, num_classes),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _prepare_patch(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            assert x.shape[1] == 1, f"SpectralFormerLite expected channel=1, got {tuple(x.shape)}"
            x = x.squeeze(1).permute(0, 2, 3, 1)
        assert x.ndim == 4, f"SpectralFormerLite expected [B, H, W, C], got {tuple(x.shape)}"
        assert x.shape[-1] == self.pca_channels, (
            f"SpectralFormerLite expected pca_channels={self.pca_channels}, got {tuple(x.shape)}"
        )
        assert x.shape[1] == self.patch_size and x.shape[2] == self.patch_size, (
            f"SpectralFormerLite expected patch_size={self.patch_size}, got {tuple(x.shape)}"
        )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_patch(x)
        center = x[:, self.patch_size // 2, self.patch_size // 2, :].unsqueeze(-1)
        tokens = self.band_embed(center)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        spectral = self.encoder(tokens)[:, 0]
        spatial = self.spatial_pool(x.permute(0, 3, 1, 2)).flatten(1)
        return self.classifier(torch.cat([spectral, spatial], dim=1))


class DBDALite(nn.Module):
    """Dual-branch dense attention baseline with spectral and spatial streams."""

    def __init__(self, pca_channels: int, num_classes: int, patch_size: int = 19, width: int = 48):
        super().__init__()
        self.pca_channels = int(pca_channels)
        self.patch_size = int(patch_size)
        self.spectral_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(self.pca_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            _SqueezeExcite2d(width),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(width * 2, width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(width, num_classes),
        )

    def _prepare_patch(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            assert x.shape[1] == 1, f"DBDALite expected channel=1, got {tuple(x.shape)}"
            x = x.squeeze(1).permute(0, 2, 3, 1)
        assert x.ndim == 4, f"DBDALite expected [B, H, W, C], got {tuple(x.shape)}"
        assert x.shape[-1] == self.pca_channels, (
            f"DBDALite expected pca_channels={self.pca_channels}, got {tuple(x.shape)}"
        )
        assert x.shape[1] == self.patch_size and x.shape[2] == self.patch_size, (
            f"DBDALite expected patch_size={self.patch_size}, got {tuple(x.shape)}"
        )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_patch(x)
        center = x[:, self.patch_size // 2, self.patch_size // 2, :].unsqueeze(1)
        spectral = self.spectral_branch(center)
        spatial = self.spatial_branch(x.permute(0, 3, 1, 2))
        return self.classifier(torch.cat([spectral, spatial], dim=1))
