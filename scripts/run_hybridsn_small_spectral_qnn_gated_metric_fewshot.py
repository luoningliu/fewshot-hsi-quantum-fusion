from __future__ import annotations

import torch
from torch import nn

from src.models.quantum import DataReuploadingQNNClassifier, QNNClassifier


class SpectralQNNGatedMetricFusion(nn.Module):
    """Frozen HybridSN feature plus center-spectrum QNN gated residual head.

    This lightweight module is imported by the fair-control and margin-analysis
    scripts. It intentionally contains only the model definition; training logic
    lives in the caller scripts.
    """

    def __init__(
        self,
        embedding_dim: int,
        spectral_dim: int,
        num_classes: int,
        gate_mode: str = "classwise",
        qnn_variant: str = "standard",
        residual_scale_mode: str = "none",
        residual_alpha_init: float = -4.0,
        **qnn_kwargs,
    ):
        super().__init__()
        self.base_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes))
        if qnn_variant == "standard":
            self.qnn_head = QNNClassifier(spectral_dim, num_classes, **qnn_kwargs)
        elif qnn_variant == "reupload_multiobs":
            qnn_kwargs.pop("trainable_qnn", None)
            self.qnn_head = DataReuploadingQNNClassifier(spectral_dim, num_classes, **qnn_kwargs)
        else:
            raise ValueError(f"Unsupported qnn_variant: {qnn_variant}")
        self.qnn_variant = qnn_variant
        if residual_scale_mode not in {"none", "learnable_sigmoid"}:
            raise ValueError(f"Unsupported residual_scale_mode: {residual_scale_mode}")
        self.residual_scale_mode = residual_scale_mode
        if residual_scale_mode == "learnable_sigmoid":
            self.residual_alpha = nn.Parameter(torch.tensor(float(residual_alpha_init)))
        else:
            self.register_buffer("residual_alpha", torch.tensor(0.0), persistent=False)
        self.register_buffer("residual_warmup_factor", torch.tensor(1.0))
        gate_dim = 1 if gate_mode == "scalar" else num_classes
        if gate_mode not in {"scalar", "classwise"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        self.gate = nn.Sequential(
            nn.LayerNorm(embedding_dim + spectral_dim),
            nn.Linear(embedding_dim + spectral_dim, gate_dim),
            nn.Sigmoid(),
        )
        self.feature_norm = nn.LayerNorm(embedding_dim + int(getattr(self.qnn_head, "feature_dim", self.qnn_head.qnn.qubits)))

    def residual_scale(self) -> torch.Tensor:
        if self.residual_scale_mode == "learnable_sigmoid":
            return self.residual_warmup_factor * torch.sigmoid(self.residual_alpha)
        return self.residual_alpha.new_tensor(1.0)

    def set_residual_warmup_factor(self, factor: float) -> None:
        clipped = max(0.0, min(1.0, float(factor)))
        self.residual_warmup_factor.fill_(clipped)

    def spectral_features(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.qnn_head.forward_features(spectrum)

    def fused_features(self, z: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        q = self.spectral_features(spectrum)
        return self.feature_norm(torch.cat([z, q], dim=1))

    def forward(self, z: torch.Tensor, spectrum: torch.Tensor, return_aux: bool = False):
        base_logits = self.base_head(z)
        q = self.spectral_features(spectrum)
        spectral_logits = self.qnn_head.classifier(q)
        gate = self.gate(torch.cat([z, spectrum], dim=1))
        scale = self.residual_scale()
        logits = base_logits + scale * gate * spectral_logits
        if return_aux:
            return logits, {
                "gate": gate,
                "residual_scale": scale.detach(),
                "base_logits": base_logits,
                "spectral_logits": spectral_logits,
                "spectral_feature": q,
            }
        return logits
