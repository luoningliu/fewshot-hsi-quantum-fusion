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
        gate_context_mode: str = "none",
        high_confidence_guard_mode: str = "none",
        guard_floor: float = 0.05,
        guard_tau: float = 0.35,
        guard_temperature: float = 0.08,
        base_logit_mode: str = "learned_head",
        **qnn_kwargs,
    ):
        super().__init__()
        if base_logit_mode not in {"learned_head", "pretrained"}:
            raise ValueError(f"Unsupported base_logit_mode: {base_logit_mode}")
        self.base_logit_mode = base_logit_mode
        self.base_head = None
        if base_logit_mode == "learned_head":
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
        if gate_context_mode not in {"none", "base_confidence_margin"}:
            raise ValueError(f"Unsupported gate_context_mode: {gate_context_mode}")
        self.gate_context_mode = gate_context_mode
        if high_confidence_guard_mode not in {"none", "margin_suppression"}:
            raise ValueError(f"Unsupported high_confidence_guard_mode: {high_confidence_guard_mode}")
        if not 0.0 <= guard_floor <= 1.0:
            raise ValueError(f"guard_floor must be in [0, 1], got {guard_floor}")
        if guard_temperature <= 0.0:
            raise ValueError(f"guard_temperature must be > 0, got {guard_temperature}")
        self.high_confidence_guard_mode = high_confidence_guard_mode
        self.guard_floor = float(guard_floor)
        self.guard_tau = float(guard_tau)
        self.guard_temperature = float(guard_temperature)
        gate_context_dim = 2 if gate_context_mode == "base_confidence_margin" else 0
        self.gate = nn.Sequential(
            nn.LayerNorm(embedding_dim + spectral_dim + gate_context_dim),
            nn.Linear(embedding_dim + spectral_dim + gate_context_dim, gate_dim),
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

    def forward(
        self,
        z: torch.Tensor,
        spectrum: torch.Tensor,
        return_aux: bool = False,
        base_logits: torch.Tensor | None = None,
    ):
        if self.base_logit_mode == "pretrained":
            if base_logits is None:
                raise ValueError("base_logits is required when base_logit_mode='pretrained'")
            base_logits = base_logits.detach()
        else:
            if self.base_head is None:
                raise RuntimeError("base_head is not initialized")
            base_logits = self.base_head(z)
        q = self.spectral_features(spectrum)
        spectral_logits = self.qnn_head.classifier(q)
        gate_context = self._gate_context(base_logits)
        gate_inputs = [z, spectrum] if gate_context is None else [z, spectrum, gate_context]
        gate = self.gate(torch.cat(gate_inputs, dim=1))
        guard = self._high_confidence_guard(base_logits)
        scale = self.residual_scale()
        logits = base_logits + scale * gate * guard * spectral_logits
        if return_aux:
            return logits, {
                "gate": gate,
                "guard": guard,
                "base_margin_norm": self._base_margin_norm(base_logits).detach(),
                "residual_scale": scale.detach(),
                "base_confidence": self._base_confidence(base_logits).detach(),
                "base_logits": base_logits,
                "spectral_logits": spectral_logits,
                "spectral_feature": q,
            }
        return logits

    def _gate_context(self, base_logits: torch.Tensor) -> torch.Tensor | None:
        if self.gate_context_mode == "none":
            return None
        probs = torch.softmax(base_logits.detach(), dim=1)
        top2 = torch.topk(base_logits.detach(), k=2, dim=1).values
        confidence = probs.max(dim=1, keepdim=True).values
        margin = torch.tanh((top2[:, :1] - top2[:, 1:2]) / 5.0)
        return torch.cat([confidence, margin], dim=1)

    def _high_confidence_guard(self, base_logits: torch.Tensor) -> torch.Tensor:
        if self.high_confidence_guard_mode == "none":
            return base_logits.new_ones((base_logits.shape[0], 1))
        margin_norm = self._base_margin_norm(base_logits)
        raw_guard = torch.sigmoid((self.guard_tau - margin_norm) / self.guard_temperature)
        return self.guard_floor + (1.0 - self.guard_floor) * raw_guard

    def _base_margin_norm(self, base_logits: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(base_logits.detach(), k=2, dim=1).values
        return torch.tanh((top2[:, :1] - top2[:, 1:2]) / 5.0)

    def _base_confidence(self, base_logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(base_logits.detach(), dim=1).max(dim=1).values
