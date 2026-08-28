"""DRS baseline and subject-feature fusion models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

import torch
import torch.nn as nn

from .config import CFG


DRS_EMBEDDING_DIM = 64


def _group_norm(channels: int) -> nn.GroupNorm:
    """Choose the largest configured group count that divides channels."""
    groups = min(CFG.group_norm_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class CNN1D(nn.Module):
    """Encode multi-wavelength DRS along the SDS axis and regress one target.

    ``latent_dim`` is retained for backward compatibility with older direct
    additive-fusion experiments. New experiments should use one of the fusion
    wrappers created by :func:`build_baseline_fusion_model`.
    """

    embedding_dim = DRS_EMBEDDING_DIM

    def __init__(self, output_dim: int = 1, latent_dim: int = 0) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.use_latent = self.latent_dim > 0

        # Input: [batch, wavelength, SDS]. Convolution runs along SDS while
        # wavelength is treated as the input-channel dimension.
        self.drs_encoder = nn.Sequential(
            nn.Conv1d(CFG.num_wl, 32, kernel_size=3, padding=1),
            _group_norm(32),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            _group_norm(64),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            _group_norm(64),
            nn.ReLU(),
        )
        self.drs_projector = nn.Sequential(
            nn.Linear(64 * CFG.num_sds, 128),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU(),
        )

        if self.use_latent:
            self.latent_proj = nn.Sequential(
                nn.Linear(self.latent_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(CFG.dropout),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
            self.gate_net = nn.Sequential(
                nn.Linear(2 * self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.constant_(self.gate_net[-1].bias, -1.0)

        self.regression_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(64, output_dim),
        )

    def encode_drs(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != CFG.input_dim:
            raise ValueError(
                f"Expected x shape [batch, {CFG.input_dim}], got {tuple(x.shape)}"
            )
        x = x.reshape(x.shape[0], CFG.num_wl, CFG.num_sds)
        encoded = self.drs_encoder(x)
        return self.drs_projector(encoded.flatten(1))

    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor | None = None,
        return_parts: bool = False,
    ):
        drs_embedding = self.encode_drs(x)
        latent_gate = None
        gated_latent = None
        fused_embedding = drs_embedding

        if self.use_latent:
            _validate_latent(latent, x.shape[0], self.latent_dim)
            assert latent is not None
            projected_latent = self.latent_proj(latent)
            gate_input = torch.cat([drs_embedding, projected_latent], dim=1)
            latent_gate = torch.sigmoid(self.gate_net(gate_input))
            gated_latent = projected_latent * latent_gate
            fused_embedding = drs_embedding + gated_latent

        prediction = self.regression_head(fused_embedding)
        if return_parts:
            return {
                "pred": prediction,
                "drs_embedding": drs_embedding,
                "adapted_drs_embedding": fused_embedding,
                "gate": latent_gate,
                "residual": gated_latent,
            }
        return prediction


def _validate_latent(
    latent: torch.Tensor | None,
    batch_size: int,
    latent_dim: int,
) -> None:
    if latent is None:
        raise ValueError("latent is required for subject-feature fusion")
    expected = (batch_size, latent_dim)
    if tuple(latent.shape) != expected:
        raise ValueError(f"Expected latent shape {expected}, got {tuple(latent.shape)}")


class BaselineFusionModel(nn.Module, ABC):
    """Shared lifecycle and interface for DRS feature-fusion wrappers."""

    def __init__(
        self,
        baseline: CNN1D,
        latent_dim: int,
        freeze_baseline: bool,
    ) -> None:
        super().__init__()
        if baseline.use_latent:
            raise ValueError("Fusion baseline must be a DRS-only CNN1D")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.baseline = baseline
        self.latent_dim = int(latent_dim)
        self.freeze_baseline = bool(freeze_baseline)
        self.embedding_dim = int(baseline.embedding_dim)

        for parameter in self.baseline.parameters():
            parameter.requires_grad_(not self.freeze_baseline)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen baseline dropout must remain deterministic. A fine-tuned
        # baseline follows the wrapper's normal train/eval state.
        if self.freeze_baseline:
            self.baseline.eval()
        return self

    def _encode_baseline(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.freeze_baseline:
            with torch.no_grad():
                return self.baseline.encode_drs(x)
        return self.baseline.encode_drs(x)

    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor | None = None,
        return_parts: bool = False,
    ):
        _validate_latent(latent, x.shape[0], self.latent_dim)
        assert latent is not None

        drs_embedding = self._encode_baseline(x)
        adapted_embedding, fusion_parts = self.adapt_embedding(
            drs_embedding,
            latent,
        )
        prediction = self.baseline.regression_head(adapted_embedding)

        if return_parts:
            baseline_prediction = self.baseline.regression_head(drs_embedding)
            return {
                "pred": prediction,
                "baseline_pred": baseline_prediction,
                "drs_embedding": drs_embedding,
                "adapted_drs_embedding": adapted_embedding,
                **fusion_parts,
            }
        return prediction

    @abstractmethod
    def adapt_embedding(
        self,
        drs_embedding: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the adapted DRS embedding and diagnostic tensors."""


class BaselineResidualCNN(BaselineFusionModel):
    """Add a gated latent projection to the DRS embedding."""

    def __init__(
        self,
        baseline: CNN1D,
        latent_dim: int,
        output_dim: int = 1,
        use_gate: bool = True,
        gate_init_logit: float = -1.0,
        freeze_baseline: bool = True,
    ) -> None:
        del output_dim  # The loaded baseline already owns the regression head.
        super().__init__(baseline, latent_dim, freeze_baseline)
        self.use_gate = bool(use_gate)

        self.latent_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        # Zero initialization preserves the exact baseline mapping at step 0.
        nn.init.zeros_(self.latent_proj[-1].weight)
        nn.init.zeros_(self.latent_proj[-1].bias)

        if self.use_gate:
            self.gate_head: nn.Module | None = nn.Sequential(
                nn.Linear(2 * self.embedding_dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.embedding_dim),
            )
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.constant_(self.gate_head[-1].bias, gate_init_logit)
        else:
            self.gate_head = None

    def adapt_embedding(
        self,
        drs_embedding: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projected_latent = self.latent_proj(latent)
        if self.gate_head is None:
            gate = torch.ones_like(projected_latent)
        else:
            gate_input = torch.cat([drs_embedding, projected_latent], dim=1)
            gate = torch.sigmoid(self.gate_head(gate_input))

        residual = gate * projected_latent
        return drs_embedding + residual, {
            "projected_latent": projected_latent,
            "residual": residual,
            "gate": gate,
        }


class FiLMBaselineCNN(BaselineFusionModel):
    """Apply latent-conditioned feature-wise scaling and bias to DRS."""

    def __init__(
        self,
        baseline: CNN1D,
        latent_dim: int,
        output_dim: int = 1,
        freeze_baseline: bool = True,
    ) -> None:
        del output_dim
        super().__init__(baseline, latent_dim, freeze_baseline)

        self.film_conditioner = nn.Sequential(
            nn.Linear(self.latent_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(self.embedding_dim, 2 * self.embedding_dim),
        )
        # gamma=0 and beta=0 make FiLM an identity transform at step 0.
        nn.init.zeros_(self.film_conditioner[-1].weight)
        nn.init.zeros_(self.film_conditioner[-1].bias)

    def adapt_embedding(
        self,
        drs_embedding: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gamma_raw, beta = torch.chunk(
            self.film_conditioner(latent),
            chunks=2,
            dim=1,
        )
        gamma = torch.tanh(gamma_raw)
        adapted = (1.0 + gamma) * drs_embedding + beta
        return adapted, {"gamma": gamma, "beta": beta}


FUSION_MODEL_REGISTRY: dict[str, Type[BaselineFusionModel]] = {
    "residual": BaselineResidualCNN,
    "film": FiLMBaselineCNN,
}


def build_baseline_fusion_model(
    fusion_method: str,
    *,
    baseline: CNN1D,
    latent_dim: int,
    output_dim: int,
    freeze_baseline: bool,
    residual_use_gate: bool,
    residual_gate_init_logit: float,
) -> BaselineFusionModel:
    """Build a registered fusion model without changing the training loop."""
    try:
        model_class = FUSION_MODEL_REGISTRY[fusion_method]
    except KeyError as exc:
        available = ", ".join(sorted(FUSION_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown fusion_method={fusion_method!r}; available: {available}"
        ) from exc

    common = {
        "baseline": baseline,
        "latent_dim": latent_dim,
        "output_dim": output_dim,
        "freeze_baseline": freeze_baseline,
    }
    if model_class is BaselineResidualCNN:
        return model_class(
            **common,
            use_gate=residual_use_gate,
            gate_init_logit=residual_gate_init_logit,
        )
    return model_class(**common)


# Compatibility for scripts that imported the original class name.
FrozenBaselineResidualCNN = BaselineResidualCNN
