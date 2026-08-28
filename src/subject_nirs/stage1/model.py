from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DTOFSubjectEncoder(nn.Module):
    """Encode one multi-SDS DTOF acquisition into a unit subject embedding.

    The model deliberately avoids a Transformer. Every SDS curve is processed by
    the same temporal CNN, its relative integrated area is appended, and the SDS
    features are concatenated in their fixed acquisition order before a small MLP.
    The fixed concatenation order already identifies each SDS channel.
    """

    def __init__(
        self,
        num_sds: int,
        num_time_bins: int,
        latent_dim: int = 32,
        feature_dim: int = 48,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_sds = int(num_sds)
        self.num_time_bins = int(num_time_bins)
        self.latent_dim = int(latent_dim)
        self.feature_dim = int(feature_dim)

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, feature_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8 if feature_dim % 8 == 0 else 1, feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        input_dim = self.num_sds * (self.feature_dim + 1)
        hidden_dim = max(2 * self.feature_dim, 2 * self.latent_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def encode(
        self,
        temporal: torch.Tensor,
        relative_area: torch.Tensor,
    ) -> torch.Tensor:
        if temporal.ndim != 3:
            raise ValueError(f"temporal must be [B,S,T], got {tuple(temporal.shape)}")
        if relative_area.ndim != 2:
            raise ValueError(
                f"relative_area must be [B,S], got {tuple(relative_area.shape)}"
            )

        batch, num_sds, num_time_bins = temporal.shape
        if num_sds != self.num_sds or num_time_bins != self.num_time_bins:
            raise ValueError(
                f"Expected [B,{self.num_sds},{self.num_time_bins}], "
                f"got {tuple(temporal.shape)}"
            )
        if relative_area.shape != (batch, num_sds):
            raise ValueError(
                f"Expected relative_area {(batch, num_sds)}, "
                f"got {tuple(relative_area.shape)}"
            )

        temporal_features = self.temporal_encoder(
            temporal.reshape(batch * num_sds, 1, num_time_bins)
        ).reshape(batch, num_sds, self.feature_dim)

        # relative_area is derived from the same DTOF, not an external feature.
        sds_features = torch.cat(
            [temporal_features, relative_area.unsqueeze(-1)],
            dim=-1,
        )
        z = self.head(sds_features.flatten(start_dim=1))
        return F.normalize(z, dim=1)

    def forward(
        self,
        temporal: torch.Tensor,
        relative_area: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode(temporal, relative_area)
