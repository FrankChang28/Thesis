"""Fast, data-free verification for configuration and both model stages."""

from __future__ import annotations

import torch

from subject_nirs.stage1.model import DTOFSubjectEncoder
from subject_nirs.stage2.config import CFG
from subject_nirs.stage2.model import CNN1D, build_baseline_fusion_model


def main() -> None:
    stage1 = DTOFSubjectEncoder(num_sds=6, num_time_bins=128, latent_dim=32)
    z = stage1(torch.randn(2, 6, 128), torch.rand(2, 6))
    assert z.shape == (2, 32)
    assert torch.allclose(z.norm(dim=1), torch.ones(2), atol=1e-5)

    x = torch.randn(2, CFG.input_dim)
    latent = torch.randn(2, CFG.structure_feature_dim)
    baseline = CNN1D(output_dim=1)
    assert baseline(x).shape == (2, 1)

    for method in ("residual", "film"):
        fusion = build_baseline_fusion_model(
            method,
            baseline=CNN1D(output_dim=1),
            latent_dim=CFG.structure_feature_dim,
            output_dim=1,
            freeze_baseline=False,
            residual_use_gate=True,
            residual_gate_init_logit=0.0,
        )
        assert fusion(x, latent).shape == (2, 1)

    print("smoke test: Stage 1 (32D), baseline, residual, and FiLM are OK")


if __name__ == "__main__":
    main()
