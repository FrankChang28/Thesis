from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _prototype_contrastive_loss(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Match the two prototypes of each subject and separate other subjects."""
    if view_a.shape != view_b.shape or view_a.ndim != 2:
        raise ValueError("Prototype inputs must have matching shape [G,D]")
    if view_a.shape[0] < 2:
        raise ValueError("At least two subjects are required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    a = F.normalize(view_a, dim=1)
    b = F.normalize(view_b, dim=1)
    logits = (a @ b.T) / float(temperature)
    target = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, target)
        + F.cross_entropy(logits.T, target)
    )

    return loss, {
        "prototype": loss.detach(),
    }


def _angular_compactness(
    z_group: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Pull each acquisition toward its subject spherical centroid."""
    z_unit = F.normalize(z_group, dim=2)
    subject_center = F.normalize(z_unit.mean(dim=1), dim=1)
    distance = 1.0 - (z_unit * subject_center[:, None, :]).sum(dim=2)
    loss = distance.mean()

    return loss, {
        "compactness": loss.detach(),
    }


def subject_group_loss(
    z_group: torch.Tensor,
    *,
    temperature: float = 0.10,
    compact_weight: float = 0.25,
    randomize_split: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Prototype contrastive loss plus angular within-subject compactness.

    Parameters
    ----------
    z_group:
        Unit embeddings with shape [subjects, acquisitions, latent_dim].
    """
    if z_group.ndim != 3:
        raise ValueError("z_group must have shape [G,K,D]")
    subjects, acquisitions, _ = z_group.shape
    if subjects < 2:
        raise ValueError("At least two subjects are required")
    if acquisitions < 4 or acquisitions % 2 != 0:
        raise ValueError("The number of acquisitions must be even and at least four")
    if compact_weight < 0:
        raise ValueError("compact_weight must be non-negative")

    if randomize_split:
        order = torch.rand(subjects, acquisitions, device=z_group.device).argsort(dim=1)
        z_views = torch.gather(
            z_group,
            dim=1,
            index=order.unsqueeze(-1).expand_as(z_group),
        )
    else:
        z_views = z_group

    split = acquisitions // 2
    prototype_a = z_views[:, :split].mean(dim=1)
    prototype_b = z_views[:, split:].mean(dim=1)

    prototype, prototype_metrics = _prototype_contrastive_loss(
        prototype_a,
        prototype_b,
        temperature,
    )
    compactness, compactness_metrics = _angular_compactness(z_group)
    total = prototype + float(compact_weight) * compactness

    metrics: Dict[str, torch.Tensor] = {"total": total.detach()}
    metrics.update(prototype_metrics)
    metrics.update(compactness_metrics)
    return total, metrics
