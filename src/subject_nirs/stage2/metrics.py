"""Regression metrics for one-target or two-target experiments."""

from __future__ import annotations

import numpy as np

from .config import CFG


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute error summaries for each active regression target."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred shapes differ: {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.shape[1] != len(CFG.target_names):
        raise ValueError(
            f"Expected {len(CFG.target_names)} target columns, got {y_true.shape[1]}"
        )
    if y_true.shape[0] == 0:
        raise ValueError("Cannot compute metrics for an empty array")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics input contains NaN or Inf")

    out = {}

    for j, name in enumerate(CFG.target_names):
        err = y_pred[:, j] - y_true[:, j]
        ae = np.abs(err)
        bias = float(np.mean(err))

        out[f"{name}_MAE"] = float(np.mean(ae))
        out[f"{name}_RMSE"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"{name}_Bias"] = bias
        out[f"{name}_AbsBias"] = float(abs(bias))
        out[f"{name}_ErrorStd"] = float(np.std(err))
        out[f"{name}_P50_AE"] = float(np.percentile(ae, 50))
        out[f"{name}_P75_AE"] = float(np.percentile(ae, 75))
        out[f"{name}_P90_AE"] = float(np.percentile(ae, 90))
        out[f"{name}_P95_AE"] = float(np.percentile(ae, 95))
        out[f"{name}_MaxAE"] = float(np.max(ae))

    return out
