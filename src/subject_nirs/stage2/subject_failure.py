"""Focused Stage 2 subject-failure analysis used for thesis Figure 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUM_WAVELENGTHS = 5
NUM_SDS = 6


def _metric_column(frame: pd.DataFrame, metric: str) -> str:
    exact = f"test_GM_hc_{metric}"
    if exact in frame:
        return exact
    candidates = [c for c in frame if c.startswith("test_") and c.endswith(f"_{metric}")]
    if len(candidates) != 1:
        raise KeyError(f"Cannot uniquely identify HC {metric}: {candidates}")
    return candidates[0]


def load_baseline_performance(experiment_dir: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(Path(experiment_dir) / "loso_results.csv")
    id_col = "original_matlab_subj_id" if "original_matlab_subj_id" in raw else "test_id"
    subject_id = pd.to_numeric(raw[id_col], errors="coerce")
    if id_col == "original_matlab_subj_id":
        subject_id = subject_id - 1
    return (
        pd.DataFrame(
            {
                "subject_id": subject_id,
                "baseline_rmse": pd.to_numeric(raw[_metric_column(raw, "RMSE")], errors="coerce"),
                "baseline_bias": pd.to_numeric(raw[_metric_column(raw, "Bias")], errors="coerce"),
            }
        )
        .dropna()
        .assign(subject_id=lambda d: d.subject_id.astype(int))
        .drop_duplicates("subject_id", keep="last")
    )


def compute_subject_signatures(
    x_path: str | Path,
    y_path: str | Path,
    subject_id_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an equal-target-weight mean 30-channel DRS signature per subject."""
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    subject_ids = np.load(subject_id_path, mmap_mode="r")
    if x.ndim != 2 or x.shape[1] != NUM_WAVELENGTHS * NUM_SDS:
        raise ValueError(f"Expected X [N, 30], got {x.shape}")
    if y.ndim != 2 or y.shape[1] < 2 or len(x) != len(y) or len(x) != len(subject_ids):
        raise ValueError("X, Y, and subject IDs have incompatible shapes")

    unique_subjects = np.unique(subject_ids).astype(int)
    signatures: list[np.ndarray] = []
    for subject_id in unique_subjects:
        mask = np.asarray(subject_ids == subject_id)
        subject_x = np.asarray(x[mask], dtype=np.float64)
        subject_y = np.asarray(y[mask, :2], dtype=np.float64)
        target_pairs = np.unique(subject_y, axis=0)
        cell_means = [
            subject_x[np.all(np.isclose(subject_y, pair), axis=1)].mean(axis=0)
            for pair in target_pairs
        ]
        signatures.append(np.mean(cell_means, axis=0))
    return unique_subjects, np.asarray(signatures, dtype=np.float32)


def load_or_compute_signatures(
    cache_path: str | Path,
    x_path: str | Path,
    y_path: str | Path,
    subject_id_path: str | Path,
    *,
    recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not recompute:
        payload = np.load(cache_path)
        return payload["subject_ids"].astype(int), payload["signatures"]
    subject_ids, signatures = compute_subject_signatures(x_path, y_path, subject_id_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, subject_ids=subject_ids, signatures=signatures)
    return subject_ids, signatures


def make_figure2(
    performance: pd.DataFrame,
    signature_subject_ids: np.ndarray,
    signatures: np.ndarray,
    output_dir: str | Path,
) -> dict[str, int | float]:
    """Plot robust spectral-shape contrasts for difficult bias groups."""
    signature_index = {int(sid): index for index, sid in enumerate(signature_subject_ids)}
    common = performance[performance.subject_id.isin(signature_index)].copy()
    fixed_od = np.vstack([signatures[signature_index[int(sid)]] for sid in common.subject_id])
    fixed_shape = fixed_od - fixed_od.mean(axis=1, keepdims=True)

    q25, q75 = common.baseline_rmse.quantile([0.25, 0.75])
    accurate = common.baseline_rmse.to_numpy() <= q25
    difficult = common.baseline_rmse.to_numpy() >= q75
    groups = {
        "overestimation": difficult & (common.baseline_bias.to_numpy() > 0),
        "underestimation": difficult & (common.baseline_bias.to_numpy() < 0),
    }
    accurate_center = np.median(fixed_shape[accurate], axis=0)
    mad = 1.4826 * np.median(np.abs(fixed_shape[accurate] - accurate_center), axis=0)
    positive = mad[np.isfinite(mad) & (mad > 1e-12)]
    scale = np.maximum(mad, np.percentile(positive, 10) if len(positive) else 1.0)
    contrasts = {
        name: ((np.median(fixed_shape[mask], axis=0) - accurate_center) / scale).reshape(
            NUM_WAVELENGTHS, NUM_SDS
        )
        for name, mask in groups.items()
    }
    limit = max(float(np.abs(matrix).max()) for matrix in contrasts.values())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, matrix in contrasts.items():
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        for wl in range(NUM_WAVELENGTHS):
            for sds in range(NUM_SDS):
                value = matrix[wl, sds]
                ax.text(sds, wl, f"{value:+.2f}", ha="center", va="center", color="white" if abs(value) > 0.58 * limit else "black")
        ax.set_xticks(range(NUM_SDS), [f"SDS {i + 1}" for i in range(NUM_SDS)])
        ax.set_yticks(range(NUM_WAVELENGTHS), [f"WL {i + 1}" for i in range(NUM_WAVELENGTHS)])
        ax.set(xlabel="Source-detector separation", ylabel="Wavelength", title=f"DRS shape associated with HC {name}")
        fig.colorbar(image, ax=ax, label="Robust z: relative OD log-ratio contrast")
        fig.tight_layout()
        fig.savefig(output_dir / f"subject_failure_{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "accurate_n": int(accurate.sum()),
        "overestimation_n": int(groups["overestimation"].sum()),
        "underestimation_n": int(groups["underestimation"].sum()),
        "rmse_q25": float(q25),
        "rmse_q75": float(q75),
    }
