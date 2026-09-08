"""DRS spectral-shape analysis for difficult Stage 2 baseline subjects."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUM_WAVELENGTHS = 5
NUM_SDS = 6
WAVELENGTHS_NM = (660, 730, 810, 850, 940)
SDS_LABELS = tuple(f"SDS{index}" for index in range(1, NUM_SDS + 1))
TARGET_INFO = {
    "hc": {"column": "GM_hc", "display": "tHb", "unit": "μM"},
    "sto2": {"column": "GM_StO2", "display": "StO₂", "unit": "fraction"},
}


def _normalize_target_mode(target_mode: str) -> str:
    normalized = str(target_mode).strip().lower()
    if normalized not in TARGET_INFO:
        raise ValueError(
            f"target_mode must be one of {tuple(TARGET_INFO)}, got {target_mode!r}."
        )
    return normalized


def _metric_column(frame: pd.DataFrame, target_mode: str, metric: str) -> str:
    target_mode = _normalize_target_mode(target_mode)
    target_column = TARGET_INFO[target_mode]["column"]
    exact = f"test_{target_column}_{metric}"
    if exact in frame:
        return exact
    candidates = [
        column
        for column in frame
        if column.startswith("test_") and column.endswith(f"_{metric}")
    ]
    raise KeyError(
        f"Cannot find {target_mode} {metric} column {exact!r}; candidates: {candidates}"
    )


def load_baseline_performance(
    experiment_dir: str | Path,
    target_mode: str = "hc",
) -> pd.DataFrame:
    target_mode = _normalize_target_mode(target_mode)
    raw = pd.read_csv(Path(experiment_dir) / "loso_results.csv")
    id_col = "original_matlab_subj_id" if "original_matlab_subj_id" in raw else "test_id"
    subject_id = pd.to_numeric(raw[id_col], errors="coerce")
    if id_col == "original_matlab_subj_id":
        subject_id = subject_id - 1
    return (
        pd.DataFrame(
            {
                "subject_id": subject_id,
                "baseline_rmse": pd.to_numeric(
                    raw[_metric_column(raw, target_mode, "RMSE")], errors="coerce"
                ),
                "baseline_bias": pd.to_numeric(
                    raw[_metric_column(raw, target_mode, "Bias")], errors="coerce"
                ),
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


def compute_failure_contrasts(
    performance: pd.DataFrame,
    signature_subject_ids: np.ndarray,
    signatures: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    """Compute robust DRS-shape contrasts for difficult signed-bias groups."""
    required_columns = {"subject_id", "baseline_rmse", "baseline_bias"}
    missing_columns = required_columns - set(performance.columns)
    if missing_columns:
        raise KeyError(f"Missing performance columns: {sorted(missing_columns)}")
    if signatures.ndim != 2 or signatures.shape[1] != NUM_WAVELENGTHS * NUM_SDS:
        raise ValueError(f"Expected signatures [N, 30], got {signatures.shape}.")
    if len(signature_subject_ids) != len(signatures):
        raise ValueError("Signature IDs and signatures have incompatible lengths.")

    signature_index = {int(sid): index for index, sid in enumerate(signature_subject_ids)}
    common = (
        performance[performance.subject_id.isin(signature_index)]
        .drop_duplicates("subject_id", keep="last")
        .sort_values("subject_id")
        .copy()
    )
    if len(common) < 4:
        raise ValueError("At least four common subjects are required.")
    fixed_od = np.vstack([signatures[signature_index[int(sid)]] for sid in common.subject_id])
    fixed_shape = fixed_od - fixed_od.mean(axis=1, keepdims=True)

    q25, q75 = common.baseline_rmse.quantile([0.25, 0.75])
    accurate = common.baseline_rmse.to_numpy() <= q25
    difficult = common.baseline_rmse.to_numpy() >= q75
    groups = {
        "overestimation": difficult & (common.baseline_bias.to_numpy() > 0),
        "underestimation": difficult & (common.baseline_bias.to_numpy() < 0),
    }
    empty_groups = [name for name, mask in groups.items() if not np.any(mask)]
    if empty_groups:
        raise ValueError(f"Difficult bias groups are empty: {empty_groups}")

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
    summary: dict[str, int | float] = {
        "common_n": int(len(common)),
        "accurate_n": int(accurate.sum()),
        "difficult_n": int(difficult.sum()),
        "overestimation_n": int(groups["overestimation"].sum()),
        "underestimation_n": int(groups["underestimation"].sum()),
        "zero_bias_difficult_n": int(
            np.sum(difficult & np.isclose(common.baseline_bias.to_numpy(), 0.0))
        ),
        "rmse_q25": float(q25),
        "rmse_q75": float(q75),
    }
    return contrasts, summary


def make_subject_failure_figure(
    performance: pd.DataFrame,
    signature_subject_ids: np.ndarray,
    signatures: np.ndarray,
    output_dir: str | Path,
    target_mode: str,
    *,
    dpi: int = 300,
) -> dict[str, int | float | str]:
    """Save one thesis-ready two-panel figure for a prediction target."""
    target_mode = _normalize_target_mode(target_mode)
    target_info = TARGET_INFO[target_mode]
    contrasts, summary = compute_failure_contrasts(
        performance,
        signature_subject_ids,
        signatures,
    )
    limit = max(
        1e-12,
        max(float(np.abs(matrix).max()) for matrix in contrasts.values()),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharex=True, sharey=True)
    group_titles = {
        "overestimation": "Difficult overestimation",
        "underestimation": "Difficult underestimation",
    }
    image = None
    for panel, (ax, name) in enumerate(
        zip(axes, ("overestimation", "underestimation"))
    ):
        matrix = contrasts[name]
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        for wl in range(NUM_WAVELENGTHS):
            for sds in range(NUM_SDS):
                value = matrix[wl, sds]
                ax.text(
                    sds,
                    wl,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if abs(value) > 0.58 * limit else "black",
                )
        ax.set_xticks(range(NUM_SDS), SDS_LABELS)
        ax.set_yticks(range(NUM_WAVELENGTHS), [str(value) for value in WAVELENGTHS_NM])
        group_n = int(summary[f"{name}_n"])
        ax.set_title(
            f"{'AB'[panel]}  {group_titles[name]} (n = {group_n})",
            loc="left",
            fontweight="bold",
        )
        ax.set_xlabel("Source–detector separation")
        if panel == 0:
            ax.set_ylabel("Wavelength (nm)")

    if image is None:
        raise RuntimeError("No failure contrast was plotted.")
    colorbar = fig.colorbar(
        image,
        ax=axes,
        fraction=0.035,
        pad=0.025,
    )
    colorbar.set_label("Relative optical-density shape contrast (robust z)")
    unit = str(target_info["unit"])
    threshold_format = ".3f" if target_mode == "hc" else ".4f"
    q25_text = format(float(summary["rmse_q25"]), threshold_format)
    q75_text = format(float(summary["rmse_q75"]), threshold_format)
    fig.suptitle(
        f"DRS spectral-shape contrasts for difficult {target_info['display']} subjects",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        f"Reference: DRS-only Q1 (RMSE ≤ {q25_text} {unit}, n = {summary['accurate_n']}); "
        f"difficult: DRS-only Q4 (RMSE ≥ {q75_text} {unit}, n = {summary['difficult_n']}).",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.16, top=0.82, wspace=0.16)
    output_path = output_dir / f"subject_failure_{target_mode}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "target": target_mode,
        "target_display": str(target_info["display"]),
        "figure": str(output_path),
        **summary,
    }


def make_figure2(
    performance: pd.DataFrame,
    signature_subject_ids: np.ndarray,
    signatures: np.ndarray,
    output_dir: str | Path,
    target_mode: str = "hc",
) -> dict[str, int | float | str]:
    """Backward-compatible alias for the target-specific thesis figure."""
    return make_subject_failure_figure(
        performance,
        signature_subject_ids,
        signatures,
        output_dir,
        target_mode,
    )
