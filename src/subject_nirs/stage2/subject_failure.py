"""DRS spectral-shape analysis for difficult Stage 2 baseline subjects."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .display import BASELINE_DISPLAY, target_display, target_display_scale, target_error_unit

NUM_WAVELENGTHS = 5
NUM_SDS = 6
WAVELENGTHS_NM = (660, 730, 810, 850, 940)
SDS_LABELS = tuple(f"SDS{index}" for index in range(1, NUM_SDS + 1))
TARGET_INFO = {
    "hc": {"column": "GM_hc", "display": target_display("hc")},
    "sto2": {"column": "GM_StO2", "display": target_display("sto2")},
}
THICKNESS_COLUMNS = ("scalp", "skull", "csf", "sinus")
THICKNESS_LABELS = {
    "scalp": "Scalp",
    "skull": "Skull",
    "csf": "CSF",
    "sinus": "Frontal sinus",
}


def _normalized_mad_scale(values: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Return the Gaussian-consistent MAD scale for each channel."""
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    if not np.isfinite(scale).all() or np.any(scale <= 1e-12):
        raise ValueError("Reference-group normalized MAD must be finite and positive.")
    return scale


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


def load_layer_thickness_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load the four unstandardized layer-thickness features in millimetres."""
    raw = pd.read_csv(metadata_path)
    required = {"sbj", *THICKNESS_COLUMNS}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Missing layer-thickness columns: {sorted(missing)}")
    frame = pd.DataFrame(
        {
            "subject_id": pd.to_numeric(raw["sbj"], errors="coerce"),
            **{
                column: pd.to_numeric(raw[column], errors="coerce")
                for column in THICKNESS_COLUMNS
            },
        }
    ).dropna()
    frame["subject_id"] = frame["subject_id"].astype(int)
    if frame["subject_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["subject_id"].duplicated(False), "subject_id"].unique()
        )
        raise ValueError(f"Duplicate subject IDs in layer-thickness metadata: {duplicates}")
    values = frame.loc[:, THICKNESS_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Layer thicknesses must be finite, non-negative values in mm.")
    return frame.sort_values("subject_id").reset_index(drop=True)


def compute_subject_signatures(
    x_path: str | Path,
    y_path: str | Path,
    subject_id_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one pooled mean 30-channel DRS signature per subject.

    The current validation design contributes the same number of rows for
    every tHb–StO₂ target pair.  This is checked explicitly, making the pooled
    mean algebraically identical to the former equal-target-weight mean.
    """
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
        _, target_pair_counts = np.unique(subject_y, axis=0, return_counts=True)
        if np.unique(target_pair_counts).size != 1:
            raise ValueError(
                f"Subject {subject_id} has unequal row counts across target pairs; "
                "a pooled mean would not equal the target-balanced mean."
            )
        signatures.append(subject_x.mean(axis=0))
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
    scale = _normalized_mad_scale(fixed_shape[accurate], accurate_center)
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


def _bootstrap_median_difference(
    reference: np.ndarray,
    comparison: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100.")
    reference_draws = rng.integers(
        0, len(reference), size=(n_bootstrap, len(reference))
    )
    comparison_draws = rng.integers(
        0, len(comparison), size=(n_bootstrap, len(comparison))
    )
    differences = np.median(comparison[comparison_draws], axis=1) - np.median(
        reference[reference_draws], axis=1
    )
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def compute_failure_layer_thickness_summary(
    performance: pd.DataFrame,
    thickness_metadata: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260917,
) -> pd.DataFrame:
    """Compare raw layer thicknesses for baseline-defined Q4 bias groups.

    Differences are group medians minus the DRS-only Q1 median in millimetres.
    Confidence intervals use independent subject-level bootstrap resampling of
    the failure group and the Q1 reference group.
    """
    required_performance = {"subject_id", "baseline_rmse", "baseline_bias"}
    missing_performance = required_performance - set(performance.columns)
    if missing_performance:
        raise KeyError(
            f"Missing performance columns: {sorted(missing_performance)}"
        )
    required_thickness = {"subject_id", *THICKNESS_COLUMNS}
    missing_thickness = required_thickness - set(thickness_metadata.columns)
    if missing_thickness:
        raise KeyError(
            f"Missing layer-thickness columns: {sorted(missing_thickness)}"
        )

    common = (
        performance.loc[:, sorted(required_performance)]
        .drop_duplicates("subject_id", keep="last")
        .merge(
            thickness_metadata.loc[:, ["subject_id", *THICKNESS_COLUMNS]],
            on="subject_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    if len(common) < 4:
        raise ValueError("At least four common subjects are required.")
    q25, q75 = common["baseline_rmse"].quantile([0.25, 0.75])
    accurate = common["baseline_rmse"].to_numpy() <= q25
    difficult = common["baseline_rmse"].to_numpy() >= q75
    groups = {
        "overestimation": difficult & (common["baseline_bias"].to_numpy() > 0),
        "underestimation": difficult & (common["baseline_bias"].to_numpy() < 0),
    }
    empty_groups = [name for name, mask in groups.items() if not np.any(mask)]
    if empty_groups:
        raise ValueError(f"Difficult bias groups are empty: {empty_groups}")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, int | float | str]] = []
    for group_name, group_mask in groups.items():
        for layer in THICKNESS_COLUMNS:
            reference = common.loc[accurate, layer].to_numpy(dtype=float)
            comparison = common.loc[group_mask, layer].to_numpy(dtype=float)
            reference_median = float(np.median(reference))
            comparison_median = float(np.median(comparison))
            ci_low, ci_high = _bootstrap_median_difference(
                reference,
                comparison,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            rows.append(
                {
                    "group": group_name,
                    "layer": layer,
                    "layer_display": THICKNESS_LABELS[layer],
                    "common_n": int(len(common)),
                    "reference_n": int(accurate.sum()),
                    "group_n": int(group_mask.sum()),
                    "rmse_q25": float(q25),
                    "rmse_q75": float(q75),
                    "reference_median_mm": reference_median,
                    "reference_q1_mm": float(np.quantile(reference, 0.25)),
                    "reference_q3_mm": float(np.quantile(reference, 0.75)),
                    "group_median_mm": comparison_median,
                    "group_q1_mm": float(np.quantile(comparison, 0.25)),
                    "group_q3_mm": float(np.quantile(comparison, 0.75)),
                    "median_difference_mm": comparison_median - reference_median,
                    "bootstrap_ci_low_mm": ci_low,
                    "bootstrap_ci_high_mm": ci_high,
                    "bootstrap_repeats": int(n_bootstrap),
                    "bootstrap_seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


def _compute_model_specific_failure_contrasts(
    performance_by_model: dict[str, pd.DataFrame],
    signature_subject_ids: np.ndarray,
    signatures: np.ndarray,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, int | float | str]],
]:
    """Compare model-specific difficult groups on one DRS-only reference scale.

    Each model defines its own Q1/Q4 and signed-bias groups. All rows are then
    expressed relative to the baseline Q1 median and MAD, so color magnitude is
    directly comparable across models. The DRS spectra themselves are fixed;
    only the subjects selected by each model can differ.
    """
    if "drs_only" not in performance_by_model:
        raise KeyError("performance_by_model must contain a 'drs_only' reference.")
    if len(performance_by_model) < 2:
        raise ValueError("At least two models are required for a comparison.")
    required_columns = {"subject_id", "baseline_rmse", "baseline_bias"}
    for name, performance in performance_by_model.items():
        missing_columns = required_columns - set(performance.columns)
        if missing_columns:
            raise KeyError(f"Missing {name} performance columns: {sorted(missing_columns)}")
    if signatures.ndim != 2 or signatures.shape[1] != NUM_WAVELENGTHS * NUM_SDS:
        raise ValueError(f"Expected signatures [N, 30], got {signatures.shape}.")
    if len(signature_subject_ids) != len(signatures):
        raise ValueError("Signature IDs and signatures have incompatible lengths.")

    signature_index = {int(sid): index for index, sid in enumerate(signature_subject_ids)}
    common_subject_set = set(signature_index)
    for performance in performance_by_model.values():
        common_subject_set &= set(performance.subject_id.astype(int))
    common_subjects = sorted(common_subject_set)
    if len(common_subjects) < 4:
        raise ValueError("At least four common subjects are required.")

    def align(performance: pd.DataFrame) -> pd.DataFrame:
        return (
            performance.drop_duplicates("subject_id", keep="last")
            .assign(subject_id=lambda frame: frame.subject_id.astype(int))
            .set_index("subject_id")
            .loc[common_subjects]
            .reset_index()
        )

    frames = {
        model_name: align(performance)
        for model_name, performance in performance_by_model.items()
    }
    fixed_od = np.vstack([signatures[signature_index[sid]] for sid in common_subjects])
    fixed_shape = fixed_od - fixed_od.mean(axis=1, keepdims=True)

    masks: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, int | float | str]] = {}
    for model_name, frame in frames.items():
        q25, q75 = frame.baseline_rmse.quantile([0.25, 0.75])
        accurate = frame.baseline_rmse.to_numpy() <= q25
        difficult = frame.baseline_rmse.to_numpy() >= q75
        model_masks = {
            "accurate": accurate,
            "difficult": difficult,
            "overestimation": difficult & (frame.baseline_bias.to_numpy() > 0),
            "underestimation": difficult & (frame.baseline_bias.to_numpy() < 0),
        }
        empty_groups = [
            name
            for name in ("overestimation", "underestimation")
            if not np.any(model_masks[name])
        ]
        if empty_groups:
            raise ValueError(f"{model_name} difficult bias groups are empty: {empty_groups}")
        masks[model_name] = model_masks
        summaries[model_name] = {
            "model": model_name,
            "common_n": int(len(common_subjects)),
            "accurate_n": int(accurate.sum()),
            "difficult_n": int(difficult.sum()),
            "overestimation_n": int(model_masks["overestimation"].sum()),
            "underestimation_n": int(model_masks["underestimation"].sum()),
            "zero_bias_difficult_n": int(
                np.sum(difficult & np.isclose(frame.baseline_bias.to_numpy(), 0.0))
            ),
            "rmse_q25": float(q25),
            "rmse_q75": float(q75),
        }

    reference_mask = masks["drs_only"]["accurate"]
    reference_center = np.median(fixed_shape[reference_mask], axis=0)
    reference_scale = _normalized_mad_scale(
        fixed_shape[reference_mask], reference_center
    )
    contrasts = {
        model_name: {
            group_name: (
                (np.median(fixed_shape[model_masks[group_name]], axis=0) - reference_center)
                / reference_scale
            ).reshape(NUM_WAVELENGTHS, NUM_SDS)
            for group_name in ("overestimation", "underestimation")
        }
        for model_name, model_masks in masks.items()
    }
    reference_difficult = masks["drs_only"]["difficult"]
    for model_name, summary in summaries.items():
        summary["difficult_overlap_n"] = int(
            np.sum(reference_difficult & masks[model_name]["difficult"])
        )
        summary["contrast_reference"] = "drs_only_q1_median_and_mad"
        for group_name in ("overestimation", "underestimation"):
            summary[f"{group_name}_mean_abs_contrast"] = float(
                np.mean(np.abs(contrasts[model_name][group_name]))
            )
    return contrasts, summaries


def compute_failure_comparison_contrasts(
    drs_only_performance: pd.DataFrame,
    residual_performance: pd.DataFrame,
    signature_subject_ids: np.ndarray,
    signatures: np.ndarray,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, int | float | str]],
]:
    """Compare the baseline with DTOF-descriptor difficult groups."""
    return _compute_model_specific_failure_contrasts(
        {
            "drs_only": drs_only_performance,
            "residual_finetune_dtof": residual_performance,
        },
        signature_subject_ids,
        signatures,
    )


def compute_feature_failure_comparison_contrasts(
    drs_only_performance: pd.DataFrame,
    dtof_performance: pd.DataFrame,
    thickness_performance: pd.DataFrame,
    signature_subject_ids: np.ndarray,
    signatures: np.ndarray,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, int | float | str]],
]:
    """Compare baseline, DTOF-descriptor, and tissue-thickness difficult groups."""
    return _compute_model_specific_failure_contrasts(
        {
            "drs_only": drs_only_performance,
            "residual_finetune_dtof": dtof_performance,
            "residual_finetune_thickness": thickness_performance,
        },
        signature_subject_ids,
        signatures,
    )


def _compute_iqr_consistency(
    performance_by_model: dict[str, pd.DataFrame],
    signature_subject_ids: np.ndarray,
    signatures: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Mark channels whose difficult-group IQR stays on one side of Q1.

    This is a descriptive within-group consistency marker, not a hypothesis
    test. Every model selects its own difficult groups, while all models use
    the baseline Q1 median/MAD reference used by the comparison heatmaps.
    """
    if "drs_only" not in performance_by_model:
        raise KeyError("performance_by_model must contain a 'drs_only' reference.")
    signature_index = {int(sid): index for index, sid in enumerate(signature_subject_ids)}
    common_subject_set = set(signature_index)
    for performance in performance_by_model.values():
        common_subject_set &= set(performance.subject_id.astype(int))
    common_subjects = sorted(common_subject_set)

    frames = {
        model_name: (
            performance.drop_duplicates("subject_id", keep="last")
            .assign(subject_id=lambda frame: frame.subject_id.astype(int))
            .set_index("subject_id")
            .loc[common_subjects]
            .reset_index()
        )
        for model_name, performance in performance_by_model.items()
    }
    fixed_od = np.vstack([signatures[signature_index[sid]] for sid in common_subjects])
    fixed_shape = fixed_od - fixed_od.mean(axis=1, keepdims=True)

    reference_frame = frames["drs_only"]
    reference_q25 = reference_frame.baseline_rmse.quantile(0.25)
    reference_mask = reference_frame.baseline_rmse.to_numpy() <= reference_q25
    reference_center = np.median(fixed_shape[reference_mask], axis=0)
    reference_scale = _normalized_mad_scale(
        fixed_shape[reference_mask], reference_center
    )
    standardized = (fixed_shape - reference_center) / reference_scale

    consistency: dict[str, dict[str, np.ndarray]] = {}
    for model_name, frame in frames.items():
        q75 = frame.baseline_rmse.quantile(0.75)
        difficult = frame.baseline_rmse.to_numpy() >= q75
        groups = {
            "overestimation": difficult & (frame.baseline_bias.to_numpy() > 0),
            "underestimation": difficult & (frame.baseline_bias.to_numpy() < 0),
        }
        consistency[model_name] = {}
        for group_name, mask in groups.items():
            q25, q75 = np.quantile(standardized[mask], [0.25, 0.75], axis=0)
            consistency[model_name][group_name] = (
                (q25 > 0) | (q75 < 0)
            ).reshape(NUM_WAVELENGTHS, NUM_SDS)
    return consistency


def _annotate_heatmap_cell(
    ax: plt.Axes,
    sds_index: int,
    wavelength_index: int,
    value: float,
    limit: float,
    iqr_consistent: bool,
    *,
    fontsize: float,
) -> None:
    text_color = "white" if abs(value) > 0.58 * limit else "black"
    ax.text(
        sds_index,
        wavelength_index,
        f"{value:+.2f}",
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
    )
    if iqr_consistent:
        ax.scatter(
            sds_index + 0.34,
            wavelength_index - 0.32,
            s=13,
            marker="o",
            facecolor=text_color,
            edgecolor="black" if text_color == "white" else "white",
            linewidth=0.35,
            zorder=3,
        )


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
    consistency = _compute_iqr_consistency(
        {"drs_only": performance}, signature_subject_ids, signatures
    )["drs_only"]
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
                _annotate_heatmap_cell(
                    ax, sds, wl, value, limit, bool(consistency[name][wl, sds]),
                    fontsize=8.5,
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
    unit = target_error_unit(target_mode)
    display_scale = target_display_scale(target_mode)
    threshold_format = ".3f" if target_mode == "hc" else ".2f"
    q25_text = format(float(summary["rmse_q25"]) * display_scale, threshold_format)
    q75_text = format(float(summary["rmse_q75"]) * display_scale, threshold_format)
    fig.suptitle(
        f"DRS spectral-shape contrasts for difficult {target_info['display']} subjects",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        f"Reference: baseline Q1 (RMSE ≤ {q25_text} {unit}, n = {summary['accurate_n']}); "
        f"difficult: baseline Q4 (RMSE ≥ {q75_text} {unit}, n = {summary['difficult_n']}).\n"
        "Dot: the difficult-group channel-wise IQR remains on one side of the Q1 median.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.19, top=0.82, wspace=0.16)
    output_path = output_dir / f"subject_failure_{target_mode}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    for group_name, matrix in consistency.items():
        summary[f"{group_name}_iqr_consistent_channels"] = int(matrix.sum())
    return {
        "target": target_mode,
        "target_display": str(target_info["display"]),
        "figure": str(output_path),
        **summary,
    }


def make_subject_failure_comparison_figure(
    drs_only_performance: pd.DataFrame,
    residual_performance: pd.DataFrame,
    signature_subject_ids: np.ndarray,
    signatures: np.ndarray,
    output_dir: str | Path,
    target_mode: str,
    *,
    dpi: int = 300,
) -> list[dict[str, int | float | str]]:
    """Save a shared-scale 2x2 baseline versus DTOF-descriptor figure."""
    target_mode = _normalize_target_mode(target_mode)
    target_info = TARGET_INFO[target_mode]
    contrasts, summaries = compute_failure_comparison_contrasts(
        drs_only_performance,
        residual_performance,
        signature_subject_ids,
        signatures,
    )
    consistency = _compute_iqr_consistency(
        {
            "drs_only": drs_only_performance,
            "residual_finetune_dtof": residual_performance,
        },
        signature_subject_ids,
        signatures,
    )
    limit = max(
        1e-12,
        max(
            float(np.abs(matrix).max())
            for model_contrasts in contrasts.values()
            for matrix in model_contrasts.values()
        ),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_rows = (
        ("drs_only", BASELINE_DISPLAY),
        ("residual_finetune_dtof", "DTOF descriptor"),
    )
    group_columns = (
        ("overestimation", "Overestimation"),
        ("underestimation", "Underestimation"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), sharex=True, sharey=True)
    image = None
    panel = 0
    for row, (model_name, model_display) in enumerate(model_rows):
        for column, (group_name, group_display) in enumerate(group_columns):
            ax = axes[row, column]
            matrix = contrasts[model_name][group_name]
            image = ax.imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
            for wavelength_index in range(NUM_WAVELENGTHS):
                for sds_index in range(NUM_SDS):
                    value = matrix[wavelength_index, sds_index]
                    _annotate_heatmap_cell(
                        ax, sds_index, wavelength_index, value, limit,
                        bool(consistency[model_name][group_name][wavelength_index, sds_index]),
                        fontsize=8.2,
                    )
            ax.set_xticks(range(NUM_SDS), SDS_LABELS)
            ax.set_yticks(
                range(NUM_WAVELENGTHS),
                [str(value) for value in WAVELENGTHS_NM],
            )
            group_n = int(summaries[model_name][f"{group_name}_n"])
            ax.set_title(
                f"{'ABCD'[panel]}  {model_display} | {group_display} (n = {group_n})",
                loc="left",
                fontweight="bold",
                fontsize=10.5,
            )
            if row == 1:
                ax.set_xlabel("Source–detector separation")
            if column == 0:
                ax.set_ylabel("Wavelength (nm)")
            panel += 1

    if image is None:
        raise RuntimeError("No failure contrast was plotted.")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label(
        "Robust z score relative to baseline Q1"
    )
    overlap_n = int(summaries["drs_only"]["difficult_overlap_n"])
    fig.suptitle(
        f"DRS spectral shapes of model-specific difficult {target_info['display']} subjects",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Each model reselects Q1/Q4 by its own RMSE and splits Q4 by its own Bias; "
        f"all panels use the baseline Q1 median/MAD reference and one color scale. "
        f"Difficult-group overlap: n = {overlap_n}.\n"
        "Dot: the difficult-group channel-wise IQR remains on one side of the Q1 median.",
        ha="center",
        va="bottom",
        fontsize=8.6,
    )
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.14, top=0.91, hspace=0.29, wspace=0.16)
    output_path = output_dir / f"subject_failure_comparison_{target_mode}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    for model_name, model_consistency in consistency.items():
        for group_name, matrix in model_consistency.items():
            summaries[model_name][f"{group_name}_iqr_consistent_channels"] = int(
                matrix.sum()
            )
    return [
        {
            "target": target_mode,
            "target_display": str(target_info["display"]),
            "figure": str(output_path),
            "shared_color_limit": float(limit),
            **summaries[model_name],
        }
        for model_name, _ in model_rows
    ]


def make_subject_failure_feature_comparison_figure(
    drs_only_performance: pd.DataFrame,
    dtof_performance: pd.DataFrame,
    thickness_performance: pd.DataFrame,
    signature_subject_ids: np.ndarray,
    signatures: np.ndarray,
    output_dir: str | Path,
    target_mode: str,
    *,
    dpi: int = 300,
) -> list[dict[str, int | float | str]]:
    """Save a shared-scale 3x2 failure-phenotype comparison by feature source."""
    target_mode = _normalize_target_mode(target_mode)
    target_info = TARGET_INFO[target_mode]
    contrasts, summaries = compute_feature_failure_comparison_contrasts(
        drs_only_performance,
        dtof_performance,
        thickness_performance,
        signature_subject_ids,
        signatures,
    )
    consistency = _compute_iqr_consistency(
        {
            "drs_only": drs_only_performance,
            "residual_finetune_dtof": dtof_performance,
            "residual_finetune_thickness": thickness_performance,
        },
        signature_subject_ids,
        signatures,
    )
    limit = max(
        1e-12,
        max(
            float(np.abs(matrix).max())
            for model_contrasts in contrasts.values()
            for matrix in model_contrasts.values()
        ),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_rows = (
        ("drs_only", BASELINE_DISPLAY),
        ("residual_finetune_dtof", "DTOF descriptor"),
        (
            "residual_finetune_thickness",
            "Tissue-thickness feature",
        ),
    )
    group_columns = (
        ("overestimation", "Q4-OE"),
        ("underestimation", "Q4-UE"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.6), sharex=True, sharey=True)
    image = None
    panel = 0
    for row, (model_name, model_display) in enumerate(model_rows):
        for column, (group_name, group_display) in enumerate(group_columns):
            ax = axes[row, column]
            matrix = contrasts[model_name][group_name]
            image = ax.imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
            for wavelength_index in range(NUM_WAVELENGTHS):
                for sds_index in range(NUM_SDS):
                    value = matrix[wavelength_index, sds_index]
                    _annotate_heatmap_cell(
                        ax, sds_index, wavelength_index, value, limit,
                        bool(consistency[model_name][group_name][wavelength_index, sds_index]),
                        fontsize=8.0,
                    )
            ax.set_xticks(range(NUM_SDS), SDS_LABELS)
            ax.set_yticks(
                range(NUM_WAVELENGTHS),
                [str(value) for value in WAVELENGTHS_NM],
            )
            group_n = int(summaries[model_name][f"{group_name}_n"])
            ax.set_title(
                f"({'abcdef'[panel]})  {model_display} | {group_display} (n = {group_n})",
                loc="left",
                fontweight="bold",
                fontsize=10.0,
            )
            if column == 0:
                ax.set_ylabel("Wavelength (nm)")
            panel += 1

    if image is None:
        raise RuntimeError("No failure contrast was plotted.")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.025)
    colorbar.set_label(
        "Robust z score relative to baseline Q1"
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.06,
        top=0.98,
        hspace=0.31,
        wspace=0.16,
    )
    output_path = output_dir / f"subject_failure_feature_comparison_{target_mode}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    for model_name, model_consistency in consistency.items():
        for group_name, matrix in model_consistency.items():
            summaries[model_name][f"{group_name}_iqr_consistent_channels"] = int(
                matrix.sum()
            )
    return [
        {
            "target": target_mode,
            "target_display": str(target_info["display"]),
            "figure": str(output_path),
            "shared_color_limit": float(limit),
            **summaries[model_name],
        }
        for model_name, _ in model_rows
    ]


def make_failure_layer_thickness_figure(
    performance_by_target: dict[str, pd.DataFrame],
    thickness_metadata: pd.DataFrame,
    output_dir: str | Path,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260917,
    dpi: int = 300,
) -> pd.DataFrame:
    """Plot raw-mm thickness differences for baseline-defined failure groups."""
    target_order = ("hc", "sto2")
    missing_targets = set(target_order) - set(performance_by_target)
    if missing_targets:
        raise KeyError(f"Missing target performance: {sorted(missing_targets)}")

    summaries: list[pd.DataFrame] = []
    for target_index, target_mode in enumerate(target_order):
        current = compute_failure_layer_thickness_summary(
            performance_by_target[target_mode],
            thickness_metadata,
            n_bootstrap=n_bootstrap,
            seed=seed + target_index,
        )
        current.insert(0, "target", target_mode)
        current.insert(1, "target_display", str(TARGET_INFO[target_mode]["display"]))
        summaries.append(current)
    summary = pd.concat(summaries, ignore_index=True)

    bounds = summary.loc[
        :, ["bootstrap_ci_low_mm", "bootstrap_ci_high_mm", "median_difference_mm"]
    ].to_numpy(dtype=float)
    limit = max(0.1, float(np.max(np.abs(bounds))) * 1.12)
    layer_positions = np.arange(len(THICKNESS_COLUMNS))[::-1]
    group_styles = {
        "overestimation": {
            "label": "Q4-OE",
            "color": "#E07A5F",
            "marker": "o",
            "offset": 0.10,
        },
        "underestimation": {
            "label": "Q4-UE",
            "color": "#4E79A7",
            "marker": "D",
            "offset": -0.10,
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True, sharey=True)
    for panel, (ax, target_mode) in enumerate(zip(axes, target_order)):
        current = summary[summary["target"] == target_mode]
        for group_name, style in group_styles.items():
            group = (
                current[current["group"] == group_name]
                .set_index("layer")
                .loc[list(THICKNESS_COLUMNS)]
            )
            y = layer_positions + float(style["offset"])
            for index, (_, row) in enumerate(group.iterrows()):
                ax.plot(
                    [row["bootstrap_ci_low_mm"], row["bootstrap_ci_high_mm"]],
                    [y[index], y[index]],
                    color=str(style["color"]),
                    linewidth=1.8,
                    solid_capstyle="round",
                    zorder=2,
                )
                ax.plot(
                    [row["bootstrap_ci_low_mm"], row["bootstrap_ci_low_mm"]],
                    [y[index] - 0.035, y[index] + 0.035],
                    color=str(style["color"]),
                    linewidth=1.2,
                    zorder=2,
                )
                ax.plot(
                    [row["bootstrap_ci_high_mm"], row["bootstrap_ci_high_mm"]],
                    [y[index] - 0.035, y[index] + 0.035],
                    color=str(style["color"]),
                    linewidth=1.2,
                    zorder=2,
                )
            ax.scatter(
                group["median_difference_mm"],
                y,
                s=42,
                marker=str(style["marker"]),
                color=str(style["color"]),
                edgecolor="white",
                linewidth=0.7,
                label=str(style["label"]),
                zorder=3,
            )
        ax.axvline(0, color="#303642", linewidth=1.2, linestyle="--")
        ax.set_title(
            f"({'ab'[panel]}) {TARGET_INFO[target_mode]['display']}",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_yticks(
            layer_positions,
            [THICKNESS_LABELS[layer] for layer in THICKNESS_COLUMNS],
        )
        ax.set_xlim(-limit, limit)
        ax.grid(axis="x", color="#D9DEE7", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Median thickness difference from DRS-only Q1 (mm)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
    )

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.84, wspace=0.18)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "subject_failure_layer_thickness.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    summary["figure"] = str(output_path)
    return summary


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
