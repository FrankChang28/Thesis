#!/usr/bin/env python3
"""Create thesis figures for Section 4.2.2.1 baseline LOSO validation.

The independent unit in LOSO is the held-out subject.  Therefore, calibration
curves and primary summary statistics in this module give every subject equal
weight.  Pooled sample-level metrics are retained only as secondary summaries.

Outputs
-------
1. 4_2_2_1_baseline_calibration.png
2. 4_2_2_1_subject_error_distribution.png
3. 4_2_2_1_subject_metrics.csv
4. 4_2_2_1_summary_metrics.csv
5. 4_2_2_1_calibration_data.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_ORDER = ("HC", "StO2")
TARGET_LABELS = {
    "HC": r"tHb",
    "StO2": r"StO$_2$",
}
TARGET_COLORS = {
    "HC": "#2878B5",
    "StO2": "#E07A2D",
}


@dataclass(frozen=True)
class TargetInput:
    name: str
    result_dir: Path
    target_index: int | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create subject-balanced HC/StO2 baseline validation figures."
    )
    parser.add_argument("--hc-dir", required=True, type=Path)
    parser.add_argument("--sto2-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hc-index", type=int, default=None)
    parser.add_argument("--sto2-index", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--min-subjects-per-bin", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _prediction_files(result_dir: Path) -> list[Path]:
    prediction_dir = result_dir / "predictions"
    files = sorted(prediction_dir.glob("fold_*_test_id_*_test_predictions.npz"))
    if not files:
        raise FileNotFoundError(
            f"No test prediction files found in {prediction_dir.resolve()}"
        )
    return files


def _as_target_matrix(values: np.ndarray, name: str, path: Path) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim == 2 and values.shape[1] >= 1:
        return values
    raise ValueError(
        f"{path.name}: {name} must have shape (N,) or (N, T); got {values.shape}."
    )


def _resolve_target_index(
    target: str,
    n_targets: int,
    requested_index: int | None,
) -> int:
    if requested_index is not None:
        index = requested_index
    elif n_targets == 1:
        index = 0
    elif n_targets == 2:
        # Training output order used by this project: GM_hc, GM_StO2.
        index = 0 if target == "HC" else 1
    else:
        raise ValueError(
            f"Cannot infer the {target} column from {n_targets} targets. "
            "Set target_index explicitly."
        )

    if not 0 <= index < n_targets:
        raise IndexError(
            f"Target index {index} is outside the available range 0..{n_targets - 1}."
        )
    return index


def _first_scalar(data: np.lib.npyio.NpzFile, key: str) -> object | None:
    if key not in data.files:
        return None
    values = np.asarray(data[key]).reshape(-1)
    if values.size == 0:
        return None
    value = values[0]
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _subject_id(data: np.lib.npyio.NpzFile, path: Path) -> int:
    for key in ("original_matlab_subj_id", "test_id"):
        value = _first_scalar(data, key)
        if value is not None:
            return int(value)

    match = re.search(r"test_id_(\d+)", path.name)
    if match:
        return int(match.group(1))

    if "subject_id" in data.files:
        values = pd.unique(np.asarray(data["subject_id"]).reshape(-1))
        values = [value for value in values if not pd.isna(value)]
        if len(values) == 1:
            return int(values[0])

    raise ValueError(f"Cannot identify the held-out subject in {path.name}.")


def load_target_predictions(spec: TargetInput) -> pd.DataFrame:
    """Load one target and retain one row per held-out test observation."""
    rows: list[pd.DataFrame] = []
    subject_sources: dict[int, Path] = {}

    for path in _prediction_files(spec.result_dir):
        with np.load(path, allow_pickle=True) as data:
            if "y_true" not in data.files or "y_pred" not in data.files:
                raise KeyError(f"{path.name} does not contain y_true and y_pred.")

            y_true = _as_target_matrix(data["y_true"], "y_true", path)
            y_pred = _as_target_matrix(data["y_pred"], "y_pred", path)
            if y_true.shape != y_pred.shape:
                raise ValueError(
                    f"{path.name}: y_true shape {y_true.shape} does not match "
                    f"y_pred shape {y_pred.shape}."
                )

            index = _resolve_target_index(spec.name, y_true.shape[1], spec.target_index)
            subject_id = _subject_id(data, path)

        if subject_id in subject_sources:
            previous = subject_sources[subject_id]
            raise ValueError(
                "LOSO requires one test prediction file per subject, but subject "
                f"{subject_id} appears in both {previous.name} and {path.name}."
            )
        subject_sources[subject_id] = path

        current = pd.DataFrame(
            {
                "target": spec.name,
                "subject_id": subject_id,
                "y_true": y_true[:, index],
                "y_pred": y_pred[:, index],
            }
        )
        rows.append(current)

    frame = pd.concat(rows, ignore_index=True)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["y_true", "y_pred"]).copy()
    if frame.empty:
        raise ValueError(f"No finite predictions were loaded for {spec.name}.")
    frame["error"] = frame["y_pred"] - frame["y_true"]
    return frame


def compute_subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics separately for each independent LOSO subject.

    NRMSE uses one common reference SD per prediction task.  The reference SD
    gives every subject equal total weight, regardless of how many repeated
    observations that subject contributes.  It is intentionally not calculated
    from each subject's own SD, which can be unstable when a subject spans only
    a narrow target range.
    """
    target_scales: dict[str, float] = {}
    for target, target_frame in predictions.groupby("target", sort=False):
        subject_means = target_frame.groupby("subject_id")["y_true"].mean()
        subject_balanced_mean = float(subject_means.mean())
        subject_mean_squared_deviations = target_frame.groupby("subject_id")[
            "y_true"
        ].apply(
            lambda values: float(
                np.mean(np.square(values.to_numpy(dtype=float) - subject_balanced_mean))
            )
        )
        scale = float(np.sqrt(subject_mean_squared_deviations.mean()))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"{target}: subject-balanced reference SD must be positive; got {scale}."
            )
        target_scales[str(target)] = scale

    rows: list[dict[str, float | int | str]] = []
    for (target, subject_id), group in predictions.groupby(
        ["target", "subject_id"], sort=False
    ):
        error = group["error"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(np.square(error))))
        target_scale = target_scales[str(target)]
        rows.append(
            {
                "target": target,
                "subject_id": int(subject_id),
                "n_samples": int(error.size),
                "reference_SD_subject_balanced": target_scale,
                "RMSE": rmse,
                "NRMSE_percent": rmse / target_scale * 100.0,
                "MAE": float(np.mean(np.abs(error))),
                "Bias": float(np.mean(error)),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(
    predictions: pd.DataFrame,
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for target in TARGET_ORDER:
        pred = predictions[predictions["target"] == target]
        subjects = subject_metrics[subject_metrics["target"] == target]
        if pred.empty:
            continue

        error = pred["error"].to_numpy(dtype=float)
        y_true = pred["y_true"].to_numpy(dtype=float)
        ss_res = float(np.sum(np.square(error)))
        ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
        row: dict[str, float | int | str] = {
            "target": target,
            "n_subjects": int(subjects["subject_id"].nunique()),
            "n_samples": int(len(pred)),
            "reference_SD_subject_balanced": float(
                subjects["reference_SD_subject_balanced"].iloc[0]
            ),
            "pooled_RMSE_secondary": float(np.sqrt(np.mean(np.square(error)))),
            "pooled_MAE_secondary": float(np.mean(np.abs(error))),
            "pooled_Bias_secondary": float(np.mean(error)),
            "pooled_R2_secondary": np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        }
        for metric in ("RMSE", "NRMSE_percent", "MAE", "Bias"):
            values = subjects[metric].to_numpy(dtype=float)
            row[f"subject_{metric}_median"] = float(np.median(values))
            row[f"subject_{metric}_Q1"] = float(np.percentile(values, 25))
            row[f"subject_{metric}_Q3"] = float(np.percentile(values, 75))
            row[f"subject_{metric}_mean"] = float(np.mean(values))
            row[f"subject_{metric}_SD"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def subject_balanced_calibration(
    predictions: pd.DataFrame,
    n_bins: int,
    min_subjects_per_bin: int,
) -> pd.DataFrame:
    """Bin within subjects first, then summarize across equally weighted subjects."""
    output: list[pd.DataFrame] = []
    for target in TARGET_ORDER:
        frame = predictions[predictions["target"] == target].copy()
        if frame.empty:
            continue

        low = float(frame["y_true"].min())
        high = float(frame["y_true"].max())
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            raise ValueError(f"{target}: y_true does not span a usable range.")

        edges = np.linspace(low, high, n_bins + 1)
        frame["bin"] = pd.cut(
            frame["y_true"], edges, labels=False, include_lowest=True
        )
        within_subject = (
            frame.dropna(subset=["bin"])
            .groupby(["subject_id", "bin"], observed=True, as_index=False)
            .agg(
                truth=("y_true", "mean"),
                prediction=("y_pred", "mean"),
                n_samples=("error", "size"),
            )
        )

        rows = []
        for bin_id, group in within_subject.groupby("bin", observed=True, sort=True):
            if group["subject_id"].nunique() < min_subjects_per_bin:
                continue
            rows.append(
                {
                    "target": target,
                    "bin": int(bin_id),
                    "truth_median": float(group["truth"].median()),
                    "prediction_median": float(group["prediction"].median()),
                    "prediction_Q1": float(group["prediction"].quantile(0.25)),
                    "prediction_Q3": float(group["prediction"].quantile(0.75)),
                    "n_subjects": int(group["subject_id"].nunique()),
                    "n_samples": int(group["n_samples"].sum()),
                }
            )
        output.append(pd.DataFrame(rows))

    calibration = pd.concat(output, ignore_index=True)
    if calibration.empty:
        raise ValueError(
            "No calibration bins met min_subjects_per_bin. Lower that setting."
        )
    return calibration


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")


def plot_calibration(
    calibration: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)

    for panel, (ax, target) in enumerate(zip(axes, TARGET_ORDER)):
        frame = calibration[calibration["target"] == target].sort_values("bin")
        stats = summary[summary["target"] == target].iloc[0]
        color = TARGET_COLORS[target]
        x = frame["truth_median"].to_numpy(dtype=float)
        y = frame["prediction_median"].to_numpy(dtype=float)
        q1 = frame["prediction_Q1"].to_numpy(dtype=float)
        q3 = frame["prediction_Q3"].to_numpy(dtype=float)

        plot_low = min(float(x.min()), float(q1.min()))
        plot_high = max(float(x.max()), float(q3.max()))
        padding = max((plot_high - plot_low) * 0.06, 1e-9)
        limits = (plot_low - padding, plot_high + padding)

        ax.plot(limits, limits, "--", color="0.35", linewidth=1.2, label="Ideal: y = x")
        ax.fill_between(x, q1, q3, color=color, alpha=0.22, label="Subject IQR")
        ax.plot(
            x,
            y,
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            label="Subject-balanced median",
        )
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"True {TARGET_LABELS[target]}")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[target]}")
        ax.set_title(f"({chr(97 + panel)}) {TARGET_LABELS[target]}", loc="left")
        ax.text(
            0.04,
            0.96,
            "Subjects = {n}\nSubject RMSE = {median:.3f} "
            "[{q1:.3f}, {q3:.3f}]\nSubject nRMSE = {nmedian:.1f}% "
            "[{nq1:.1f}%, {nq3:.1f}%]".format(
                n=int(stats["n_subjects"]),
                median=stats["subject_RMSE_median"],
                q1=stats["subject_RMSE_Q1"],
                q3=stats["subject_RMSE_Q3"],
                nmedian=stats["subject_NRMSE_percent_median"],
                nq1=stats["subject_NRMSE_percent_Q1"],
                nq3=stats["subject_NRMSE_percent_Q3"],
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.75", "alpha": 0.9},
        )
        ax.legend(loc="lower right", fontsize=8.5)

    _save_figure(fig, output_dir, "baseline_calibration", dpi)
    return fig


def _violin_box_points(
    ax: plt.Axes,
    values: Iterable[float],
    position: float,
    color: str,
    rng: np.random.Generator,
) -> None:
    values = np.asarray(list(values), dtype=float)
    violin = ax.violinplot(
        [values], positions=[position], widths=0.72,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)

    box = ax.boxplot(
        [values], positions=[position], widths=0.22, patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": color, "linewidth": 1.1},
        capprops={"color": color, "linewidth": 1.1},
        boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.2},
    )
    del box
    jitter = rng.uniform(-0.10, 0.10, size=len(values))
    ax.scatter(
        position + jitter, values, s=10, color=color, alpha=0.30,
        edgecolors="none", rasterized=True,
    )


def plot_subject_distributions(
    subject_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2), constrained_layout=True)
    rng = np.random.default_rng(20260826)

    for col, target in enumerate(TARGET_ORDER):
        frame = subject_metrics[subject_metrics["target"] == target]
        stats = summary[summary["target"] == target].iloc[0]
        color = TARGET_COLORS[target]

        for row, metric in enumerate(("RMSE", "Bias")):
            ax = axes[row, col]
            _violin_box_points(ax, frame[metric], 1.0, color, rng)
            if metric == "Bias":
                ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
            ax.set_xlim(0.52, 1.48)
            ax.set_xticks([1.0], [TARGET_LABELS[target]])
            ax.set_ylabel(f"Subject-level {metric}")
            panel = chr(97 + row * 2 + col)
            ax.set_title(f"({panel}) {TARGET_LABELS[target]} {metric}", loc="left")
            ax.text(
                0.04,
                0.96,
                (
                    "Median [IQR]\n{median:.3f} [{q1:.3f}, {q3:.3f}]"
                    + (
                        "\nnRMSE: {nmedian:.1f}% [{nq1:.1f}%, {nq3:.1f}%]"
                        if metric == "RMSE"
                        else ""
                    )
                ).format(
                    median=stats[f"subject_{metric}_median"],
                    q1=stats[f"subject_{metric}_Q1"],
                    q3=stats[f"subject_{metric}_Q3"],
                    nmedian=stats["subject_NRMSE_percent_median"],
                    nq1=stats["subject_NRMSE_percent_Q1"],
                    nq3=stats["subject_NRMSE_percent_Q3"],
                ),
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )

    _save_figure(fig, output_dir, "subject_error_distribution", dpi)
    return fig


def generate_thesis_figures(
    hc_dir: str | Path,
    sto2_dir: str | Path,
    output_dir: str | Path,
    *,
    hc_target_index: int | None = None,
    sto2_target_index: int | None = None,
    n_bins: int = 12,
    min_subjects_per_bin: int = 5,
    dpi: int = 300,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate the complete figure set for baseline result"""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = (
        TargetInput("HC", Path(hc_dir).expanduser(), hc_target_index),
        TargetInput("StO2", Path(sto2_dir).expanduser(), sto2_target_index),
    )
    predictions = pd.concat(
        [load_target_predictions(spec) for spec in specs], ignore_index=True
    )
    subject_metrics = compute_subject_metrics(predictions)
    summary = summarize_metrics(predictions, subject_metrics)
    calibration = subject_balanced_calibration(
        predictions, n_bins=n_bins, min_subjects_per_bin=min_subjects_per_bin
    )

    subject_metrics.to_csv(
        output_dir / "subject_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    calibration.to_csv(
        output_dir / "calibration_data.csv", index=False, encoding="utf-8-sig"
    )

    _apply_style()
    calibration_figure = plot_calibration(calibration, summary, output_dir, dpi)
    distribution_figure = plot_subject_distributions(
        subject_metrics, summary, output_dir, dpi
    )
    if show:
        plt.show()
    else:
        plt.close(calibration_figure)
        plt.close(distribution_figure)

    print(f"Saved thesis outputs to: {output_dir}")
    print(summary.to_string(index=False))
    return {
        "predictions": predictions,
        "subject_metrics": subject_metrics,
        "summary": summary,
        "calibration": calibration,
    }


def main() -> None:
    args = _parse_args()
    generate_thesis_figures(
        hc_dir=args.hc_dir,
        sto2_dir=args.sto2_dir,
        output_dir=args.output_dir,
        hc_target_index=args.hc_index,
        sto2_target_index=args.sto2_index,
        n_bins=args.n_bins,
        min_subjects_per_bin=args.min_subjects_per_bin,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
