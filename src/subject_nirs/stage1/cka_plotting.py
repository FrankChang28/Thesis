#!/usr/bin/env python3
"""Plot CKA summaries without rerunning model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PARAMETER_DISPLAY = {
    "mu_a_scalp": "Scalp μₐ (cm⁻¹)",
    "mu_a_skull": "Skull μₐ (cm⁻¹)",
    "mu_a_gm": "GM μₐ (cm⁻¹)",
    "mu_s_scalp": "Scalp μs′ (cm⁻¹)",
    "mu_s_skull": "Skull μs′ (cm⁻¹)",
    "mu_s_gm": "GM μs′ (cm⁻¹)",
}

TRANSITION_PARAMETER_DISPLAY = {
    "mu_a_scalp": r"$\mu_{a,\mathrm{scalp}}$",
    "mu_a_skull": r"$\mu_{a,\mathrm{skull}}$",
    "mu_a_gm": r"$\mu_{a,\mathrm{GM}}$",
    "mu_s_scalp": r"$\mu^{\prime}_{s,\mathrm{scalp}}$",
    "mu_s_skull": r"$\mu^{\prime}_{s,\mathrm{skull}}$",
    "mu_s_gm": r"$\mu^{\prime}_{s,\mathrm{GM}}$",
}


def parameter_display(parameter: str) -> str:
    """Return a thesis-facing optical-parameter label."""

    return PARAMETER_DISPLAY.get(parameter, parameter)


def format_parameter_value(parameter: str, value: float) -> str:
    """Format physical OP levels for figure labels only."""

    if parameter.startswith("mu_s_"):
        return f"{value / 10.0:g}"
    if parameter.startswith("mu_a_"):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:g}"


def format_parameter_transition(parameter: str, transition: str) -> str:
    """Format stored transition levels without modifying summary data."""

    arrow = "\u2192"
    return arrow.join(
        format_parameter_value(parameter, float(value))
        for value in transition.split(arrow)
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str, metadata: dict[str, str]) -> None:
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        metadata={key: str(value) for key, value in metadata.items()},
    )
    plt.close(fig)


def _plot_op_pair_heatmap(
    matrix: np.ndarray,
    output_dir: Path,
    *,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    cmap: str,
    stem: str,
    metadata_title: str,
    metadata_description: str,
) -> None:
    """Use the same OP axes and block boundaries for mean and SD."""

    op_count = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape[1] != op_count:
        raise ValueError(f"Expected a square OP-pair matrix, got {matrix.shape}")
    fig, ax = plt.subplots(figsize=(10.8, 9.4), constrained_layout=True)
    image = ax.imshow(
        matrix,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(0.5, op_count + 0.5, 0.5, op_count + 0.5),
        rasterized=True,
    )
    for boundary in range(100, op_count, 100):
        width = 0.35
        if boundary % 900 == 0:
            width = 1.7
        elif boundary % 300 == 0:
            width = 0.9
        ax.axvline(boundary + 0.5, color="white", linewidth=width, alpha=0.75)
        ax.axhline(boundary + 0.5, color="white", linewidth=width, alpha=0.75)
    ax.set_xlabel("OP index")
    ax.set_ylabel("OP index")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(colorbar_label)
    _save(
        fig,
        output_dir,
        stem,
        {"Title": metadata_title, "Description": metadata_description},
    )


def _plot_mean_heatmap(mean_cka: np.ndarray, output_dir: Path, vmin: float, vmax: float) -> None:
    _plot_op_pair_heatmap(
        mean_cka,
        output_dir,
        vmin=vmin,
        vmax=vmax,
        colorbar_label="Mean U-centered linear CKA",
        cmap="viridis",
        stem="fig1_all_op_mean_cka_heatmap",
        metadata_title="All OP-pair mean debiased linear CKA",
        metadata_description=(
            "Validation and test subjects (held-out inference subjects). "
            "100/300/900 OP blocks represent GM/skull/scalp scattering groups. "
            f"Display vmin={vmin}, vmax={vmax}; raw values are not clipped."
        ),
    )


def plot_full_heatmap(output_dir: Path, vmin: float, vmax: float) -> None:
    """Retain the mean-only entry point for existing callers."""

    with h5py.File(output_dir / "full_cka_summary.h5", "r") as handle:
        mean_cka = handle["mean_cka"][:]
    _plot_mean_heatmap(mean_cka, output_dir, vmin, vmax)


def plot_full_heatmaps(output_dir: Path, vmin: float, vmax: float) -> float:
    """Read the CKA summary once and save matching mean/SD heatmaps."""

    with h5py.File(output_dir / "full_cka_summary.h5", "r") as handle:
        mean_cka = handle["mean_cka"][:]
        fold_sd = handle["fold_sd"][:]
    if mean_cka.shape != fold_sd.shape:
        raise ValueError(f"mean_cka shape {mean_cka.shape} does not match fold_sd shape {fold_sd.shape}")

    _plot_mean_heatmap(mean_cka, output_dir, vmin, vmax)

    sd = fold_sd.astype(np.float64)
    upper = np.triu_indices_from(sd, k=1)
    off_diagonal_sd = sd[upper]
    sd_vmax = float(np.quantile(off_diagonal_sd, 0.99))
    correlation = float(
        np.corrcoef(1.0 - mean_cka[upper].astype(np.float64), off_diagonal_sd)[0, 1]
    )
    _plot_op_pair_heatmap(
        sd,
        output_dir,
        vmin=0.0,
        vmax=sd_vmax,
        colorbar_label="Across-fold SD",
        cmap="magma",
        stem="fig1b_all_op_cka_fold_sd_heatmap",
        metadata_title="All OP-pair across-fold CKA SD",
        metadata_description=(
            "Sample SD of fold-level debiased linear CKA. "
            f"Display vmin=0, vmax={sd_vmax}; raw values are not clipped."
        ),
    )
    return correlation


def plot_transitions(output_dir: Path) -> None:
    frame = pd.read_csv(output_dir / "op_transition_summary.csv")
    families = [
        ("absorption", r"(a) Absorption coefficients ($\mathrm{cm}^{-1}$)"),
        ("scattering", r"(b) Reduced scattering coefficients ($\mathrm{cm}^{-1}$)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.3), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.14, top=0.92, wspace=0.40)
    xmin = float((frame["mean_cka"] - frame["fold_sd"]).min())
    xmax = float((frame["mean_cka"] + frame["fold_sd"]).max())
    padding = max(0.02, 0.08 * (xmax - xmin))
    colors = {False: "#5B8DB8", True: "#173F5F"}
    markers = {False: "o", True: "D"}
    for ax, (family, title) in zip(axes, families):
        subset = frame[frame["family"] == family].reset_index(drop=True)
        y = np.arange(len(subset))[::-1]
        for position, (_, row) in zip(y, subset.iterrows()):
            endpoint = bool(row["is_endpoint"])
            ax.errorbar(
                row["mean_cka"],
                position,
                xerr=row["fold_sd"],
                fmt=markers[endpoint],
                color=colors[endpoint],
                ecolor=colors[endpoint],
                markersize=6.5,
                capsize=3,
                linewidth=1.4,
            )
        labels = [
            f"{TRANSITION_PARAMETER_DISPLAY.get(row.parameter, row.parameter)}: {format_parameter_transition(row.parameter, row.transition)}"
            for row in subset.itertuples()
        ]
        ax.set_yticks(y, labels)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="0.88", linewidth=0.7)
        ax.set_xlim(xmin - padding, xmax + padding)
    fig.supxlabel("Mean U-centered linear CKA", y=0.07)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=colors[False], label="Adjacent levels"),
            Line2D([], [], marker="D", linestyle="none", color=colors[True], label="Full-range comparison"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
    )
    _save(
        fig,
        output_dir,
        "fig2_op_transition_cka",
        {
            "Title": "Controlled OP transition CKA",
            "Description": (
                "Circle=adjacent transition, diamond=endpoint transition. Error bars are descriptive SD, not confidence intervals. "
                "Inference subjects comprise checkpoint validation and test subjects."
            ),
        },
    )


def plot_conditional(output_dir: Path) -> None:
    frame = pd.read_csv(output_dir / "conditional_transition_summary.csv")
    target = str(frame["target_parameter"].iloc[0])
    axis_x = str(frame["interaction_axis_x"].iloc[0])
    axis_y = str(frame["interaction_axis_y"].iloc[0])
    transitions = frame[["low_level_index", "high_level_index", "transition", "is_endpoint"]].drop_duplicates()
    values = frame["mean_cka"].to_numpy()
    vmin, vmax = float(values.min()), float(values.max())
    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6
    count = len(transitions)
    fig, axes = plt.subplots(1, count, figsize=(4.8 * count + 1.2, 5.1), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.91, bottom=0.14, top=0.90, wspace=0.34)
    image = None
    for ax, transition in zip(axes[0], transitions.itertuples(index=False)):
        subset = frame[
            (frame["low_level_index"] == transition.low_level_index)
            & (frame["high_level_index"] == transition.high_level_index)
        ]
        pivot = subset.pivot(index="y_level_index", columns="x_level_index", values="mean_cka").sort_index(ascending=True)
        image = ax.imshow(pivot.to_numpy(), origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        x_rows = subset.sort_values("x_level_index").drop_duplicates("x_level_index")
        y_rows = subset.sort_values("y_level_index").drop_duplicates("y_level_index")
        ax.set_xticks(np.arange(len(x_rows)), [format_parameter_value(axis_x, value) for value in x_rows["x_value"]])
        ax.set_yticks(np.arange(len(y_rows)), [format_parameter_value(axis_y, value) for value in y_rows["y_value"]])
        ax.set_xlabel(parameter_display(axis_x))
        ax.set_ylabel(parameter_display(axis_y))
        ax.set_title(f"{parameter_display(target)}: {format_parameter_transition(target, transition.transition)}")
        for row_position in range(pivot.shape[0]):
            for column_position in range(pivot.shape[1]):
                value = pivot.iloc[row_position, column_position]
                normalized = (value - vmin) / (vmax - vmin)
                ax.text(
                    column_position,
                    row_position,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if normalized < 0.55 else "black",
                    fontsize=10,
                )
    colorbar_axis = fig.add_axes([0.93, 0.18, 0.014, 0.54])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Mean CKA")
    _save(
        fig,
        output_dir,
        "fig3_conditional_op_heatmaps",
        {
            "Title": f"Conditional OP heatmaps for {target}",
            "Description": (
                f"Interaction axes: x={axis_x}, y={axis_y}. Remaining OP combinations and folds are averaged. "
                "Shared color scale across transitions."
            ),
        },
    )


def plot_all(output_dir: Path, vmin: float, vmax: float) -> None:
    correlation = plot_full_heatmaps(output_dir, vmin, vmax)
    plot_transitions(output_dir)
    plot_conditional(output_dir)
    print("Saved mean CKA, across-fold SD, transition, and conditional figures as 300 dpi PNG")
    print(f"Descriptive Pearson r between (1 - mean CKA) and across-fold SD: {correlation:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis_dir", type=Path, required=True)
    parser.add_argument("--heatmap_vmin", type=float, default=None)
    parser.add_argument("--heatmap_vmax", type=float, default=None)
    parser.add_argument("--target_param", default=None)
    parser.add_argument("--interaction_axis_x", default=None)
    parser.add_argument("--interaction_axis_y", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.analysis_dir.expanduser().resolve()
    with h5py.File(output_dir / "full_cka_summary.h5", "r") as handle:
        default_vmin = float(handle.attrs["default_heatmap_vmin"])
        default_vmax = float(handle.attrs["default_heatmap_vmax"])
    vmin = default_vmin if args.heatmap_vmin is None else args.heatmap_vmin
    vmax = default_vmax if args.heatmap_vmax is None else args.heatmap_vmax
    if vmin >= vmax:
        raise ValueError("heatmap_vmin must be smaller than heatmap_vmax")
    requested = {
        "target_param": args.target_param,
        "interaction_axis_x": args.interaction_axis_x,
        "interaction_axis_y": args.interaction_axis_y,
    }
    if any(value is not None for value in requested.values()):
        frame = pd.read_csv(output_dir / "conditional_transition_summary.csv")
        actual = {
            "target_param": str(frame["target_parameter"].iloc[0]),
            "interaction_axis_x": str(frame["interaction_axis_x"].iloc[0]),
            "interaction_axis_y": str(frame["interaction_axis_y"].iloc[0]),
        }
        mismatch = {
            key: value
            for key, value in requested.items()
            if value is not None and value != actual[key]
        }
        if mismatch:
            from .cka import compute_conditional_summary

            metadata_path = output_dir / "analysis_metadata.json"
            if metadata_path.is_file():
                folds = json.loads(metadata_path.read_text(encoding="utf-8"))["folds"]
            else:
                folds = sorted(
                    int(path.stem.split("_")[1])
                    for path in (output_dir / "cache").glob("fold_*_cka_vectors.npz")
                )
            ranking = json.loads(
                (output_dir / "endpoint_sensitivity_ranking.json").read_text(encoding="utf-8")
            )
            compute_conditional_summary(
                folds,
                output_dir,
                args.target_param or actual["target_param"],
                args.interaction_axis_x,
                args.interaction_axis_y,
                ranking,
            )
    plot_all(output_dir, vmin, vmax)
    metadata_path = output_dir / "figure_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({"heatmap_vmin": vmin, "heatmap_vmax": vmax, "raw_matrix_clipped": False})
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
