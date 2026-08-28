"""Draw fold-level RMSE and signed-bias distributions for new-mu_s inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RMSE and signed-bias distributions across LOSO folds."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path(
            "./artifacts/stage2/unseen_scattering/per_fold_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./results/figures/stage2"),
    )
    parser.add_argument(
        "--output_prefix",
        default="fig_new_mus_rmse_bias_distribution",
    )
    parser.add_argument("--fold_column", default="fold")
    parser.add_argument("--rmse_column", default="RMSE")
    parser.add_argument("--bias_column", default="Bias")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--unit", default="µM")
    parser.add_argument(
        "--title",
        default="Prediction errors under unseen scattering spectra",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional metadata description. Defaults to the figure title.",
    )
    return parser.parse_args()


def read_metrics(
    path: Path,
    fold_column: str,
    rmse_column: str,
    bias_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    folds: list[int] = []
    rmse: list[float] = []
    bias: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {fold_column, rmse_column, bias_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row in reader:
            folds.append(int(row[fold_column]))
            rmse.append(float(row[rmse_column]))
            bias.append(float(row[bias_column]))
    fold_array = np.asarray(folds, dtype=np.int64)
    rmse_array = np.asarray(rmse, dtype=np.float64)
    bias_array = np.asarray(bias, dtype=np.float64)
    if fold_array.size == 0:
        raise ValueError("Input CSV contains no rows")
    if np.unique(fold_array).size != fold_array.size:
        raise ValueError("Input CSV contains duplicate fold IDs")
    if not np.isfinite(rmse_array).all() or not np.isfinite(bias_array).all():
        raise ValueError("RMSE or Bias contains NaN/Inf")
    if np.any(rmse_array < 0):
        raise ValueError("RMSE must be non-negative")
    order = np.argsort(fold_array)
    return fold_array[order], rmse_array[order], bias_array[order]


def describe(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def padded_limits(values: np.ndarray, include_zero: bool) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    if include_zero:
        bound = max(abs(lower), abs(upper))
        # Keep signed distributions symmetric around zero, but do not impose an
        # absolute padding: StO2 is stored as a fraction while HC is in µM.
        bound = max(bound, np.finfo(float).eps)
        padding = 0.08 * bound
        return -bound - padding, bound + padding
    span = upper - lower
    if span <= np.finfo(float).eps:
        span = max(abs(lower), abs(upper), np.finfo(float).eps)
    return max(0.0, lower - 0.08 * span), upper + 0.08 * span


def format_stat(value: float, decimals: int) -> str:
    """Format small fractional metrics without displaying negative zero."""
    if abs(value) < 0.5 * 10 ** (-decimals):
        value = 0.0
    return f"{value:.{decimals}f}"


def draw_panel(
    ax,
    values: np.ndarray,
    stats: dict[str, float | int],
    color: str,
    panel_title: str,
    y_label: str,
    show_zero: bool,
) -> None:
    violin = ax.violinplot(
        values,
        positions=[1.0],
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=200,
        bw_method="scott",
    )
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
        body.set_linewidth(1.2)

    box = ax.boxplot(
        values,
        positions=[1.0],
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.4},
        medianprops={"color": "#222222", "linewidth": 1.8},
        whiskerprops={"color": color, "linewidth": 1.2},
        capprops={"color": color, "linewidth": 1.2},
    )
    # Fixed seed makes the fold-level jitter reproducible across reruns.
    rng = np.random.default_rng(20250828)
    jitter = rng.uniform(-0.115, 0.115, size=values.size)
    ax.scatter(
        1.0 + jitter,
        values,
        s=13,
        color=color,
        alpha=0.42,
        edgecolors="none",
        zorder=3,
    )
    if show_zero:
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.3, zorder=1)

    ax.set_xlim(0.55, 1.45)
    ax.set_ylim(*padded_limits(values, include_zero=show_zero))
    ax.set_title(panel_title, loc="left", fontweight="bold", pad=10)
    ax.set_xticks([1.0])
    ax.set_xticklabels(["LOSO folds"])
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.7, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    decimals = 3 if max(abs(float(stats["min"])), abs(float(stats["max"]))) < 1.0 else 2
    annotation = (
        f"n = {int(stats['n'])}\n"
        f"Median [IQR] = {format_stat(float(stats['median']), decimals)} "
        f"[{format_stat(float(stats['q25']), decimals)}, "
        f"{format_stat(float(stats['q75']), decimals)}]\n"
        f"5th–95th = {format_stat(float(stats['q05']), decimals)}–"
        f"{format_stat(float(stats['q95']), decimals)}"
    )
    ax.text(
        0.98,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#C8C8C8",
            "alpha": 0.92,
        },
        zorder=5,
    )


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.bins <= 0 or args.dpi <= 0:
        raise ValueError("--bins and --dpi must be positive")
    folds, rmse, bias = read_metrics(
        args.input_csv,
        args.fold_column,
        args.rmse_column,
        args.bias_column,
    )
    rmse_stats = describe(rmse)
    bias_stats = describe(bias)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.20, wspace=0.20)
    target_color = "#E69F68" if args.unit.strip().lower() == "fraction" else "#5B9BD5"
    draw_panel(
        axes[0],
        rmse,
        rmse_stats,
        color=target_color,
        panel_title="A  RMSE distribution",
        y_label=f"RMSE ({args.unit})",
        show_zero=False,
    )
    draw_panel(
        axes[1],
        bias,
        bias_stats,
        color=target_color,
        panel_title="B  Signed bias distribution",
        y_label=f"Prediction bias ({args.unit})",
        show_zero=True,
    )
    fig.suptitle(args.title, fontsize=13, fontweight="bold", y=0.965)
    legend_handles = [
        Patch(facecolor=target_color, edgecolor=target_color, alpha=0.24, label="Density"),
        Patch(facecolor="white", edgecolor=target_color, label="IQR (whiskers: 5th–95th)"),
        Line2D([0], [0], marker="o", linestyle="none", color=target_color, alpha=0.55,
               markersize=5, label="Individual LOSO fold"),
        Line2D([0], [0], color="#666666", linestyle="--", lw=1.3, label="Zero bias (panel B)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )

    png_path = args.output_dir / f"{args.output_prefix}.png"
    metadata_path = args.output_dir / f"{args.output_prefix}_metadata.json"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    atomic_json(
        metadata_path,
        {
            "description": args.description or args.title,
            "input_csv": str(args.input_csv.resolve()),
            "fold_column": args.fold_column,
            "rmse_column": args.rmse_column,
            "bias_column": args.bias_column,
            "folds": folds.tolist(),
            "num_folds": int(folds.size),
            "unit": args.unit,
            "plot_type": "violin_box_scatter",
            "rmse": rmse_stats,
            "bias": bias_stats,
            "png": str(png_path.resolve()),
            "dpi": args.dpi,
            "matplotlib_version": matplotlib.__version__,
            "numpy_version": np.__version__,
        },
    )
    print(f"RMSE: mean±SD={rmse_stats['mean']:.3f}±{rmse_stats['sd']:.3f}")
    print(f"Bias: mean±SD={bias_stats['mean']:.3f}±{bias_stats['sd']:.3f}")
    print(f"saved: {png_path}")
    print(f"saved: {metadata_path}")


if __name__ == "__main__":
    main()
