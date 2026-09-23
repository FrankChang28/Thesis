"""Plot paired subject-level RMSE changes on separate validation datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import rankdata, wilcoxon

from .display import (
    BASELINE_DISPLAY,
    target_display,
    target_display_scale,
    target_error_unit,
)


COLORS = ("#4472C4", "#ED7D31", "#70AD47", "#A5A5A5")
CONDITION_LABELS = {
    "unseen_scattering": ("MK-1", "MK-2", "MK-3", "MK-4"),
    "unseen_wm": ("G2-1′", "G2-2′", "G2-3′"),
}
ORIGINAL_PREDICTION_RE = re.compile(
    r"fold_(?P<fold>\d+)_test_id_(?P<test_id>\d+)_test_predictions\.npz$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paired delta-RMSE distributions.")
    parser.add_argument("--original_csv", type=Path, required=True)
    parser.add_argument("--original_prediction_dir", type=Path, required=True)
    parser.add_argument("--original_conditions", type=int, default=3)
    parser.add_argument("--rows_per_original_condition", type=int, default=12960)
    parser.add_argument("--target", choices=("hc", "sto2"), required=True)
    parser.add_argument(
        "--new_dataset", nargs=5, action="append", required=True,
        metavar=(
            "KEY", "LABEL", "EXPECTED_CONDITIONS", "PARENT_CONDITIONS",
            "PER_SIM_CSV",
        ),
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("./results/figures/stage2/unseen_dataset_validation"),
    )
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20250904)
    parser.add_argument("--unit", default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def original_fold_summary(path: Path, target: str) -> dict[int, tuple[int, float]]:
    rows = read_csv(path)
    column = {"hc": "test_GM_hc_RMSE", "sto2": "test_GM_StO2_RMSE"}[target]
    missing = {"fold", "test_id", column} - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    result = {}
    for row in rows:
        fold, value = int(row["fold"]), float(row[column])
        if fold in result:
            raise ValueError(f"Duplicate original fold {fold} in {path}")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid original RMSE for fold {fold}: {value}")
        result[fold] = (int(row["test_id"]), value)
    return result


def original_rmse_by_fold_condition(
    path: Path,
    prediction_dir: Path,
    target: str,
    rows_per_condition: int,
    expected_conditions: int,
) -> dict[tuple[int, int], tuple[int, float]]:
    if rows_per_condition <= 0 or expected_conditions <= 0:
        raise ValueError("Original condition counts and row counts must be positive")
    fold_summary = original_fold_summary(path, target)
    result: dict[tuple[int, int], tuple[int, float]] = {}
    expected_rows = rows_per_condition * expected_conditions
    for fold, (test_id, pooled_rmse) in fold_summary.items():
        prediction_path = prediction_dir / (
            f"fold_{fold:03d}_test_id_{test_id}_test_predictions.npz"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        match = ORIGINAL_PREDICTION_RE.match(prediction_path.name)
        if match is None or int(match.group("fold")) != fold or int(match.group("test_id")) != test_id:
            raise ValueError(f"Unexpected original prediction filename: {prediction_path}")
        with np.load(prediction_path) as prediction:
            missing = {"y_true", "y_pred"} - set(prediction.files)
            if missing:
                raise ValueError(
                    f"{prediction_path} is missing arrays: {sorted(missing)}"
                )
            y_true = np.asarray(prediction["y_true"], dtype=np.float64).reshape(-1)
            y_pred = np.asarray(prediction["y_pred"], dtype=np.float64).reshape(-1)
        if y_true.shape != y_pred.shape or y_true.size != expected_rows:
            raise ValueError(
                f"Unexpected original prediction shapes for fold {fold}: "
                f"true={y_true.shape}, pred={y_pred.shape}, expected rows={expected_rows}"
            )
        error = y_pred - y_true
        pooled_from_predictions = float(np.sqrt(np.mean(error**2)))
        if not np.isclose(pooled_from_predictions, pooled_rmse, rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"Fold {fold} pooled RMSE mismatch: csv={pooled_rmse}, "
                f"predictions={pooled_from_predictions}"
            )
        for condition in range(1, expected_conditions + 1):
            start = (condition - 1) * rows_per_condition
            stop = condition * rows_per_condition
            rmse = float(np.sqrt(np.mean(error[start:stop] ** 2)))
            result[(fold, condition)] = (test_id, rmse)
    return result


def parse_parent_conditions(
    text: str, expected_conditions: int, original_conditions: int
) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise ValueError(f"Invalid parent-condition mapping: {text!r}") from error
    if len(values) != expected_conditions:
        raise ValueError(
            f"Parent mapping {text!r} has {len(values)} entries; "
            f"expected {expected_conditions}"
        )
    if any(value < 1 or value > original_conditions for value in values):
        raise ValueError(
            f"Parent mapping {text!r} must use original conditions "
            f"1..{original_conditions}"
        )
    return values


def paired_deltas(
    path, original, dataset_key, dataset_label, expected_conditions,
    parent_conditions,
):
    rows = read_csv(path)
    required = {"fold", "test_id", "sim_number_1based", "RMSE"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    detail, seen, conditions = [], set(), set()
    for row in rows:
        fold, test_id = int(row["fold"]), int(row["test_id"])
        condition = int(row["sim_number_1based"])
        if (fold, condition) in seen:
            raise ValueError(f"Duplicate fold/condition {(fold, condition)} in {path}")
        seen.add((fold, condition))
        conditions.add(condition)
        parent_condition = int(parent_conditions[condition - 1])
        reference_key = (fold, parent_condition)
        if reference_key not in original:
            raise ValueError(
                f"Fold/parent condition {reference_key} from {path} is absent "
                "from original results"
            )
        original_test_id, original_rmse = original[reference_key]
        if test_id != original_test_id:
            raise ValueError(
                f"Fold {fold} test_id mismatch: original={original_test_id}, new={test_id}"
            )
        new_rmse = float(row["RMSE"])
        if not np.isfinite(new_rmse) or new_rmse < 0:
            raise ValueError(f"Invalid new RMSE for fold {fold}: {new_rmse}")
        detail.append({
            "dataset_key": dataset_key, "dataset_label": dataset_label,
            "condition": condition, "fold": fold, "test_id": test_id,
            "reference_condition": parent_condition,
            "original_RMSE": original_rmse, "new_RMSE": new_rmse,
            "delta_RMSE": new_rmse - original_rmse,
        })
    expected = set(range(1, expected_conditions + 1))
    if conditions != expected:
        raise ValueError(f"{dataset_label} has conditions {sorted(conditions)}; expected {sorted(expected)}")
    original_folds = {fold for fold, _ in original}
    for condition in conditions:
        folds = {int(row["fold"]) for row in detail if row["condition"] == condition}
        if folds != original_folds:
            raise ValueError(f"{dataset_label} condition {condition} lacks matched original folds")
    return detail


def bootstrap_median_ci(values, samples, rng):
    medians = np.median(rng.choice(values, size=(samples, values.size), replace=True), axis=1)
    return tuple(float(value) for value in np.quantile(medians, (0.025, 0.975)))


def rank_biserial(values):
    """Return paired rank-biserial effect; positive values mean improvement."""
    nonzero = values[values != 0]
    if not nonzero.size:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    improvement = ranks[nonzero < 0].sum()
    worsening = ranks[nonzero > 0].sum()
    return float((improvement - worsening) / (improvement + worsening))


def scale_detail_for_display(detail, target):
    """Convert native target errors to thesis display units."""
    scale = target_display_scale(target)
    if scale == 1.0:
        return detail
    for row in detail:
        for key in ("original_RMSE", "new_RMSE", "delta_RMSE"):
            row[key] = float(row[key]) * scale
    return detail


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order, adjusted_sorted, running = np.argsort(values), np.empty_like(values), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def summarize(detail, bootstrap_samples, seed):
    rng, output = np.random.default_rng(seed), []
    dataset_order = list(dict.fromkeys(str(row["dataset_key"]) for row in detail))
    for dataset_key in dataset_order:
        dataset_rows = [row for row in detail if row["dataset_key"] == dataset_key]
        for condition in sorted({int(row["condition"]) for row in dataset_rows}):
            current = [row for row in dataset_rows if row["condition"] == condition]
            values = np.asarray([float(row["delta_RMSE"]) for row in current])
            ci_low, ci_high = bootstrap_median_ci(values, bootstrap_samples, rng)
            pvalue = 1.0 if np.all(values == 0) else float(
                wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue
            )
            output.append({
                "dataset_key": dataset_key, "dataset_label": current[0]["dataset_label"],
                "condition": condition, "n_subjects": values.size,
                "median_delta_RMSE": np.median(values),
                "q25_delta_RMSE": np.quantile(values, .25),
                "q75_delta_RMSE": np.quantile(values, .75),
                "median_ci_low": ci_low, "median_ci_high": ci_high,
                "worsened_fraction": np.mean(values > 0),
                "improved_fraction": np.mean(values < 0),
                "unchanged_fraction": np.mean(values == 0),
                "wilcoxon_p": pvalue, "rank_biserial": rank_biserial(values),
            })
    for row, value in zip(output, holm_adjust([float(row["wilcoxon_p"]) for row in output])):
        row["holm_p"] = value
    return output


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def condition_tick_labels(dataset_key, conditions):
    labels = CONDITION_LABELS.get(dataset_key)
    if labels is None:
        return [f"Set {condition}" for condition in conditions]
    if len(labels) != len(conditions):
        raise ValueError(
            f"{dataset_key} has {len(conditions)} conditions but {len(labels)} labels"
        )
    return list(labels)


def number(value):
    return f"{value:+.4f}" if abs(value) < .01 else f"{value:+.2f}"


def pvalue_text(value):
    return "< 0.001" if value < .001 else f"= {value:.3f}"


def draw_panel(ax, detail, summary_rows, color, letter, unit, seed):
    label = str(detail[0]["dataset_label"])
    dataset_key = str(detail[0]["dataset_key"])
    conditions = sorted({int(row["condition"]) for row in detail})
    values = [np.asarray([float(row["delta_RMSE"]) for row in detail if int(row["condition"]) == c]) for c in conditions]
    positions = np.arange(1, len(conditions) + 1, dtype=float)
    violin = ax.violinplot(values, positions=positions, widths=.72, showextrema=False, points=200)
    for body in violin["bodies"]:
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.22)
    ax.boxplot(
        values, positions=positions, widths=.20, patch_artist=True, showfliers=False, whis=(5, 95),
        boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.3},
        medianprops={"color": "#222222", "linewidth": 1.7},
        whiskerprops={"color": color}, capprops={"color": color},
    )
    rng = np.random.default_rng(seed)
    for position, condition, current in zip(positions, conditions, values):
        ax.scatter(position + rng.uniform(-.12, .12, current.size), current, s=12, color=color,
                   alpha=.33, edgecolors="none", rasterized=True, zorder=3)
        row = next(item for item in summary_rows if int(item["condition"]) == condition)
        median, low, high = (float(row[key]) for key in ("median_delta_RMSE", "median_ci_low", "median_ci_high"))
        ax.errorbar([position], [median], yerr=[[median-low], [high-median]], marker="D",
                    markersize=5.2, color="#111111", capsize=3, linewidth=1.2, zorder=5)
        ax.text(position, .985, f"Median {number(median)}\nWorsened {float(row['worsened_fraction']):.0%}\n$p_{{\\mathrm{{Holm}}}}$ {pvalue_text(float(row['holm_p']))}",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78, "pad": 1.2})
    ax.axhline(0, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_title(f"{letter}  {label}", loc="left", fontweight="bold")
    ax.set_xticks(positions, condition_tick_labels(dataset_key, conditions))
    ax.set_xlabel("Scattering condition")
    ax.set_ylabel(f"ΔRMSE ({unit})")
    ax.grid(axis="y", color="#D7D7D7", linewidth=.7, alpha=.65)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def main():
    args = parse_args()
    if (
        args.dpi <= 0 or args.bootstrap_samples <= 0
        or args.original_conditions <= 0 or args.rows_per_original_condition <= 0
    ):
        raise ValueError("DPI, bootstrap samples, and original-condition sizes must be positive")
    original = original_rmse_by_fold_condition(
        args.original_csv,
        args.original_prediction_dir,
        args.target,
        args.rows_per_original_condition,
        args.original_conditions,
    )
    detail, specs = [], []
    for key, label, expected_text, parent_text, csv_text in args.new_dataset:
        expected, path = int(expected_text), Path(csv_text)
        if expected <= 0:
            raise ValueError("EXPECTED_CONDITIONS must be positive")
        parents = parse_parent_conditions(
            parent_text, expected, args.original_conditions
        )
        detail.extend(
            paired_deltas(path, original, key, label, expected, parents)
        )
        specs.append({
            "key": key,
            "label": label,
            "expected_conditions": expected,
            "parent_conditions": list(parents),
            "csv": str(path.resolve()),
        })
    detail = scale_detail_for_display(detail, args.target)
    summary = summarize(detail, args.bootstrap_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"fig_unseen_dataset_validation_{args.target}_delta_rmse"
    detail_path, summary_path = args.output_dir/f"{prefix}_paired_detail.csv", args.output_dir/f"{prefix}_summary.csv"
    png_path, metadata_path = args.output_dir/f"{prefix}.png", args.output_dir/f"{prefix}_metadata.json"
    write_csv(detail_path, detail); write_csv(summary_path, summary)
    unit = args.unit or target_error_unit(args.target)
    target_name = target_display(args.target)
    title = args.title or f"{BASELINE_DISPLAY} {target_name} robustness under unseen scattering conditions"
    order = [str(item["key"]) for item in specs]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(order), figsize=(6.3*len(order), 6), squeeze=False)
    bound = max(float(np.max(np.abs([float(row["delta_RMSE"]) for row in detail])))*1.08, np.finfo(float).eps)
    for index, (ax, key) in enumerate(zip(axes.ravel(), order)):
        current = [row for row in detail if row["dataset_key"] == key]
        current_summary = [row for row in summary if row["dataset_key"] == key]
        draw_panel(ax, current, current_summary, COLORS[index % len(COLORS)], chr(65+index), unit, args.seed+index)
        ax.set_ylim(-bound, bound)
    fig.subplots_adjust(left=.075, right=.985, top=.94, bottom=.17, wspace=.22)
    fig.legend(handles=[
        Patch(facecolor="#777", edgecolor="#777", alpha=.22, label="Density"),
        Patch(facecolor="white", edgecolor="#555", label="IQR (whiskers: 5th–95th)"),
        Line2D([0], [0], marker="o", linestyle="none", color="#555", alpha=.55, markersize=5, label="Held-out subject"),
        Line2D([0], [0], marker="D", color="#111", markersize=5, label="Median and 95% bootstrap CI"),
        Line2D([0], [0], color="#555", linestyle="--", label="No RMSE change"),
    ], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .025))
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white"); plt.close(fig)
    atomic_json(metadata_path, {
        "description": title,
        "delta_definition": (
            "new-condition subject RMSE minus the same subject's RMSE in the "
            "matched original Group 2 parent condition"
        ),
        "paired_unit": "held-out LOSO subject/fold",
        "original_csv": str(args.original_csv.resolve()),
        "original_prediction_dir": str(args.original_prediction_dir.resolve()),
        "original_conditions": args.original_conditions,
        "rows_per_original_condition": args.rows_per_original_condition,
        "target": args.target, "unit": unit, "datasets": specs,
        "multiple_comparison": "Holm adjustment across all unseen conditions in this target figure",
        "effect_size": "matched-pairs rank-biserial correlation; positive means improvement relative to the original validation reference",
        "display_scale": target_display_scale(args.target),
        "bootstrap_samples": args.bootstrap_samples, "seed": args.seed,
        "num_original_folds": len({fold for fold, _ in original}),
        "png": str(png_path.resolve()),
        "paired_detail_csv": str(detail_path.resolve()), "summary_csv": str(summary_path.resolve()),
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
    })
    for path in (png_path, detail_path, summary_path, metadata_path):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
