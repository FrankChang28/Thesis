#!/usr/bin/env python3
"""Probe whether frozen DTOF structural latents predict head-layer thicknesses.

The script supports two evaluation modes:

1) Single-fold evaluation
   Pass one subject_features_eval_grid.npz file. The probe is trained on the
   subjects marked ``train``, alpha is selected on ``val``, and the single
   held-out ``test`` subject is predicted.

2) Multi-fold aggregation (recommended)
   Pass a glob matching feature files from multiple TEST_ID runs. A separate
   probe is fitted inside each fold, and only that fold's held-out test
   prediction is retained. Test predictions are then aggregated across folds.

Expected metadata layout:
    pd.read_csv(metadata_csv).to_numpy()[:, 5:8]
where the three columns are scalp, skull, and CSF thickness, and metadata row i
corresponds to zero-based subject_id i.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_NAMES = ("scalp", "skull", "csf")
TARGET_LABELS = {
    "scalp": "Scalp",
    "skull": "Skull",
    "csf": "CSF",
}


@dataclass
class TargetMetrics:
    n: int
    mae: float
    rmse: float
    r2: float
    pearson_r: float
    spearman_rho: float


@dataclass
class FoldResult:
    feature_file: str
    test_subject_id: int
    test_matlab_subject_id: int
    selected_alpha: float
    validation_standardized_rmse: float
    validation_metrics: Dict[str, TargetMetrics]
    test_true: List[float]
    test_pred: List[float]
    train_subject_ids: List[int]
    val_subject_ids: List[int]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fit a leakage-safe frozen-latent probe from DTOF z_struct to "
            "scalp/skull/CSF thickness."
        )
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--feature_npz",
        type=str,
        help="One subject_features_eval_grid.npz file.",
    )
    group.add_argument(
        "--feature_glob",
        type=str,
        help=(
            "Glob for multiple folds, e.g. "
            "artifacts/stage1/*/relat_cons_32/subject_features_all_op.npz"
        ),
    )
    p.add_argument(
        "--metadata_csv",
        type=str,
        default="./data/temp_metadata_v2.csv",
    )
    p.add_argument(
        "--metadata_start_col",
        type=int,
        default=5,
        help="Zero-based first metadata column; columns start:start+3 are used.",
    )
    p.add_argument("--output_dir", type=str, default="./thickness_probe")
    p.add_argument(
        "--feature_key",
        choices=["features", "latent_normalized", "latent_raw"],
        default="features",
        help=(
            "Use 'features' for the training-only normalized subject latent "
            "saved by the V4 extractor."
        ),
    )
    p.add_argument(
        "--alphas",
        type=str,
        default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100,300,1000",
        help="Comma-separated Ridge alpha candidates.",
    )
    p.add_argument(
        "--no_refit_train_val",
        action="store_true",
        help=(
            "Do not refit the selected probe on train+val before predicting test. "
            "By default refitting is enabled."
        ),
    )
    p.add_argument(
        "--save_fold_plots",
        action="store_true",
        help="Also save one validation scatter plot per fold.",
    )
    p.add_argument(
        "--plot_format",
        choices=["png"],
        default="png",
        help="Aggregate figures are saved as PNG only.",
    )
    p.add_argument(
        "--plot_dpi",
        type=int,
        default=300,
        help="Resolution of raster figures (default: 300 dpi).",
    )
    p.add_argument(
        "--thickness_unit",
        type=str,
        default="",
        help=(
            "Optional physical unit shown on plot axes and metric annotations, "
            "for example 'mm'. Leave empty if the metadata unit is unknown."
        ),
    )
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Compute and save tabular results without generating figures.",
    )
    return p


def finite_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def parse_alphas(text: str) -> np.ndarray:
    values = np.asarray([float(x.strip()) for x in text.split(",") if x.strip()])
    if values.size == 0 or np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("--alphas must contain finite positive values")
    return np.unique(values)


def load_metadata(path: str, start_col: int) -> np.ndarray:
    frame = pd.read_csv(path, header=0)
    if frame.shape[1] < start_col + 3:
        raise ValueError(
            f"Metadata has {frame.shape[1]} columns, but columns "
            f"{start_col}:{start_col + 3} were requested."
        )
    values = frame.iloc[:, start_col : start_col + 3].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    if values.shape[1] != 3:
        raise RuntimeError(f"Expected three thickness columns, got {values.shape}")
    bad = np.argwhere(~np.isfinite(values))
    if bad.size:
        row, col = bad[0]
        raise ValueError(
            f"Non-finite metadata at row {row}, target {TARGET_NAMES[col]}."
        )
    return values


def resolve_feature_files(feature_npz: str | None, feature_glob: str | None) -> List[Path]:
    if feature_npz:
        files = [Path(feature_npz)]
    else:
        files = [Path(x) for x in sorted(glob.glob(str(feature_glob)))]
    if not files:
        raise FileNotFoundError("No feature files matched the requested path/glob")
    missing = [str(x) for x in files if not x.is_file()]
    if missing:
        raise FileNotFoundError(f"Feature files do not exist: {missing[:3]}")
    return files


def load_fold(path: Path, feature_key: str, metadata: np.ndarray):
    with np.load(path, allow_pickle=False) as data:
        if feature_key not in data.files:
            fallback = "features" if "features" in data.files else None
            if fallback is None:
                raise KeyError(
                    f"{path} has no key '{feature_key}'. Available: {data.files}"
                )
            print(
                f"[Warning] {path}: key '{feature_key}' missing; using '{fallback}'."
            )
            feature_key = fallback
        x = np.asarray(data[feature_key], dtype=np.float64)
        subject_ids = np.asarray(data["subject_ids"], dtype=np.int64)
        splits = np.asarray(data["splits"]).astype(str)

    if x.ndim != 2:
        raise ValueError(f"{path}: features must be [subjects, dims], got {x.shape}")
    if len(x) != len(subject_ids) or len(x) != len(splits):
        raise ValueError(f"{path}: feature, subject_ids, and splits lengths differ")
    if len(np.unique(subject_ids)) != len(subject_ids):
        raise ValueError(f"{path}: duplicate subject IDs")
    if subject_ids.min() < 0 or subject_ids.max() >= len(metadata):
        raise IndexError(
            f"{path}: subject IDs [{subject_ids.min()}, {subject_ids.max()}] "
            f"are incompatible with {len(metadata)} metadata rows"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: non-finite feature values")

    y = metadata[subject_ids]
    masks = {name: splits == name for name in ("train", "val", "test")}
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    if counts["train"] < 2 or counts["val"] < 2 or counts["test"] != 1:
        raise ValueError(
            f"{path}: expected train>=2, val>=2, test=1; got {counts}"
        )
    return x, y, subject_ids, masks


def make_model(alpha: float) -> TransformedTargetRegressor:
    regressor = Pipeline(
        steps=[
            ("x_scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    # Scale each thickness using training subjects only. This keeps one target's
    # physical scale from dominating the shared alpha selection.
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def standardized_rmse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    scale = y_train.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return float(np.sqrt(np.mean(((y_true - y_pred) / scale) ** 2)))


def safe_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(fn(y_true, y_pred)[0])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, TargetMetrics]:
    result: Dict[str, TargetMetrics] = {}
    for j, name in enumerate(TARGET_NAMES):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        r2 = float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan")
        result[name] = TargetMetrics(
            n=int(len(yt)),
            mae=float(mean_absolute_error(yt, yp)),
            rmse=float(np.sqrt(mean_squared_error(yt, yp))),
            r2=r2,
            pearson_r=safe_corr(pearsonr, yt, yp),
            spearman_rho=safe_corr(spearmanr, yt, yp),
        )
    return result


def select_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Sequence[float],
) -> Tuple[float, float, np.ndarray, pd.DataFrame]:
    rows = []
    best = None
    for alpha in alphas:
        model = make_model(float(alpha))
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        score = standardized_rmse(y_val, pred, y_train)
        rows.append({"alpha": float(alpha), "validation_standardized_rmse": score})
        candidate = (score, float(alpha), pred)
        if best is None or candidate[0] < best[0] - 1e-12 or (
            abs(candidate[0] - best[0]) <= 1e-12 and candidate[1] > best[1]
        ):
            best = candidate
    assert best is not None
    return best[1], best[0], best[2], pd.DataFrame(rows)


def save_figure(
    fig: plt.Figure,
    output_stem: Path,
    plot_format: str,
    dpi: int,
) -> List[Path]:
    """Save a figure in the requested publication and preview formats."""
    output_stem = output_stem.with_suffix("")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    formats = ("png",)
    saved_paths = []
    for suffix in formats:
        path = output_stem.with_suffix(f".{suffix}")
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)
    return saved_paths


def format_axis_label(prefix: str, target: str, thickness_unit: str) -> str:
    label = f"{prefix} {TARGET_LABELS[target]} thickness"
    return f"{label} ({thickness_unit})" if thickness_unit else label


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_stem: Path,
    title: str,
    metrics: Dict[str, TargetMetrics],
    plot_format: str,
    dpi: int,
    thickness_unit: str,
) -> List[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    for j, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
        ax.scatter(
            y_true[:, j],
            y_pred[:, j],
            s=34,
            alpha=0.78,
            color="#2C7FB8",
            edgecolors="white",
            linewidths=0.45,
        )
        lo = min(float(y_true[:, j].min()), float(y_pred[:, j].min()))
        hi = max(float(y_true[:, j].max()), float(y_pred[:, j].max()))
        if math.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
        padding = max((hi - lo) * 0.06, 1e-6)
        axis_lo, axis_hi = lo - padding, hi + padding
        ax.plot(
            [axis_lo, axis_hi],
            [axis_lo, axis_hi],
            linestyle="--",
            linewidth=1.2,
            color="#555555",
            label="Ideal prediction",
        )
        ax.set_xlim(axis_lo, axis_hi)
        ax.set_ylim(axis_lo, axis_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(format_axis_label("True", name, thickness_unit))
        ax.set_ylabel(format_axis_label("Predicted", name, thickness_unit))
        ax.set_title(TARGET_LABELS[name], fontweight="bold")
        metric = metrics[name]
        unit_suffix = f" {thickness_unit}" if thickness_unit else ""
        annotation = (
            f"n = {metric.n}\n"
            f"MAE = {metric.mae:.3f}{unit_suffix}\n"
            f"RMSE = {metric.rmse:.3f}{unit_suffix}\n"
            f"$R^2$ = {metric.r2:.3f}"
        )
        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "alpha": 0.92,
            },
        )
        ax.grid(alpha=0.22, linewidth=0.7)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    saved_paths = save_figure(fig, output_stem, plot_format, dpi)
    plt.close(fig)
    return saved_paths


def plot_probe_vs_baseline(
    probe_metrics: Dict[str, TargetMetrics],
    baseline_metrics: Dict[str, TargetMetrics],
    output_stem: Path,
    plot_format: str,
    dpi: int,
    thickness_unit: str,
) -> List[Path]:
    """Plot held-out MAE and RMSE for the probe and train-mean baseline."""
    labels = [TARGET_LABELS[name] for name in TARGET_NAMES]
    x = np.arange(len(TARGET_NAMES), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.3))

    for ax, metric_name, title in zip(
        axes,
        ("mae", "rmse", "r2"),
        ("Mean absolute error", "Root mean squared error", "Coefficient of determination"),
    ):
        probe_values = [getattr(probe_metrics[name], metric_name) for name in TARGET_NAMES]
        baseline_values = [
            getattr(baseline_metrics[name], metric_name) for name in TARGET_NAMES
        ]
        probe_bars = ax.bar(
            x - width / 2,
            probe_values,
            width,
            label="Frozen-latent Ridge probe",
            color="#2C7FB8",
        )
        baseline_bars = ax.bar(
            x + width / 2,
            baseline_values,
            width,
            label="Train-mean baseline",
            color="#BDBDBD",
        )
        ax.set_xticks(x, labels)
        ax.set_title(title, fontweight="bold")
        if metric_name == "r2":
            ax.set_ylabel("$R^2$")
            ax.axhline(0.0, color="#555555", linewidth=0.9)
        else:
            ylabel = metric_name.upper()
            ax.set_ylabel(
                f"{ylabel} ({thickness_unit})" if thickness_unit else ylabel
            )
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)
        for bars in (probe_bars, baseline_bars):
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    saved_paths = save_figure(fig, output_stem, plot_format, dpi)
    plt.close(fig)
    return saved_paths


def metrics_to_json(metrics: Dict[str, TargetMetrics]) -> Dict[str, dict]:
    output = {}
    for name, metric in metrics.items():
        row = asdict(metric)
        output[name] = {k: finite_float(v) if isinstance(v, float) else v for k, v in row.items()}
    return output


def fit_one_fold(
    feature_file: Path,
    feature_key: str,
    metadata: np.ndarray,
    alphas: np.ndarray,
    refit_train_val: bool,
    output_dir: Path,
    save_fold_plot: bool,
    plot_format: str,
    plot_dpi: int,
    thickness_unit: str,
) -> Tuple[FoldResult, pd.DataFrame, pd.DataFrame]:
    x, y, subject_ids, masks = load_fold(feature_file, feature_key, metadata)

    x_train, y_train = x[masks["train"]], y[masks["train"]]
    x_val, y_val = x[masks["val"]], y[masks["val"]]
    x_test, y_test = x[masks["test"]], y[masks["test"]]

    alpha, val_score, val_pred, alpha_table = select_alpha(
        x_train, y_train, x_val, y_val, alphas
    )
    val_metrics = compute_metrics(y_val, val_pred)

    if refit_train_val:
        x_fit = np.concatenate([x_train, x_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
    else:
        x_fit, y_fit = x_train, y_train

    final_model = make_model(alpha)
    final_model.fit(x_fit, y_fit)
    test_pred = final_model.predict(x_test)

    test_sid = int(subject_ids[masks["test"]][0])
    fold_name = f"test_{test_sid:03d}"
    fold_dir = output_dir / "folds" / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    alpha_table.to_csv(fold_dir / "alpha_search.csv", index=False)

    val_table = pd.DataFrame(
        {
            "subject_id": subject_ids[masks["val"]],
            "matlab_subject_id": subject_ids[masks["val"]] + 1,
            **{f"true_{name}": y_val[:, j] for j, name in enumerate(TARGET_NAMES)},
            **{f"pred_{name}": val_pred[:, j] for j, name in enumerate(TARGET_NAMES)},
        }
    )
    val_table.to_csv(fold_dir / "validation_predictions.csv", index=False)

    test_table = pd.DataFrame(
        {
            "feature_file": [str(feature_file)],
            "subject_id": [test_sid],
            "matlab_subject_id": [test_sid + 1],
            "selected_alpha": [alpha],
            **{f"true_{name}": [y_test[0, j]] for j, name in enumerate(TARGET_NAMES)},
            **{f"pred_{name}": [test_pred[0, j]] for j, name in enumerate(TARGET_NAMES)},
        }
    )
    test_table.to_csv(fold_dir / "test_prediction.csv", index=False)

    if save_fold_plot:
        plot_prediction_scatter(
            y_val,
            val_pred,
            fold_dir / "validation_scatter",
            f"Validation thickness probe: test subject {test_sid}",
            metrics=val_metrics,
            plot_format=plot_format,
            dpi=plot_dpi,
            thickness_unit=thickness_unit,
        )

    result = FoldResult(
        feature_file=str(feature_file),
        test_subject_id=test_sid,
        test_matlab_subject_id=test_sid + 1,
        selected_alpha=float(alpha),
        validation_standardized_rmse=float(val_score),
        validation_metrics=val_metrics,
        test_true=y_test[0].astype(float).tolist(),
        test_pred=test_pred[0].astype(float).tolist(),
        train_subject_ids=subject_ids[masks["train"]].astype(int).tolist(),
        val_subject_ids=subject_ids[masks["val"]].astype(int).tolist(),
    )

    fold_json = {
        **{k: v for k, v in asdict(result).items() if k != "validation_metrics"},
        "validation_metrics": metrics_to_json(result.validation_metrics),
        "refit_train_val": bool(refit_train_val),
    }
    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(fold_json, f, indent=2, allow_nan=False)

    return result, val_table, test_table


def baseline_predictions_for_fold(
    feature_file: Path,
    feature_key: str,
    metadata: np.ndarray,
    refit_train_val: bool,
) -> Tuple[int, np.ndarray, np.ndarray]:
    _, y, subject_ids, masks = load_fold(feature_file, feature_key, metadata)
    fit_mask = masks["train"] | masks["val"] if refit_train_val else masks["train"]
    mean_pred = y[fit_mask].mean(axis=0, keepdims=True)
    return (
        int(subject_ids[masks["test"]][0]),
        y[masks["test"]],
        mean_pred,
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_csv, args.metadata_start_col)
    feature_files = resolve_feature_files(args.feature_npz, args.feature_glob)
    alphas = parse_alphas(args.alphas)
    refit_train_val = not args.no_refit_train_val

    print(f"Loaded metadata: {metadata.shape} (scalp, skull, csf)")
    print(f"Matched {len(feature_files)} feature fold(s)")
    print(f"Feature key: {args.feature_key}")
    print(f"Refit selected probe on train+val: {refit_train_val}")

    fold_results: List[FoldResult] = []
    test_tables = []
    seen_test_subjects = set()

    for idx, feature_file in enumerate(feature_files, start=1):
        result, _, test_table = fit_one_fold(
            feature_file=feature_file,
            feature_key=args.feature_key,
            metadata=metadata,
            alphas=alphas,
            refit_train_val=refit_train_val,
            output_dir=output_dir,
            save_fold_plot=args.save_fold_plots,
            plot_format=args.plot_format,
            plot_dpi=args.plot_dpi,
            thickness_unit=args.thickness_unit,
        )
        if result.test_subject_id in seen_test_subjects:
            raise ValueError(
                f"Duplicate held-out test subject {result.test_subject_id} across feature files"
            )
        seen_test_subjects.add(result.test_subject_id)
        fold_results.append(result)
        test_tables.append(test_table)
        print(
            f"[{idx:03d}/{len(feature_files):03d}] test_id={result.test_subject_id:3d} "
            f"alpha={result.selected_alpha:g} "
            f"val_std_RMSE={result.validation_standardized_rmse:.4f} "
            f"true={np.round(result.test_true, 4)} "
            f"pred={np.round(result.test_pred, 4)}"
        )

    predictions = pd.concat(test_tables, ignore_index=True).sort_values("subject_id")
    y_true = predictions[[f"true_{name}" for name in TARGET_NAMES]].to_numpy()
    y_pred = predictions[[f"pred_{name}" for name in TARGET_NAMES]].to_numpy()
    aggregate_metrics = compute_metrics(y_true, y_pred)

    baseline_rows = []
    for feature_file in feature_files:
        sid, true, pred = baseline_predictions_for_fold(
            feature_file, args.feature_key, metadata, refit_train_val
        )
        baseline_rows.append((sid, true[0], pred[0]))
    baseline_rows.sort(key=lambda x: x[0])
    baseline_true = np.stack([row[1] for row in baseline_rows])
    baseline_pred = np.stack([row[2] for row in baseline_rows])
    baseline_metrics = compute_metrics(baseline_true, baseline_pred)
    for j, name in enumerate(TARGET_NAMES):
        predictions[f"baseline_pred_{name}"] = baseline_pred[:, j]
    predictions.to_csv(output_dir / "heldout_test_predictions.csv", index=False)

    report = {
        "description": (
            "Frozen z_struct Ridge probe. Alpha selected on each fold's validation "
            "subjects; reported aggregate test metrics use only held-out test subjects."
        ),
        "num_folds": len(feature_files),
        "num_unique_test_subjects": len(seen_test_subjects),
        "feature_key": args.feature_key,
        "metadata_csv": args.metadata_csv,
        "metadata_columns_zero_based": [
            args.metadata_start_col,
            args.metadata_start_col + 1,
            args.metadata_start_col + 2,
        ],
        "target_names": list(TARGET_NAMES),
        "refit_train_val": refit_train_val,
        "alphas": alphas.astype(float).tolist(),
        "heldout_test_metrics": metrics_to_json(aggregate_metrics),
        "train_mean_baseline_test_metrics": metrics_to_json(baseline_metrics),
        "folds": [
            {
                **{k: v for k, v in asdict(r).items() if k != "validation_metrics"},
                "validation_metrics": metrics_to_json(r.validation_metrics),
            }
            for r in fold_results
        ],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    saved_figures: List[Path] = []
    if len(feature_files) >= 2 and not args.no_plots:
        saved_figures.extend(plot_prediction_scatter(
            y_true,
            y_pred,
            output_dir / "heldout_test_scatter",
            f"Held-out test thickness predictions across {len(feature_files)} folds",
            metrics=aggregate_metrics,
            plot_format=args.plot_format,
            dpi=args.plot_dpi,
            thickness_unit=args.thickness_unit,
        ))
        saved_figures.extend(plot_probe_vs_baseline(
            aggregate_metrics,
            baseline_metrics,
            output_dir / "probe_vs_baseline",
            plot_format=args.plot_format,
            dpi=args.plot_dpi,
            thickness_unit=args.thickness_unit,
        ))

    summary_rows = []
    for name in TARGET_NAMES:
        probe = aggregate_metrics[name]
        base = baseline_metrics[name]
        summary_rows.append(
            {
                "target": name,
                "n_test_subjects": probe.n,
                "probe_mae": probe.mae,
                "baseline_mae": base.mae,
                "mae_improvement": base.mae - probe.mae,
                "probe_rmse": probe.rmse,
                "baseline_rmse": base.rmse,
                "probe_r2": probe.r2,
                "probe_pearson_r": probe.pearson_r,
                "probe_spearman_rho": probe.spearman_rho,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print("\nHeld-out test summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSaved outputs to: {output_dir}")
    if saved_figures:
        print("Saved figures:")
        for path in saved_figures:
            print(f"  {path}")
    if len(feature_files) == 1:
        print(
            "Note: one fold has only one test subject, so aggregate test R2/correlation "
            "are undefined. Use --feature_glob over multiple TEST_ID folds."
        )


if __name__ == "__main__":
    main()
