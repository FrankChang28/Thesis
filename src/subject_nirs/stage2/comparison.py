#!/usr/bin/env python3
"""Compare LOSO regression experiments for HC and StO2.

The script recursively discovers ``loso_results.csv`` files, reads the matching
``experiment_config.json`` when available, excludes incomplete LOSO runs by
default, and creates publication-ready figures and CSV summaries.

- DRS-only baseline
- DRS + layer-thickness features
- DRS + latent/DTOF features
- zero/shuffle control experiments

Typical usage
-------------
python -m subject_nirs.stage2.comparison \
    --root ./artifacts/stage2 \
    --output-dir ./results/tables/stage2/fusion_comparison

You can also specify experiments explicitly and override display labels:

python -m subject_nirs.stage2.comparison \
    --experiment ./artifacts/stage2/baseline/hc \
    --experiment "./artifacts/stage2/concatenation/finetune/metadata/hc=Tissue-thickness feature" \
    --experiment "./artifacts/stage2/residual/finetune/dtof/hc=DTOF descriptor" \
    --experiment ./artifacts/stage2/baseline/sto2 \
    --experiment "./artifacts/stage2/concatenation/finetune/metadata/sto2=Tissue-thickness feature" \
    --experiment "./artifacts/stage2/residual/finetune/dtof/sto2=DTOF descriptor" \
    --output-dir ./comparison

Outputs
-------
- all_fold_results.csv
- experiment_summary.csv
- paired_vs_baseline.csv
- conditional_effect_by_baseline_difficulty.csv
- heterogeneous_benefit_<method>.csv
- final_experiment_table.csv
- final_experiment_table_formatted.csv
- representative_loso_comparison.png
- heterogeneous_benefit_<method>.png (or an explicitly labelled preview fallback)

The thesis table reports subject-level median [Q1, Q3] RMSE and bias, plus
paired changes versus the DRS-only baseline. The single figure is restricted to one matched
fusion/training setting (residual + fine-tune by default), so feature-source
effects are not confounded with the fusion operator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .display import (
    BASELINE_DISPLAY,
    FEATURE_DISPLAY,
    FUSION_DISPLAY,
    STRATEGY_DISPLAY,
    TARGET_DISPLAY,
    target_display_scale,
    target_error_unit,
)

METRICS = ("RMSE", "MAE", "Bias", "AbsBias", "ErrorStd")


@dataclass(frozen=True)
class ExperimentSpec:
    directory: Path
    label_override: str | None = None


@dataclass(frozen=True)
class ExperimentMeta:
    directory: Path
    target_mode: str
    target_name: str
    training_mode: str
    feature_source: str
    structure_control: str
    fusion_method: str
    training_strategy: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LOSO comparison plots for HC/StO2 experiments."
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help=(
            "Root directory to scan recursively for loso_results.csv. "
            "May be supplied more than once. Defaults to ./artifacts/stage2."
        ),
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        metavar="PATH[=LABEL]",
        help=(
            "Explicit experiment directory or loso_results.csv path. "
            "Append '=LABEL' to override the inferred label."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./loso_result_comparison",
        help="Directory for figures and summary tables.",
    )
    parser.add_argument(
        "--baseline-label",
        default=BASELINE_DISPLAY,
        help="Exact display label used as the paired-comparison baseline.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not save vector PDF copies of figures.",
    )
    parser.add_argument(
        "--expected-subjects",
        type=int,
        default=154,
        help="Expected unique LOSO subjects; incomplete runs are skipped (default: 154).",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs with fewer than --expected-subjects subjects.",
    )
    parser.add_argument(
        "--representative-method",
        default="residual",
        choices=("concatenation", "film", "residual"),
        help="Fusion method shown in the representative figure.",
    )
    parser.add_argument(
        "--representative-strategy",
        default="finetune",
        choices=("finetune", "scratch"),
        help="Training strategy shown in the representative figure.",
    )
    parser.add_argument(
        "--heterogeneity-method",
        default="residual",
        choices=("concatenation", "film", "residual"),
        help=(
            "Fixed fusion method used for the 2x2 subject-level heterogeneous-"
            "benefit figure (default: residual)."
        ),
    )
    parser.add_argument(
        "--heterogeneity-fallback-method",
        default=None,
        choices=("concatenation", "film", "residual"),
        help=(
            "Optional whole-matrix preview fallback when --heterogeneity-method "
            "is incomplete. The fallback is never mixed cell-by-cell with the "
            "requested method and is labelled as a preview."
        ),
    )
    return parser.parse_args()


def split_experiment_arg(value: str) -> ExperimentSpec:
    """Parse PATH[=LABEL] while preserving ordinary paths containing '='."""
    raw = value.strip()
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return ExperimentSpec(candidate, None)

    if "=" in raw:
        path_text, label = raw.rsplit("=", 1)
        path = Path(path_text).expanduser()
        if path.exists():
            return ExperimentSpec(path, label.strip() or None)

    return ExperimentSpec(candidate, None)


def discover_experiments(args: argparse.Namespace) -> list[ExperimentSpec]:
    found: dict[Path, ExperimentSpec] = {}

    # 明確指定的 experiments
    for item in args.experiment:
        spec = split_experiment_arg(item)
        path = spec.directory.resolve()

        # 也允許直接指定 loso_results.csv
        if path.is_file() and path.name == "loso_results.csv":
            path = path.parent

        found[path] = ExperimentSpec(path, spec.label_override)

    # 決定是否還要遞迴掃描 root
    if args.root is not None:
        root_values = args.root
    elif args.experiment:
        # 已明確指定 experiment，不掃描預設 root
        root_values = []
    else:
        # 沒有指定 experiment 或 root，維持原本預設行為
        root_values = ["./artifacts/stage2"]

    skip_names = {"proc", "sys", "dev", "run", ".git", "__pycache__"}

    for root_text in root_values:
        root = Path(root_text).expanduser().resolve()

        if not root.exists():
            print(
                f"WARNING: scan root does not exist: {root}",
                file=sys.stderr,
            )
            continue

        if root.is_file() and root.name == "loso_results.csv":
            found.setdefault(
                root.parent,
                ExperimentSpec(root.parent),
            )
            continue

        def on_walk_error(exc: OSError) -> None:
            print(
                f"WARNING: skipped unreadable path: {exc}",
                file=sys.stderr,
            )

        for current, dirnames, filenames in os.walk(
            root,
            topdown=True,
            followlinks=False,
            onerror=on_walk_error,
        ):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in skip_names
            ]

            if "loso_results.csv" in filenames:
                directory = Path(current).resolve()
                found.setdefault(
                    directory,
                    ExperimentSpec(directory),
                )

    if not found:
        raise FileNotFoundError(
            "No experiments found. Use --root ROOT or --experiment PATH."
        )

    valid = []

    for spec in found.values():
        csv_path = spec.directory / "loso_results.csv"

        if csv_path.exists():
            valid.append(spec)
        else:
            print(
                f"WARNING: skipped {spec.directory}; "
                "loso_results.csv not found.",
                file=sys.stderr,
            )

    if not valid:
        raise FileNotFoundError(
            "No valid loso_results.csv files were found."
        )

    return sorted(valid, key=lambda spec: str(spec.directory))


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return {}


def infer_target_from_columns(columns: Iterable[str]) -> tuple[str, str]:
    columns = list(columns)
    if any("GM_hc" in col for col in columns):
        return "hc", "GM_hc"
    if any("GM_StO2" in col for col in columns):
        return "sto2", "GM_StO2"
    raise ValueError("Could not infer target from result columns.")


def normalized_target_mode(value: object, fallback_name: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"hc", "gm_hc"}:
        return "hc"
    if text in {"sto2", "gm_sto2", "sto₂"}:
        return "sto2"
    return "hc" if fallback_name == "GM_hc" else "sto2"


def infer_experiment_dimensions(
    config: dict,
    frame: pd.DataFrame,
    directory: Path,
) -> tuple[str, str, str, str, str]:
    training_mode = str(
        config.get(
            "training_mode",
            frame["training_mode"].iloc[-1] if "training_mode" in frame else "",
        )
    ).strip().lower()

    control = str(
        config.get(
            "structure_control",
            frame["structure_control"].iloc[-1]
            if "structure_control" in frame
            else "actual",
        )
    ).strip().lower()

    source = str(config.get("subject_feature_source", "")).strip().lower()
    path_parts = {part.lower() for part in directory.parts}
    path_text = "/".join(part.lower() for part in directory.parts)

    if training_mode == "baseline" or "baseline" in path_parts:
        return "baseline", "baseline", "none", "none", "scratch"

    if not source:
        if "metadata" in path_text or "handcraft" in path_text:
            source = "metadata"
        elif "dtof" in path_text or "latent" in path_text:
            source = "latent"
        else:
            source = "subject feature"

    if source == "dtof":
        source = "latent"

    fusion_method = str(config.get("fusion_method", "")).strip().lower()
    if not fusion_method:
        fusion_method = next(
            (item for item in ("concatenation", "film", "residual") if item in path_parts),
            "unknown",
        )

    baseline_init = str(config.get("baseline_init", "")).strip().lower()
    if baseline_init == "scratch" or "scratch" in path_parts:
        training_strategy = "scratch"
    elif baseline_init == "pretrained" or "finetune" in path_parts:
        training_strategy = "finetune"
    else:
        training_strategy = "unknown"

    if not training_mode:
        training_mode = fusion_method

    return training_mode, source, control or "actual", fusion_method, training_strategy


def default_label(
    training_mode: str,
    source: str,
    control: str,
    fusion_method: str,
    training_strategy: str,
) -> str:
    if training_mode == "baseline" or source == "baseline":
        return BASELINE_DISPLAY

    label = FEATURE_DISPLAY.get(source, source.replace("_", " ").title())

    if control not in {"", "actual", "none", "nan"}:
        label = f"{label} ({control})"
    method = FUSION_DISPLAY.get(fusion_method, fusion_method.title())
    strategy = STRATEGY_DISPLAY.get(training_strategy, training_strategy.title())
    return f"{label} | {method} | {strategy}"


def metric_column(frame: pd.DataFrame, target_name: str, metric: str) -> str:
    exact = f"test_{target_name}_{metric}"
    if exact in frame.columns:
        return exact

    candidates = [
        col
        for col in frame.columns
        if col.startswith("test_") and col.endswith(f"_{metric}")
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(
        f"Expected column {exact!r}. Candidates for {metric}: {candidates}"
    )


def load_experiment(spec: ExperimentSpec) -> tuple[ExperimentMeta, pd.DataFrame]:
    directory = spec.directory.resolve()
    csv_path = directory / "loso_results.csv"
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Empty results file: {csv_path}")

    config = read_json_if_exists(directory / "experiment_config.json")
    inferred_mode, inferred_target_name = infer_target_from_columns(frame.columns)
    target_mode = normalized_target_mode(config.get("target_mode"), inferred_target_name)
    target_name = "GM_hc" if target_mode == "hc" else "GM_StO2"

    training_mode, source, control, fusion_method, training_strategy = (
        infer_experiment_dimensions(config, frame, directory)
    )
    label = spec.label_override or default_label(
        training_mode, source, control, fusion_method, training_strategy
    )

    subject_col = (
        "original_matlab_subj_id"
        if "original_matlab_subj_id" in frame.columns
        else "test_id"
    )
    if subject_col not in frame.columns:
        raise KeyError(
            f"{csv_path} has neither original_matlab_subj_id nor test_id."
        )

    before = len(frame)
    frame = frame.dropna(subset=[subject_col]).copy()
    frame = frame.drop_duplicates(subset=[subject_col], keep="last")
    removed = before - len(frame)
    if removed:
        print(
            f"WARNING: {directory.name}: removed {removed} duplicate/invalid fold rows "
            f"(kept the last row per subject).",
            file=sys.stderr,
        )

    tidy = pd.DataFrame(
        {
            "experiment_dir": str(directory),
            "experiment": label,
            "target": target_mode,
            "target_display": TARGET_DISPLAY[target_mode],
            "training_mode": training_mode,
            "feature_source": source,
            "structure_control": control,
            "fusion_method": fusion_method,
            "training_strategy": training_strategy,
            "subject_id": pd.to_numeric(frame[subject_col], errors="coerce").astype("Int64"),
        }
    )

    for metric in METRICS:
        col = metric_column(frame, target_name, metric)
        tidy[metric] = pd.to_numeric(frame[col], errors="coerce")

    # Keep model artifacts in native units, but make every analysis/table/plot
    # emitted by this reporting module thesis-facing. StO₂ errors are therefore
    # expressed on the percentage scale (%) instead of as fractions.
    display_scale = target_display_scale(target_mode)
    if display_scale != 1.0:
        tidy.loc[:, list(METRICS)] = tidy.loc[:, list(METRICS)] * display_scale
    tidy["error_unit"] = target_error_unit(target_mode)

    tidy = tidy.dropna(subset=["subject_id", "RMSE", "MAE", "Bias"]).copy()
    tidy["subject_id"] = tidy["subject_id"].astype(int)
    tidy["AbsBias"] = tidy["AbsBias"].fillna(tidy["Bias"].abs())

    meta = ExperimentMeta(
        directory=directory,
        target_mode=target_mode,
        target_name=target_name,
        training_mode=training_mode,
        feature_source=source,
        structure_control=control,
        fusion_method=fusion_method,
        training_strategy=training_strategy,
        label=label,
    )
    return meta, tidy


def label_sort_key(label: str) -> tuple[int, str]:
    text = label.lower()
    if text in {"baseline", "drs only", "drs-only", "drs-only baseline"}:
        priority = 0
    elif "metadata" in text or "hand" in text:
        priority = 1
    elif "latent" in text or "dtof" in text:
        priority = 2
    elif "zero" in text:
        priority = 3
    elif "shuffle" in text:
        priority = 4
    else:
        priority = 5
    return priority, text


def experiment_order(frame: pd.DataFrame, target: str | None = None) -> list[str]:
    subset = frame if target is None else frame[frame["target"] == target]
    labels = subset["experiment"].drop_duplicates().tolist()
    return sorted(labels, key=label_sort_key)


def target_order(frame: pd.DataFrame) -> list[str]:
    present = set(frame["target"])
    return [target for target in ("hc", "sto2") if target in present]


def iqr(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def summarize_experiments(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, experiment), group in frame.groupby(["target", "experiment"], sort=False):
        row: dict[str, object] = {
            "target": target,
            "experiment": experiment,
            "n_subjects": int(group["subject_id"].nunique()),
        }
        for metric in METRICS:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_IQR"] = iqr(values)
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_pvalue(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0 or np.allclose(delta, 0.0):
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue)
    except (ImportError, ValueError):
        return float("nan")


def paired_rank_biserial(delta: np.ndarray) -> float:
    """Return paired rank-biserial effect; positive values favor the experiment."""
    values = np.asarray(delta, dtype=float)
    values = values[np.isfinite(values) & ~np.isclose(values, 0.0)]
    if values.size == 0:
        return float("nan")
    ranks = pd.Series(np.abs(values)).rank(method="average").to_numpy(dtype=float)
    total = float(ranks.sum())
    improvement = float(ranks[values < 0].sum())
    worsening = float(ranks[values > 0].sum())
    return (improvement - worsening) / total


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Holm-adjust finite p-values while preserving the input index."""
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    finite = pvalues[np.isfinite(pvalues.to_numpy(dtype=float))].sort_values()
    count = len(finite)
    running = 0.0
    for rank, (index, value) in enumerate(finite.items()):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted.loc[index] = running
    return adjusted


def resolve_baseline_label(target_frame: pd.DataFrame, requested_label: str) -> str | None:
    """Resolve the baseline label, preferring the user's label then metadata."""
    labels = set(target_frame["experiment"].astype(str))
    if requested_label in labels:
        return requested_label

    candidates = (
        target_frame.loc[target_frame["training_mode"] == "baseline", "experiment"]
        .drop_duplicates()
        .tolist()
    )
    if len(candidates) == 1:
        print(
            f"WARNING: baseline label {requested_label!r} not found; "
            f"using inferred baseline {candidates[0]!r}.",
            file=sys.stderr,
        )
        return str(candidates[0])
    return None


def paired_comparisons(frame: pd.DataFrame, baseline_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []

    for target in target_order(frame):
        target_frame = frame[frame["target"] == target]
        active_baseline_label = resolve_baseline_label(target_frame, baseline_label)
        if active_baseline_label is None:
            print(
                f"WARNING: target={target}: could not identify exactly one baseline; "
                "paired comparison skipped.",
                file=sys.stderr,
            )
            continue

        baseline = target_frame[target_frame["experiment"] == active_baseline_label]
        baseline = baseline.set_index("subject_id")
        for experiment in experiment_order(target_frame, target):
            if experiment == active_baseline_label:
                continue
            current = target_frame[target_frame["experiment"] == experiment].set_index("subject_id")
            common = baseline.index.intersection(current.index)
            if len(common) == 0:
                continue

            detail = pd.DataFrame(
                {
                    "target": target,
                    "experiment": experiment,
                    "subject_id": common.astype(int),
                    "baseline_RMSE": baseline.loc[common, "RMSE"].to_numpy(),
                    "experiment_RMSE": current.loc[common, "RMSE"].to_numpy(),
                    "baseline_AbsBias": baseline.loc[common, "AbsBias"].to_numpy(),
                    "experiment_AbsBias": current.loc[common, "AbsBias"].to_numpy(),
                }
            )
            detail["delta_RMSE"] = detail["experiment_RMSE"] - detail["baseline_RMSE"]
            detail["delta_RMSE_percent"] = (
                detail["delta_RMSE"] / detail["baseline_RMSE"].replace(0, np.nan) * 100.0
            )
            detail["delta_AbsBias"] = (
                detail["experiment_AbsBias"] - detail["baseline_AbsBias"]
            )
            detail_rows.append(detail)

            delta = detail["delta_RMSE"].to_numpy(dtype=float)
            delta_abs_bias = detail["delta_AbsBias"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "target": target,
                    "experiment": experiment,
                    "baseline": active_baseline_label,
                    "n_paired_subjects": int(len(detail)),
                    "RMSE_win_rate": float(np.mean(delta < 0)),
                    "RMSE_tie_rate": float(np.mean(np.isclose(delta, 0))),
                    "delta_RMSE_mean": float(np.nanmean(delta)),
                    "delta_RMSE_median": float(np.nanmedian(delta)),
                    "delta_RMSE_Q1": float(np.nanpercentile(delta, 25)),
                    "delta_RMSE_Q3": float(np.nanpercentile(delta, 75)),
                    "delta_RMSE_IQR": float(
                        np.nanpercentile(delta, 75) - np.nanpercentile(delta, 25)
                    ),
                    "delta_RMSE_percent_median": float(
                        np.nanmedian(detail["delta_RMSE_percent"])
                    ),
                    "delta_AbsBias_median": float(np.nanmedian(delta_abs_bias)),
                    "wilcoxon_RMSE_p": wilcoxon_pvalue(delta),
                    "paired_rank_biserial": paired_rank_biserial(delta),
                }
            )

    detail_frame = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        summary_frame["wilcoxon_RMSE_p_holm"] = summary_frame.groupby(
            "target", group_keys=False
        )["wilcoxon_RMSE_p"].apply(holm_adjust)
    return detail_frame, summary_frame


def build_final_experiment_table(
    frame: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact thesis table from subject-level LOSO results."""
    rows: list[dict[str, object]] = []
    for (target, experiment), group in frame.groupby(["target", "experiment"], sort=False):
        rmse = group["RMSE"].dropna()
        bias = group["Bias"].dropna()
        first = group.iloc[0]
        row: dict[str, object] = {
            "target": target,
            "feature_source": first["feature_source"],
            "fusion_method": first["fusion_method"],
            "training_strategy": first["training_strategy"],
            "experiment": experiment,
            "n_subjects": int(group["subject_id"].nunique()),
            "rmse_median": float(rmse.median()),
            "rmse_q1": float(rmse.quantile(0.25)),
            "rmse_q3": float(rmse.quantile(0.75)),
            "bias_median": float(bias.median()),
            "bias_q1": float(bias.quantile(0.25)),
            "bias_q3": float(bias.quantile(0.75)),
            "n_paired_subjects": np.nan,
            "delta_rmse_median": np.nan,
            "delta_rmse_q1": np.nan,
            "delta_rmse_q3": np.nan,
            "improved_subjects_pct": np.nan,
            "paired_rank_biserial": np.nan,
            "wilcoxon_p": np.nan,
        }
        paired = paired_summary[
            (paired_summary["target"] == target)
            & (paired_summary["experiment"] == experiment)
        ]
        if not paired.empty:
            result = paired.iloc[0]
            row.update(
                {
                    "n_paired_subjects": int(result["n_paired_subjects"]),
                    "delta_rmse_median": float(result["delta_RMSE_median"]),
                    "improved_subjects_pct": 100.0 * float(result["RMSE_win_rate"]),
                    "paired_rank_biserial": float(result["paired_rank_biserial"]),
                    "wilcoxon_p": float(result["wilcoxon_RMSE_p"]),
                }
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    # Delta quartiles are retained in paired_summary to avoid reconstructing pairs here.
    if not paired_summary.empty:
        lookup = paired_summary.set_index(["target", "experiment"])
        for index, row in table.iterrows():
            key = (row["target"], row["experiment"])
            if key in lookup.index:
                paired = lookup.loc[key]
                table.loc[index, "delta_rmse_q1"] = paired["delta_RMSE_Q1"]
                table.loc[index, "delta_rmse_q3"] = paired["delta_RMSE_Q3"]

    table["wilcoxon_p_holm"] = np.nan
    for target in table["target"].drop_duplicates():
        mask = (table["target"] == target) & table["wilcoxon_p"].notna()
        table.loc[mask, "wilcoxon_p_holm"] = holm_adjust(table.loc[mask, "wilcoxon_p"])

    source_order = {"baseline": 0, "latent": 1, "metadata": 2}
    method_order = {"none": 0, "concatenation": 1, "film": 2, "residual": 3}
    strategy_order = {"scratch": 0, "finetune": 1}
    table = table.assign(
        _target=table["target"].map({"hc": 0, "sto2": 1}).fillna(9),
        _source=table["feature_source"].map(source_order).fillna(9),
        _method=table["fusion_method"].map(method_order).fillna(9),
        _strategy=table["training_strategy"].map(strategy_order).fillna(9),
    ).sort_values(["_target", "_source", "_method", "_strategy", "experiment"])
    return table.drop(columns=["_target", "_source", "_method", "_strategy"]).reset_index(drop=True)


def format_final_experiment_table(table: pd.DataFrame) -> pd.DataFrame:
    """Create a human-readable CSV without sacrificing the numeric master table."""
    formatted = table[[
        "target", "feature_source", "fusion_method", "training_strategy",
        "experiment", "n_subjects",
    ]].copy()
    formatted["target"] = formatted["target"].map(TARGET_DISPLAY)
    formatted["feature_source"] = formatted["feature_source"].map(
        lambda value: FEATURE_DISPLAY.get(value, value)
    )
    formatted["fusion_method"] = formatted["fusion_method"].map(
        lambda value: FUSION_DISPLAY.get(value, value)
    )
    formatted["training_strategy"] = formatted["training_strategy"].map(
        lambda value: STRATEGY_DISPLAY.get(value, value)
    )
    formatted["error unit"] = table["target"].map(target_error_unit)
    formatted["RMSE median [Q1, Q3]"] = table.apply(
        lambda row: f"{row.rmse_median:.4g} [{row.rmse_q1:.4g}, {row.rmse_q3:.4g}]", axis=1
    )
    formatted["Bias median [Q1, Q3]"] = table.apply(
        lambda row: f"{row.bias_median:.4g} [{row.bias_q1:.4g}, {row.bias_q3:.4g}]", axis=1
    )
    formatted["paired ΔRMSE median [Q1, Q3]"] = table.apply(
        lambda row: "—" if pd.isna(row.delta_rmse_median) else
        f"{row.delta_rmse_median:+.4g} [{row.delta_rmse_q1:+.4g}, {row.delta_rmse_q3:+.4g}]",
        axis=1,
    )
    formatted["improved subjects (%)"] = table["improved_subjects_pct"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.1f}"
    )
    formatted["paired rank-biserial"] = table["paired_rank_biserial"].map(
        lambda value: "—" if pd.isna(value) else f"{value:+.3f}"
    )
    formatted["Holm-adjusted p"] = table["wilcoxon_p_holm"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.3g}"
    )
    return formatted


DIFFICULTY_ORDER = ("Q1", "Q2", "Q3", "Q4")
DIFFICULTY_DISPLAY = {
    "Q1": "Q1\neasiest",
    "Q2": "Q2",
    "Q3": "Q3",
    "Q4": "Q4\nhardest",
}


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260722,
) -> tuple[float, float, float]:
    """Return the sample mean and a percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value, value

    rng = np.random.default_rng(seed + values.size)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    boot_means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(values.mean()),
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1.0 - alpha)),
    )


def bootstrap_difference_of_means_ci(
    first: np.ndarray,
    second: np.ndarray,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260723,
) -> tuple[float, float, float]:
    """Return mean(second) - mean(first) and its bootstrap interval."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if first.size == 0 or second.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed + first.size + 17 * second.size)
    first_idx = rng.integers(0, first.size, size=(n_bootstrap, first.size))
    second_idx = rng.integers(0, second.size, size=(n_bootstrap, second.size))
    boot_diff = second[second_idx].mean(axis=1) - first[first_idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    observed = float(second.mean() - first.mean())
    return (
        observed,
        float(np.quantile(boot_diff, alpha)),
        float(np.quantile(boot_diff, 1.0 - alpha)),
    )


def conditional_effect_by_baseline_difficulty(
    paired_detail: pd.DataFrame,
    meaningful_delta: float = 3.0,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """Summarize feature value within common baseline-RMSE quartiles.

    For each target, Q1--Q4 are defined once from the baseline RMSE of every
    available subject, then reused for all non-baseline experiments. This makes
    the tissue-thickness feature and DTOF descriptor directly comparable in
    the same panel.

    The quartiles remain descriptive because they use held-out-subject DRS-only
    test RMSE. They demonstrate effect heterogeneity, not a deployable selector.
    """
    if paired_detail.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for target, target_detail in paired_detail.groupby("target", sort=False):
        reference = (
            target_detail[["subject_id", "baseline_RMSE"]]
            .dropna()
            .groupby("subject_id", as_index=False)["baseline_RMSE"]
            .median()
        )
        if len(reference) < 4:
            continue

        ranked = reference["baseline_RMSE"].rank(method="first")
        reference["difficulty_quartile"] = pd.qcut(
            ranked,
            q=4,
            labels=list(DIFFICULTY_ORDER),
        ).astype(str)

        for experiment, original in target_detail.groupby("experiment", sort=False):
            group = original.dropna(subset=["baseline_RMSE", "delta_RMSE"]).copy()
            group = group.drop(columns=["difficulty_quartile"], errors="ignore")
            group = group.merge(
                reference[["subject_id", "difficulty_quartile"]],
                on="subject_id",
                how="inner",
                validate="many_to_one",
            )
            if group.empty:
                continue

            experiment_rows: list[dict[str, object]] = []
            quartile_values: dict[str, np.ndarray] = {}
            for quartile in DIFFICULTY_ORDER:
                current = group[group["difficulty_quartile"] == quartile]
                delta = current["delta_RMSE"].to_numpy(dtype=float)
                quartile_values[quartile] = delta
                mean_delta, ci_low, ci_high = bootstrap_mean_ci(
                    delta,
                    n_bootstrap=n_bootstrap,
                    seed=(
                        20260722
                        + 101 * DIFFICULTY_ORDER.index(quartile)
                        + 1009 * len(experiment_rows)
                    ),
                )
                experiment_rows.append(
                    {
                        "target": target,
                        "experiment": experiment,
                        "difficulty_quartile": quartile,
                        "n_subjects": int(len(current)),
                        "baseline_RMSE_min": (
                            float(current["baseline_RMSE"].min())
                            if not current.empty
                            else float("nan")
                        ),
                        "baseline_RMSE_max": (
                            float(current["baseline_RMSE"].max())
                            if not current.empty
                            else float("nan")
                        ),
                        "delta_RMSE_mean": mean_delta,
                        "delta_RMSE_ci_low": ci_low,
                        "delta_RMSE_ci_high": ci_high,
                        "delta_RMSE_median": (
                            float(np.median(delta)) if delta.size else float("nan")
                        ),
                        "RMSE_win_rate": (
                            float(np.mean(delta < 0.0)) if delta.size else float("nan")
                        ),
                        "large_gain_rate": (
                            float(np.mean(delta <= -meaningful_delta))
                            if delta.size
                            else float("nan")
                        ),
                        "large_harm_rate": (
                            float(np.mean(delta >= meaningful_delta))
                            if delta.size
                            else float("nan")
                        ),
                        "meaningful_delta_threshold": float(meaningful_delta),
                    }
                )

            hard_easy, hard_easy_low, hard_easy_high = (
                bootstrap_difference_of_means_ci(
                    quartile_values.get("Q1", np.array([], dtype=float)),
                    quartile_values.get("Q4", np.array([], dtype=float)),
                    n_bootstrap=n_bootstrap,
                )
            )
            for row in experiment_rows:
                row["hard_minus_easy_mean_delta"] = hard_easy
                row["hard_minus_easy_ci_low"] = hard_easy_low
                row["hard_minus_easy_ci_high"] = hard_easy_high
            rows.extend(experiment_rows)

    return pd.DataFrame(rows)


HETEROGENEITY_TARGETS = ("hc", "sto2")
HETEROGENEITY_SOURCES = ("latent", "metadata")
HETEROGENEITY_STRATEGIES = ("scratch", "finetune")
HETEROGENEITY_SOURCE_DISPLAY = {
    "latent": FEATURE_DISPLAY["latent"],
    "metadata": FEATURE_DISPLAY["metadata"],
}
HETEROGENEITY_STRATEGY_DISPLAY = {
    "scratch": STRATEGY_DISPLAY["scratch"],
    "finetune": STRATEGY_DISPLAY["finetune"],
}


def heterogeneous_benefit_missing_cells(
    frame: pd.DataFrame,
    fusion_method: str,
) -> list[str]:
    """Return missing cells for the fixed-method 2x2 heterogeneity figure."""
    missing: list[str] = []
    for target in HETEROGENEITY_TARGETS:
        baseline_count = frame.loc[
            (frame["target"] == target)
            & (frame["training_mode"] == "baseline"),
            "experiment",
        ].nunique()
        if baseline_count != 1:
            missing.append(f"{target}/baseline")

        for strategy in HETEROGENEITY_STRATEGIES:
            for source in HETEROGENEITY_SOURCES:
                count = frame.loc[
                    (frame["target"] == target)
                    & (frame["fusion_method"] == fusion_method)
                    & (frame["training_strategy"] == strategy)
                    & (frame["feature_source"] == source)
                    & (frame["structure_control"] == "actual"),
                    "experiment",
                ].nunique()
                if count == 0:
                    missing.append(f"{target}/{strategy}/{source}")
                elif count > 1:
                    raise ValueError(
                        "Multiple experiments occupy heterogeneity cell "
                        f"{target}/{fusion_method}/{strategy}/{source}."
                    )
    return missing


def resolve_heterogeneous_benefit_method(
    frame: pd.DataFrame,
    requested_method: str,
    fallback_method: str | None = None,
) -> tuple[str | None, bool, list[str]]:
    """Choose one complete method matrix without mixing operators across cells."""
    missing = heterogeneous_benefit_missing_cells(frame, requested_method)
    if not missing:
        return requested_method, False, []

    if fallback_method is not None and fallback_method != requested_method:
        fallback_missing = heterogeneous_benefit_missing_cells(frame, fallback_method)
        if not fallback_missing:
            return fallback_method, True, missing

    return None, False, missing


def build_heterogeneous_benefit_detail(
    frame: pd.DataFrame,
    paired_detail: pd.DataFrame,
    fusion_method: str,
) -> pd.DataFrame:
    """Attach experiment factors and common DRS-only difficulty quartiles."""
    missing = heterogeneous_benefit_missing_cells(frame, fusion_method)
    if missing:
        raise ValueError(
            f"Incomplete {fusion_method} heterogeneity matrix: " + ", ".join(missing)
        )

    experiment_meta = (
        frame.loc[
            (frame["fusion_method"] == fusion_method)
            & (frame["feature_source"].isin(HETEROGENEITY_SOURCES))
            & (frame["training_strategy"].isin(HETEROGENEITY_STRATEGIES))
            & (frame["structure_control"] == "actual"),
            [
                "target",
                "experiment",
                "feature_source",
                "training_strategy",
                "fusion_method",
            ],
        ]
        .drop_duplicates()
    )
    detail = paired_detail.merge(
        experiment_meta,
        on=["target", "experiment"],
        how="inner",
        validate="many_to_one",
    )

    pieces: list[pd.DataFrame] = []
    for target in HETEROGENEITY_TARGETS:
        current = detail[detail["target"] == target].copy()
        reference = (
            current[["subject_id", "baseline_RMSE"]]
            .drop_duplicates("subject_id")
            .sort_values("subject_id")
        )
        if len(reference) < 4:
            raise ValueError(f"Target {target} has fewer than four paired subjects.")
        reference["difficulty_quartile"] = pd.qcut(
            reference["baseline_RMSE"].rank(method="first"),
            q=4,
            labels=list(DIFFICULTY_ORDER),
        ).astype(str)
        current = current.merge(
            reference[["subject_id", "difficulty_quartile"]],
            on="subject_id",
            how="inner",
            validate="many_to_one",
        )
        pieces.append(current)

    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(
        ["target", "training_strategy", "feature_source", "subject_id"]
    ).reset_index(drop=True)


def summarize_heterogeneous_benefit(detail: pd.DataFrame) -> pd.DataFrame:
    """Summarize subject-level delta RMSE within each common difficulty quartile."""
    rows: list[dict[str, object]] = []
    grouped = detail.groupby(
        [
            "target",
            "fusion_method",
            "training_strategy",
            "feature_source",
            "difficulty_quartile",
        ],
        sort=False,
        observed=True,
    )
    for keys, group in grouped:
        target, method, strategy, source, quartile = keys
        values = group["delta_RMSE"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "target": target,
                "fusion_method": method,
                "training_strategy": strategy,
                "feature_source": source,
                "difficulty_quartile": quartile,
                "n_subjects": int(values.size),
                "delta_RMSE_median": float(np.median(values)),
                "delta_RMSE_Q1": float(np.percentile(values, 25)),
                "delta_RMSE_Q3": float(np.percentile(values, 75)),
                "improved_subjects_pct": float(100.0 * np.mean(values < 0.0)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["target"] = pd.Categorical(
        summary["target"],
        categories=list(HETEROGENEITY_TARGETS),
        ordered=True,
    )
    summary["training_strategy"] = pd.Categorical(
        summary["training_strategy"],
        categories=list(HETEROGENEITY_STRATEGIES),
        ordered=True,
    )
    summary["feature_source"] = pd.Categorical(
        summary["feature_source"],
        categories=list(HETEROGENEITY_SOURCES),
        ordered=True,
    )
    summary["difficulty_quartile"] = pd.Categorical(
        summary["difficulty_quartile"],
        categories=list(DIFFICULTY_ORDER),
        ordered=True,
    )
    return summary.sort_values(
        ["target", "training_strategy", "feature_source", "difficulty_quartile"]
    ).reset_index(drop=True)

def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
        }
    )


def color_map(labels: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % 10) for i, label in enumerate(labels)}


def deterministic_jitter(n: int, width: float = 0.10) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    rng = np.random.default_rng(20260720 + n)
    return rng.uniform(-width, width, size=n)


def create_axes_for_targets(targets: Sequence[str], height: float = 4.4):
    fig, axes = plt.subplots(1, len(targets), figsize=(6.0 * len(targets), height), squeeze=False)
    return fig, axes[0]


def save_figure(fig, output_dir: Path, stem: str, dpi: int, save_pdf: bool) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")


def plot_heterogeneous_benefit(
    detail: pd.DataFrame,
    output_dir: Path,
    fusion_method: str,
    dpi: int,
    *,
    preview_for: str | None = None,
) -> Path:
    """Plot subject-level delta RMSE by DRS-only difficulty in a 2x2 layout."""
    source_colors = {
        "latent": "#3B82C4",
        "metadata": "#E68A3F",
    }
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.6, 8.4),
        sharex=True,
        sharey="col",
        gridspec_kw={"hspace": 0.22, "wspace": 0.20},
    )
    base_x = np.arange(1, 5, dtype=float)
    source_offsets = {"latent": -0.18, "metadata": 0.18}
    panel_labels = iter("ABCD")

    for row, strategy in enumerate(HETEROGENEITY_STRATEGIES):
        for col, target in enumerate(HETEROGENEITY_TARGETS):
            ax = axes[row, col]
            panel = next(panel_labels)
            subset = detail[
                (detail["target"] == target)
                & (detail["training_strategy"] == strategy)
            ]

            for source in HETEROGENEITY_SOURCES:
                positions = base_x + source_offsets[source]
                values = []
                subject_ids = []
                for quartile in DIFFICULTY_ORDER:
                    current = subset[
                        (subset["feature_source"] == source)
                        & (subset["difficulty_quartile"] == quartile)
                    ].sort_values("subject_id")
                    values.append(current["delta_RMSE"].to_numpy(dtype=float))
                    subject_ids.append(current["subject_id"].to_numpy(dtype=int))

                bp = ax.boxplot(
                    values,
                    positions=positions,
                    widths=0.28,
                    whis=(5, 95),
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    medianprops={"color": "black", "linewidth": 1.4},
                    whiskerprops={"color": source_colors[source], "linewidth": 1.0},
                    capprops={"color": source_colors[source], "linewidth": 1.0},
                    boxprops={
                        "facecolor": source_colors[source],
                        "edgecolor": source_colors[source],
                        "linewidth": 1.0,
                        "alpha": 0.32,
                    },
                )
                del bp

                for position, current_values, current_ids in zip(
                    positions, values, subject_ids
                ):
                    if current_values.size == 0:
                        continue
                    order = np.argsort(current_ids)
                    ax.scatter(
                        position + deterministic_jitter(len(current_values), width=0.075),
                        current_values[order],
                        s=12,
                        alpha=0.30,
                        color=source_colors[source],
                        edgecolors="none",
                        rasterized=True,
                        zorder=2,
                    )

            ax.axhline(0.0, color="0.20", linewidth=1.1, linestyle="--", zorder=1)
            ax.set_title(
                f"{panel}  {HETEROGENEITY_STRATEGY_DISPLAY[strategy]} — "
                f"{TARGET_DISPLAY[target]}",
                loc="left",
                fontweight="bold",
            )
            ax.set_xticks(base_x, [DIFFICULTY_DISPLAY[item] for item in DIFFICULTY_ORDER])
            ax.set_xlim(0.55, 4.45)
            ax.set_ylabel(f"ΔRMSE ({target_error_unit(target)})")
            if row == 1:
                ax.set_xlabel("Baseline RMSE quartile")

    handles = [
        Patch(
            facecolor=source_colors[source],
            edgecolor=source_colors[source],
            alpha=0.32,
            label=HETEROGENEITY_SOURCE_DISPLAY[source],
        )
        for source in HETEROGENEITY_SOURCES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
    )
    method_label = FUSION_DISPLAY.get(fusion_method, fusion_method.title())
    if preview_for is None:
        stem = f"heterogeneous_benefit_{fusion_method}"
        figure_top = 0.90
    else:
        requested_label = FUSION_DISPLAY.get(preview_for, preview_for.title())
        title = (
            f"PREVIEW — {method_label} layout substitute "
            f"({requested_label} matrix incomplete)"
        )
        stem = f"heterogeneous_benefit_{fusion_method}_preview_for_{preview_for}"
        fig.suptitle(title, y=0.995, fontsize=14, fontweight="bold")
        figure_top = 0.84
    fig.text(
        0.5,
        0.012,
        "Negative ΔRMSE indicates improvement. Q1–Q4 are defined once from "
        "the held-out-subject baseline RMSE for each target.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=figure_top,
        hspace=0.28,
        wspace=0.20,
    )
    output_path = output_dir / f"{stem}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def plot_box_metric(
    frame: pd.DataFrame,
    metric: str,
    output_dir: Path,
    stem: str,
    ylabel: str,
    dpi: int,
    save_pdf: bool,
    zero_line: bool = False,
) -> None:
    targets = target_order(frame)
    fig, axes = create_axes_for_targets(targets)
    colors = color_map(experiment_order(frame))

    for ax, target in zip(axes, targets):
        subset = frame[frame["target"] == target]
        labels = experiment_order(subset, target)
        values = [subset.loc[subset["experiment"] == label, metric].dropna().to_numpy() for label in labels]
        positions = np.arange(1, len(labels) + 1)

        bp = ax.boxplot(
            values,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            boxprops={"linewidth": 1.0},
        )
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.48)

        for pos, label, vals in zip(positions, labels, values):
            x = pos + deterministic_jitter(len(vals))
            ax.scatter(x, vals, s=11, alpha=0.28, color=colors[label], edgecolors="none")
            if len(vals):
                ax.scatter(
                    [pos],
                    [np.median(vals)],
                    marker="D",
                    s=34,
                    color=colors[label],
                    edgecolors="black",
                    linewidths=0.45,
                    zorder=4,
                )

        if zero_line:
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(TARGET_DISPLAY[target])
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions, labels, rotation=18, ha="right")
        ax.set_xlim(0.45, len(labels) + 0.55)

    fig.suptitle(f"LOSO test-fold {metric} distribution", y=1.02, fontsize=14)
    save_figure(fig, output_dir, stem, dpi, save_pdf)


def plot_delta_rmse(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    if detail.empty:
        return

    targets = target_order(detail)
    fig, axes = create_axes_for_targets(targets)
    colors = color_map(experiment_order(detail))

    for ax, target in zip(axes, targets):
        subset = detail[detail["target"] == target]
        labels = experiment_order(subset, target)
        values = [
            subset.loc[subset["experiment"] == label, "delta_RMSE"].dropna().to_numpy()
            for label in labels
        ]
        positions = np.arange(1, len(labels) + 1)

        bp = ax.boxplot(
            values,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
        )
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.52)

        for pos, label, vals in zip(positions, labels, values):
            ax.scatter(
                pos + deterministic_jitter(len(vals)),
                vals,
                s=11,
                alpha=0.30,
                color=colors[label],
                edgecolors="none",
            )
            row = summary[
                (summary["target"] == target) & (summary["experiment"] == label)
            ]
            if not row.empty:
                win_rate = float(row.iloc[0]["RMSE_win_rate"])
                median_delta = float(row.iloc[0]["delta_RMSE_median"])
                ax.text(
                    pos,
                    0.98,
                    f"Improved: {win_rate:.0%}\nmed {median_delta:+.3g}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.68, "edgecolor": "none", "pad": 1.5},
                )

        ax.axhline(0.0, color="black", linewidth=1.1, linestyle="--")
        ax.set_title(TARGET_DISPLAY[target])
        ax.set_ylabel(f"ΔRMSE ({target_error_unit(target)})")
        ax.set_xticks(positions, labels, rotation=18, ha="right")
        ax.set_xlim(0.45, len(labels) + 0.55)

    fig.suptitle("Paired subject-level RMSE change from baseline", y=1.02, fontsize=14)
    save_figure(fig, output_dir, "02_delta_rmse_vs_baseline", dpi, save_pdf)


def plot_rmse_vs_absbias(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    targets = target_order(frame)
    fig, axes = create_axes_for_targets(targets)
    labels_all = experiment_order(frame)
    colors = color_map(labels_all)
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    contexts = frame.loc[
        frame["training_mode"] != "baseline",
        ["fusion_method", "training_strategy"],
    ].drop_duplicates()
    if len(contexts) == 1:
        context = contexts.iloc[0]
        fig.suptitle(
            f"{FUSION_DISPLAY.get(context['fusion_method'], context['fusion_method'])} fusion · "
            f"{STRATEGY_DISPLAY.get(context['training_strategy'], context['training_strategy'])}",
            fontsize=12,
            fontweight="bold",
            y=0.995,
        )
    marker_map = {label: markers[i % len(markers)] for i, label in enumerate(labels_all)}

    for ax, target in zip(axes, targets):
        subset = frame[frame["target"] == target]
        for label in experiment_order(subset, target):
            group = subset[subset["experiment"] == label]
            ax.scatter(
                group["AbsBias"],
                group["RMSE"],
                label=label,
                s=24,
                alpha=0.44,
                color=colors[label],
                marker=marker_map[label],
                edgecolors="none",
            )
            ax.scatter(
                [group["AbsBias"].median()],
                [group["RMSE"].median()],
                s=86,
                color=colors[label],
                marker=marker_map[label],
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )

        upper = max(float(ax.get_xlim()[1]), float(ax.get_ylim()[1]))
        ax.plot([0.0, upper], [0.0, upper], linestyle=":", linewidth=1.0, color="black", alpha=0.45)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(TARGET_DISPLAY[target])
        ax.set_xlabel("Absolute bias")
        ax.set_ylabel("RMSE")
        ax.legend(loc="best")

    fig.suptitle("RMSE–bias trade-off across held-out subjects", y=1.02, fontsize=14)
    save_figure(fig, output_dir, "04_rmse_vs_absbias", dpi, save_pdf)


def relative_metric_table(frame: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    rows = []
    metrics = ("RMSE", "MAE", "AbsBias", "ErrorStd")
    for target in target_order(frame):
        target_frame = frame[frame["target"] == target]
        active_baseline_label = resolve_baseline_label(target_frame, baseline_label)
        if active_baseline_label is None:
            continue
        baseline = target_frame[
            target_frame["experiment"] == active_baseline_label
        ].set_index("subject_id")
        for experiment in experiment_order(target_frame, target):
            if experiment == active_baseline_label:
                continue
            current = target_frame[target_frame["experiment"] == experiment].set_index("subject_id")
            common = baseline.index.intersection(current.index)
            if len(common) == 0:
                continue
            for metric in metrics:
                base_values = baseline.loc[common, metric].to_numpy(dtype=float)
                exp_values = current.loc[common, metric].to_numpy(dtype=float)
                valid = np.isfinite(base_values) & np.isfinite(exp_values)
                if not np.any(valid):
                    value = float("nan")
                else:
                    baseline_median = float(np.median(base_values[valid]))
                    median_delta = float(np.median(exp_values[valid] - base_values[valid]))
                    value = (
                        float(median_delta / baseline_median * 100.0)
                        if not np.isclose(baseline_median, 0.0)
                        else float("nan")
                    )
                rows.append(
                    {
                        "target": target,
                        "experiment": experiment,
                        "metric": metric,
                        "median_relative_change_percent": value,
                    }
                )
    return pd.DataFrame(rows)


def plot_relative_metric_change(
    relative: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    if relative.empty:
        return

    row_keys = [
        (target, experiment)
        for target in target_order(relative)
        for experiment in experiment_order(relative[relative["target"] == target], target)
    ]
    metrics = ["RMSE", "MAE", "AbsBias", "ErrorStd"]
    matrix = np.full((len(row_keys), len(metrics)), np.nan, dtype=float)

    for i, (target, experiment) in enumerate(row_keys):
        for j, metric in enumerate(metrics):
            values = relative[
                (relative["target"] == target)
                & (relative["experiment"] == experiment)
                & (relative["metric"] == metric)
            ]["median_relative_change_percent"]
            if not values.empty:
                matrix[i, j] = float(values.iloc[0])

    finite = matrix[np.isfinite(matrix)]
    limit = max(5.0, float(np.percentile(np.abs(finite), 95))) if finite.size else 10.0

    fig_height = max(3.4, 0.62 * len(row_keys) + 1.6)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(np.arange(len(metrics)), metrics)
    ax.set_yticks(
        np.arange(len(row_keys)),
        [f"{TARGET_DISPLAY[target]} — {experiment}" for target, experiment in row_keys],
    )
    ax.set_title("Median paired change relative to baseline")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "—" if not np.isfinite(value) else f"{value:+.1f}%"
            text_color = "white" if np.isfinite(value) and abs(value) > limit * 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=text_color)

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Median paired change (%)\nnegative = improvement")
    ax.grid(False)
    save_figure(fig, output_dir, "05_relative_metric_change", dpi, save_pdf)



def _format_pvalue(value: float) -> str:
    if not np.isfinite(value):
        return "p = NA"
    if value < 0.001:
        return f"p = {value:.1e}"
    return f"p = {value:.3f}"


def _median_relative_change(
    summary: pd.DataFrame,
    target: str,
    experiment: str,
    baseline: str,
    metric: str,
) -> float:
    base = summary[
        (summary["target"] == target) & (summary["experiment"] == baseline)
    ]
    current = summary[
        (summary["target"] == target) & (summary["experiment"] == experiment)
    ]
    if base.empty or current.empty:
        return float("nan")
    base_value = float(base.iloc[0][f"{metric}_median"])
    current_value = float(current.iloc[0][f"{metric}_median"])
    if not np.isfinite(base_value) or np.isclose(base_value, 0.0):
        return float("nan")
    return (current_value - base_value) / base_value * 100.0


def remove_legacy_figure_files(output_dir: Path) -> None:
    """Remove figures made by older five-figure versions of this script."""
    stems = (
        "01_rmse_boxplot",
        "02_delta_rmse_vs_baseline",
        "03_bias_boxplot",
        "04_rmse_vs_absbias",
        "05_relative_metric_change",
    )
    for stem in stems:
        for suffix in (".png", ".pdf"):
            path = output_dir / f"{stem}{suffix}"
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"WARNING: could not remove old figure {path}: {exc}", file=sys.stderr)


def plot_main_loso_comparison(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    paired_detail: pd.DataFrame,
    paired_summary: pd.DataFrame,
    conditional_summary: pd.DataFrame,
    baseline_label: str,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    """Create one two-row figure for a matched representative comparison.

    Row 1: absolute held-out-subject RMSE for the DRS-only baseline,
    tissue-thickness feature, DTOF descriptor, and any additional experiments.
    Compact text reports each experiment's paired Wilcoxon p-value versus the
    baseline; the separate
    paired-difference boxplot is omitted.

    Row 2: conditional paired RMSE differences across common baseline-RMSE
    quartiles. All experiments use the same Q1--Q4 subject assignment within a
        target, so the tissue-thickness feature and DTOF descriptor can be
        compared directly.
    """
    del summary, paired_detail

    targets = target_order(frame)
    if not targets:
        raise ValueError("No HC or StO2 results are available for plotting.")

    n_targets = len(targets)
    fig, axes = plt.subplots(
        2,
        n_targets,
        figsize=(6.8 * n_targets, 9.2),
        squeeze=False,
        gridspec_kw={
            "height_ratios": [1.02, 1.18],
            "hspace": 0.58,
            "wspace": 0.25,
        },
    )
    labels_all = experiment_order(frame)
    colors = color_map(labels_all)
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    for col, target in enumerate(targets):
        target_frame = frame[frame["target"] == target]
        baseline = resolve_baseline_label(target_frame, baseline_label)
        labels = experiment_order(target_frame, target)
        positions = np.arange(1, len(labels) + 1)

        # ---- Row 1: absolute RMSE distribution ----
        ax = axes[0, col]
        rmse_values = [
            target_frame.loc[target_frame["experiment"] == label, "RMSE"]
            .dropna()
            .to_numpy(dtype=float)
            for label in labels
        ]
        bp = ax.boxplot(
            rmse_values,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            boxprops={"linewidth": 1.0},
        )
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.48)

        for pos, label, values in zip(positions, labels, rmse_values):
            ax.scatter(
                pos + deterministic_jitter(len(values), width=0.09),
                values,
                s=10,
                alpha=0.22,
                color=colors[label],
                edgecolors="none",
                rasterized=True,
            )
            if len(values):
                median_value = float(np.median(values))
                ax.scatter(
                    [pos],
                    [median_value],
                    marker="D",
                    s=38,
                    color=colors[label],
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                )
                ax.annotate(
                    f"{median_value:.3g}",
                    (pos, median_value),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        pvalue_lines: list[str] = []
        if baseline is not None:
            for label in labels:
                if label == baseline:
                    continue
                row = paired_summary[
                    (paired_summary["target"] == target)
                    & (paired_summary["experiment"] == label)
                ]
                if row.empty:
                    continue
                pvalue = float(row.iloc[0]["wilcoxon_RMSE_p_holm"])
                pvalue_lines.append(
                    f"{label.split(' | ')[0]}: {_format_pvalue(pvalue)}"
                )

        if pvalue_lines:
            ax.text(
                0.02,
                0.98,
                "Paired Wilcoxon vs baseline (Holm)\n" + "\n".join(pvalue_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.78",
                    "alpha": 0.90,
                    "boxstyle": "round,pad=0.30",
                },
            )

        ax.set_title(f"{TARGET_DISPLAY[target]} prediction performance", fontweight="bold")
        ax.set_ylabel(f"RMSE ({target_error_unit(target)})")
        compact_labels = []
        for label in labels:
            if label == baseline:
                compact_labels.append(BASELINE_DISPLAY)
                continue
            compact_labels.append(label.split(" | ")[0])
        ax.set_xticks(positions, compact_labels, rotation=0, ha="center", fontsize=8.2)
        ax.set_xlim(0.45, len(labels) + 0.55)

        # ---- Row 2: conditional effect by baseline difficulty ----
        ax = axes[1, col]
        target_conditional = conditional_summary[
            conditional_summary["target"] == target
        ]
        conditional_labels = [
            label
            for label in labels
            if label != baseline
            and label in set(target_conditional["experiment"].astype(str))
        ]
        if not conditional_labels:
            ax.text(
                0.5,
                0.5,
                "No conditional summary available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            continue

        base_x = np.arange(1, 5, dtype=float)
        offsets = (
            np.linspace(-0.08, 0.08, len(conditional_labels))
            if len(conditional_labels) > 1
            else np.zeros(1)
        )
        interaction_lines: list[str] = []

        for index, (label, offset) in enumerate(zip(conditional_labels, offsets)):
            current = (
                target_conditional[target_conditional["experiment"] == label]
                .set_index("difficulty_quartile")
                .reindex(DIFFICULTY_ORDER)
            )
            means = current["delta_RMSE_mean"].to_numpy(dtype=float)
            lows = current["delta_RMSE_ci_low"].to_numpy(dtype=float)
            highs = current["delta_RMSE_ci_high"].to_numpy(dtype=float)
            win_rates = current["RMSE_win_rate"].to_numpy(dtype=float)
            x = base_x + offset
            yerr = np.vstack([means - lows, highs - means])

            ax.errorbar(
                x,
                means,
                yerr=yerr,
                color=colors[label],
                marker=markers[index % len(markers)],
                markersize=6.4,
                linewidth=1.7,
                capsize=3.2,
                label=label.split(" | ")[0],
                zorder=4,
            )

            # Report only the hardest-quartile win rate; Q1 annotations add
            # clutter without changing the heterogeneous-benefit conclusion.
            for quartile_index in (3,):
                mean_value = means[quartile_index]
                win_rate = win_rates[quartile_index]
                if not np.isfinite(mean_value) or not np.isfinite(win_rate):
                    continue
                vertical_offset = 9 if mean_value <= 0 else -11
                horizontal_offset = (
                    (-18 if index == 0 else 18)
                    if len(conditional_labels) == 2
                    else 0
                )
                ax.annotate(
                    f"Improved: {win_rate:.0%}",
                    (x[quartile_index], mean_value),
                    xytext=(horizontal_offset, vertical_offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if mean_value <= 0 else "top",
                    fontsize=7.2,
                    color=colors[label],
                )

            first = current.iloc[0]
            interaction_lines.append(
                f"{label}: Q4−Q1 {float(first['hard_minus_easy_mean_delta']):+.2f} "
                f"[{float(first['hard_minus_easy_ci_low']):+.2f}, "
                f"{float(first['hard_minus_easy_ci_high']):+.2f}]"
            )

        ax.axhline(0.0, color="black", linewidth=1.1, linestyle="--")
        ax.set_title(
            f"{TARGET_DISPLAY[target]}: feature effect by baseline difficulty",
            fontweight="bold",
        )
        ax.set_xlabel("Baseline RMSE quartile (Q1 easiest → Q4 hardest)")
        ax.set_ylabel(f"ΔRMSE ({target_error_unit(target)})")
        ax.set_xticks(
            base_x,
            [DIFFICULTY_DISPLAY[item] for item in DIFFICULTY_ORDER],
        )
        ax.set_xlim(0.55, 4.45)
        ax.legend(loc="best", ncol=min(2, len(conditional_labels)))
        # ax.text(
        #     0.02,
        #     0.98,
        #     "Hard−easy interaction\n" + "\n".join(interaction_lines),
        #     transform=ax.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=7.6,
        #     bbox={
        #         "facecolor": "white",
        #         "edgecolor": "0.78",
        #         "alpha": 0.90,
        #         "boxstyle": "round,pad=0.30",
        #     },
        # )

    # fig.text(
    #     0.5,
    #     0.006,
    #     "Top: absolute RMSE and paired Wilcoxon p-values versus baseline. "
    #     "Bottom: Metadata/Latent effects within common baseline-RMSE quartiles. "
    #     "Q1–Q4 are descriptive test-defined groups, not a deployable selector.",
    #     ha="center",
    #     va="bottom",
    #     fontsize=8.5,
    # )
    comparison_rows = frame[frame["feature_source"].isin(HETEROGENEITY_SOURCES)]
    fusion_methods = comparison_rows["fusion_method"].dropna().unique()
    training_strategies = comparison_rows["training_strategy"].dropna().unique()
    if len(fusion_methods) == 1 and len(training_strategies) == 1:
        fig.suptitle(
            f"{FUSION_DISPLAY.get(fusion_methods[0], fusion_methods[0])} fusion · "
            f"{STRATEGY_DISPLAY.get(training_strategies[0], training_strategies[0])}",
            fontsize=12,
            fontweight="bold",
            y=0.995,
        )
    fig.subplots_adjust(top=0.92, bottom=0.09)
    fig.savefig(
        output_dir / "representative_loso_comparison.png",
        dpi=dpi,
        bbox_inches="tight",
    )
def print_console_summary(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    conditional: pd.DataFrame,
) -> None:
    print("\nExperiment summary (median RMSE / median |Bias|):")
    for _, row in summary.sort_values(
        ["target", "experiment"],
        key=lambda col: col.map(label_sort_key) if col.name == "experiment" else col,
    ).iterrows():
        print(
            f"  {TARGET_DISPLAY.get(row['target'], row['target']) :>4} | "
            f"{row['experiment']:<24} | n={int(row['n_subjects']):3d} | "
            f"RMSE={row['RMSE_median']:.5g} | |Bias|={row['AbsBias_median']:.5g}"
        )

    if not paired.empty:
        print("\nPaired comparison vs baseline:")
        for _, row in paired.iterrows():
            pvalue = row["wilcoxon_RMSE_p"]
            ptext = "NA" if not np.isfinite(pvalue) else f"{pvalue:.3g}"
            print(
                f"  {TARGET_DISPLAY.get(row['target'], row['target']) :>4} | "
                f"{row['experiment']:<24} | "
                f"median ΔRMSE={row['delta_RMSE_median']:+.5g} | "
                f"Improved={row['RMSE_win_rate']:.1%} | Wilcoxon p={ptext}"
            )

    if not conditional.empty:
        print("\nConditional effect by baseline difficulty:")
        first_rows = conditional[conditional["difficulty_quartile"] == "Q1"]
        for _, row in first_rows.iterrows():
            print(
                f"  {TARGET_DISPLAY.get(row['target'], row['target']) :>4} | "
                f"{row['experiment']:<24} | "
                f"Q4-Q1 mean ΔRMSE={row['hard_minus_easy_mean_delta']:+.5g} "
                f"[{row['hard_minus_easy_ci_low']:+.5g}, "
                f"{row['hard_minus_easy_ci_high']:+.5g}]"
            )


def main() -> int:
    args = parse_args()
    if args.expected_subjects <= 0:
        raise ValueError("--expected-subjects must be positive.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = discover_experiments(args)
    frames = []
    metas = []
    for spec in specs:
        try:
            meta, tidy = load_experiment(spec)
        except Exception as exc:  # keep other completed experiments usable
            print(f"WARNING: skipped {spec.directory}: {exc}", file=sys.stderr)
            continue
        subject_count = int(tidy["subject_id"].nunique())
        if subject_count != args.expected_subjects and not args.include_incomplete:
            print(
                f"WARNING: skipped incomplete run {spec.directory}: "
                f"{subject_count}/{args.expected_subjects} subjects. "
                "Use --include-incomplete only for diagnostics.",
                file=sys.stderr,
            )
            continue
        metas.append(meta)
        frames.append(tidy)
        print(
            f"Loaded {meta.label:<22} target={meta.target_mode:<4} "
            f"folds={len(tidy):3d} from {meta.directory}"
        )

    if not frames:
        raise RuntimeError("All discovered experiments failed to load.")

    all_results = pd.concat(frames, ignore_index=True)

    duplicate_labels = (
        all_results[["target", "experiment", "experiment_dir"]]
        .drop_duplicates()
        .groupby(["target", "experiment"])["experiment_dir"]
        .nunique()
    )
    conflicts = duplicate_labels[duplicate_labels > 1]
    if not conflicts.empty:
        raise ValueError(
            "Multiple directories map to the same target/experiment label. "
            "Use --experiment PATH=LABEL to make labels unique. Conflicts:\n"
            + conflicts.to_string()
        )

    all_results.to_csv(output_dir / "all_fold_results.csv", index=False)

    summary = summarize_experiments(all_results)
    summary.to_csv(output_dir / "experiment_summary.csv", index=False)

    paired_detail, paired_summary = paired_comparisons(
        all_results,
        baseline_label=args.baseline_label,
    )
    if not paired_detail.empty:
        paired_detail.to_csv(output_dir / "paired_fold_differences.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_vs_baseline.csv", index=False)

    final_table = build_final_experiment_table(all_results, paired_summary)
    final_table.to_csv(
        output_dir / "final_experiment_table.csv", index=False, encoding="utf-8-sig"
    )
    format_final_experiment_table(final_table).to_csv(
        output_dir / "final_experiment_table_formatted.csv",
        index=False,
        encoding="utf-8-sig",
    )

    relative = relative_metric_table(all_results, args.baseline_label)
    if not relative.empty:
        relative.to_csv(output_dir / "relative_metric_change.csv", index=False)

    conditional = conditional_effect_by_baseline_difficulty(paired_detail)
    if not conditional.empty:
        conditional.to_csv(
            output_dir / "conditional_effect_by_baseline_difficulty.csv",
            index=False,
        )

    apply_plot_style()
    save_pdf = False
    remove_legacy_figure_files(output_dir)
    representative = all_results[
        (all_results["training_mode"] == "baseline")
        | (
            (all_results["fusion_method"] == args.representative_method)
            & (all_results["training_strategy"] == args.representative_strategy)
            & (all_results["structure_control"] == "actual")
        )
    ].copy()
    nonbaseline_sources = set(
        representative.loc[
            representative["training_mode"] != "baseline", "feature_source"
        ]
    )
    missing_sources = {"latent", "metadata"} - nonbaseline_sources
    if missing_sources:
        print(
            "WARNING: representative figure currently lacks completed source(s): "
            + ", ".join(sorted(missing_sources))
            + ". Re-run after those experiments finish.",
            file=sys.stderr,
        )
    plot_main_loso_comparison(
        frame=representative,
        summary=summary,
        paired_detail=paired_detail,
        paired_summary=paired_summary,
        conditional_summary=conditional,
        baseline_label=args.baseline_label,
        output_dir=output_dir,
        dpi=args.dpi,
        save_pdf=save_pdf,
    )

    requested_method = args.heterogeneity_method
    selected_method, is_preview, missing_cells = resolve_heterogeneous_benefit_method(
        all_results,
        requested_method=requested_method,
        fallback_method=args.heterogeneity_fallback_method,
    )
    heterogeneity_manifest: dict[str, object] = {
        "requested_method": requested_method,
        "fallback_method": args.heterogeneity_fallback_method,
        "selected_method": selected_method,
        "is_preview": is_preview,
        "requested_method_missing_cells": missing_cells,
        "generated": False,
    }
    if selected_method is None:
        print(
            "WARNING: heterogeneous-benefit figure skipped; incomplete "
            f"{requested_method} matrix: " + ", ".join(missing_cells),
            file=sys.stderr,
        )
    else:
        if is_preview:
            print(
                "WARNING: heterogeneous-benefit figure uses the complete "
                f"{selected_method} matrix as an explicitly labelled preview; "
                f"the requested {requested_method} matrix is incomplete.",
                file=sys.stderr,
            )
        heterogeneity_detail = build_heterogeneous_benefit_detail(
            all_results,
            paired_detail,
            fusion_method=selected_method,
        )
        heterogeneity_summary = summarize_heterogeneous_benefit(
            heterogeneity_detail
        )
        preview_suffix = (
            f"_preview_for_{requested_method}" if is_preview else ""
        )
        summary_path = output_dir / (
            f"heterogeneous_benefit_{selected_method}{preview_suffix}.csv"
        )
        heterogeneity_summary.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        image_path = plot_heterogeneous_benefit(
            heterogeneity_detail,
            output_dir=output_dir,
            fusion_method=selected_method,
            dpi=args.dpi,
            preview_for=requested_method if is_preview else None,
        )
        heterogeneity_manifest.update(
            {
                "generated": True,
                "image": image_path.name,
                "summary": summary_path.name,
            }
        )

    with (output_dir / "heterogeneous_benefit_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(heterogeneity_manifest, handle, indent=2, ensure_ascii=False)

    print_console_summary(summary, paired_summary, conditional)
    print(f"\nSaved analysis to: {output_dir}")

    if args.show:
        plt.show()
    else:
        plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
