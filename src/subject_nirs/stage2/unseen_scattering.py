"""Evaluate frozen DRS-only HC or StO2 LOSO checkpoints on a validation cache.

This script never trains or selects a checkpoint.  For each fold it evaluates
only that fold's LOSO test subject, so the reported rows are held out from both
training and checkpoint selection.  Fold-level outputs make the run resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch

from .cache import (
    preprocess_drs_x,
    read_scalar,
    read_subject_rows,
    read_x_rows,
    read_y_rows,
)
from .config import CFG
from .model import CNN1D


EPS = 1e-8
CHECKPOINT_RE = re.compile(
    r"best_model_fold_(?P<fold>\d+)_test_id_(?P<test_id>\d+)\.pth$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with frozen HC or StO2 baseline checkpoints on the "
            "new mu_s validation cache."
        )
    )
    parser.add_argument(
        "--target",
        choices=("hc", "sto2"),
        default="hc",
        help="Prediction target. Selects the checkpoint target and cache column.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Defaults to artifacts/stage2/<target>_baseline_scratch_main.",
    )
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument(
        "--validation_mat",
        type=Path,
        default=None,
        help="Read and preprocess the validation MAT directly.",
    )
    validation_group.add_argument(
        "--validation_cache",
        type=Path,
        default=None,
        help="Optional existing preprocessed cache instead of --validation_mat.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to artifacts/stage2/unseen_scattering/<target>.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="One-based folds, e.g. '1,3,10-20'. Default: all checkpoints.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit CUDA device such as cuda:1",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument(
        "--rows_per_sim",
        type=int,
        default=12960,
        help="Used only to produce per-simulation metrics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Save metrics only. Predictions are saved by default.",
    )
    return parser.parse_args()


def parse_fold_spec(spec: str | None) -> set[int] | None:
    if spec is None:
        return None
    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid fold range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    if not selected or min(selected) < 1:
        raise ValueError("--folds must contain positive one-based fold numbers")
    return selected


def discover_checkpoints(
    checkpoint_dir: Path, selected_folds: set[int] | None
) -> list[tuple[int, int, Path]]:
    found: list[tuple[int, int, Path]] = []
    for path in checkpoint_dir.glob("best_model_fold_*_test_id_*.pth"):
        match = CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        fold = int(match.group("fold"))
        test_id = int(match.group("test_id"))
        if selected_folds is None or fold in selected_folds:
            found.append((fold, test_id, path))
    found.sort()
    if not found:
        raise FileNotFoundError(f"No matching checkpoints in {checkpoint_dir}")
    if selected_folds is not None:
        missing = sorted(selected_folds - {fold for fold, _, _ in found})
        if missing:
            raise FileNotFoundError(f"Missing requested folds: {missing}")
    return found


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            "--device must be auto, cpu, cuda, or cuda:<index>"
        ) from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device must select CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {name} requested, but CUDA is unavailable")
        index = 0 if device.index is None else device.index
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise ValueError(
                f"CUDA device index {index} is unavailable; found {count} device(s)"
            )
    return device


def load_cache(cache_dir: Path):
    paths = {
        "x": cache_dir / "val_x_preprocessed.npy",
        "y": cache_dir / "val_y_all_targets.npy",
        "subj": cache_dir / "val_subj_0based.npy",
        "metadata": cache_dir / "cache_metadata_all_targets_logod1.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing validation cache files: {missing}")
    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    x = np.load(paths["x"], mmap_mode="r")
    y = np.load(paths["y"], mmap_mode="r")
    subj = np.load(paths["subj"], mmap_mode="r")
    if not (len(x) == len(y) == len(subj)):
        raise ValueError("Validation cache arrays have inconsistent row counts")
    if x.ndim != 2 or x.shape[1] != 30 or y.ndim != 2:
        raise ValueError(f"Unexpected cache shapes: x={x.shape}, y={y.shape}")
    return x, y, subj, metadata, "cache", None


def load_mat(mat_path: Path):
    """Open a MATLAB v7.3/HDF5 validation file without materializing DRS rows."""
    if not mat_path.is_file():
        raise FileNotFoundError(f"Validation MAT does not exist: {mat_path}")
    handle = h5py.File(mat_path, "r")
    required = {
        "x": "Val_Dataset_X",
        "y": "Val_Dataset_Y",
        "subj": "Val_Dataset_SubjID",
    }
    missing = [name for name in required.values() if name not in handle]
    if missing:
        handle.close()
        raise KeyError(f"Validation MAT is missing datasets: {missing}")

    x = handle[required["x"]]
    y = handle[required["y"]]
    subject_dataset = handle[required["subj"]]
    if "num_val_rows" in handle:
        n_rows = read_scalar(handle, "num_val_rows")
    else:
        n_rows = max(subject_dataset.shape)
    subj = read_subject_rows(subject_dataset, 0, n_rows)
    metadata = {
        "mat_file": str(mat_path.resolve()),
        "cache_target_names": list(CFG.cache_target_names),
        "preprocessing": "per_sample_mean_then_negative_log10 (streamed)",
    }
    return x, y, subj, metadata, "mat", handle


def read_validation_batch(x_source, y_source, rows, source_kind: str):
    if source_kind == "mat":
        x = preprocess_drs_x(read_x_rows(x_source, rows))
        y = read_y_rows(y_source, rows)
        return x, y
    return (
        np.asarray(x_source[rows], dtype=np.float32),
        np.asarray(y_source[rows], dtype=np.float32),
    )


def scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("Invalid metric inputs")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Prediction or target contains NaN/Inf")
    error = pred - true
    absolute = np.abs(error)
    bias = float(error.mean())
    return {
        "MAE": float(absolute.mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "Bias": bias,
        "AbsBias": abs(bias),
        "ErrorStd": float(np.std(error)),
        "P50_AE": float(np.percentile(absolute, 50)),
        "P75_AE": float(np.percentile(absolute, 75)),
        "P90_AE": float(np.percentile(absolute, 90)),
        "P95_AE": float(np.percentile(absolute, 95)),
        "MaxAE": float(absolute.max()),
    }


def configure_model_from_checkpoint(config: dict) -> None:
    CFG.target_mode = str(config.get("target_mode", "hc"))
    CFG.num_wl = int(config.get("num_wl", 5))
    CFG.num_sds = int(config.get("num_sds", 6))
    CFG.dropout = float(config.get("dropout", 0.1))
    CFG.group_norm_groups = int(config.get("group_norm_groups", 8))


def checkpoint_array(config: dict, name: str, length: int) -> np.ndarray:
    if name not in config:
        raise KeyError(f"Checkpoint config is missing {name!r}")
    value = np.asarray(config[name], dtype=np.float32).reshape(-1)
    if value.size not in {1, length}:
        raise ValueError(f"Unexpected {name} shape: {value.shape}")
    return value


@torch.inference_mode()
def infer_fold(
    checkpoint_path: Path,
    expected_test_id: int,
    expected_target: str,
    x_cache,
    y_cache,
    subject_cache,
    cache_target_names: list[str],
    source_kind: str,
    device: torch.device,
    batch_size: int,
    rows_per_sim: int,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    test_id = int(config.get("test_id", expected_test_id))
    if test_id != expected_test_id:
        raise ValueError(
            f"Checkpoint filename test_id={expected_test_id}, config test_id={test_id}"
        )
    configure_model_from_checkpoint(config)
    if CFG.target_mode != expected_target:
        raise ValueError(
            f"Expected {expected_target} checkpoint, "
            f"got target_mode={CFG.target_mode!r}"
        )
    target_column = {"hc": "GM_hc", "sto2": "GM_StO2"}[expected_target]
    try:
        target_index = cache_target_names.index(target_column)
    except ValueError as error:
        raise ValueError(
            f"{target_column} missing from cache targets: {cache_target_names}"
        ) from error

    x_min = checkpoint_array(config, "x_min", CFG.input_dim)
    x_max = checkpoint_array(config, "x_max", CFG.input_dim)
    y_mean = checkpoint_array(config, "y_mean", 1)
    y_std = checkpoint_array(config, "y_std", 1)

    model = CNN1D(output_dim=1, latent_dim=0)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.to(device).eval()

    row_index = np.flatnonzero(np.asarray(subject_cache) == test_id).astype(np.int64)
    if row_index.size == 0:
        raise ValueError(f"No rows for test_id={test_id}")
    if rows_per_sim <= 0 or row_index.size % rows_per_sim != 0:
        raise ValueError(
            f"Subject {test_id} has {row_index.size} rows, not divisible by "
            f"rows_per_sim={rows_per_sim}"
        )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for start in range(0, row_index.size, batch_size):
        rows = row_index[start : start + batch_size]
        x, y_all = read_validation_batch(x_cache, y_cache, rows, source_kind)
        y = np.asarray(y_all[:, target_index], dtype=np.float32).reshape(-1, 1)
        x = (x - x_min) / (x_max - x_min + EPS)
        x_tensor = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        prediction_norm = model(x_tensor)
        prediction = prediction_norm.cpu().numpy() * (y_std + EPS) + y_mean
        predictions.append(prediction.astype(np.float32))
        targets.append(y.astype(np.float32))

    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(predictions, axis=0)
    if y_true.shape != y_pred.shape or not np.isfinite(y_pred).all():
        raise RuntimeError(
            f"Invalid inference output: true={y_true.shape}, pred={y_pred.shape}"
        )
    num_sims = row_index.size // rows_per_sim
    sim_id = np.repeat(np.arange(num_sims, dtype=np.int16), rows_per_sim)
    return config, row_index, sim_id, y_true, y_pred


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def save_prediction_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize_fold_metrics(rows: list[dict]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    metric_names = (
        "MAE",
        "RMSE",
        "Bias",
        "AbsBias",
        "ErrorStd",
        "P50_AE",
        "P75_AE",
        "P90_AE",
        "P95_AE",
        "MaxAE",
    )
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        summary[name] = {
            "fold_mean": float(values.mean()),
            "fold_sd": float(values.std()),
            "fold_median": float(np.median(values)),
            "fold_q05": float(np.quantile(values, 0.05)),
            "fold_q95": float(np.quantile(values, 0.95)),
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(
            f"./artifacts/stage2/{args.target}_baseline_scratch_main"
        )
    if args.output_dir is None:
        args.output_dir = Path(f"./artifacts/stage2/unseen_scattering/{args.target}")
    if args.validation_mat is None and args.validation_cache is None:
        args.validation_mat = Path(
            "./data/raw/stage2/NIRS_Absolute_Val_Dataset_new.mat"
        )
    selected_folds = parse_fold_spec(args.folds)
    checkpoints = discover_checkpoints(args.checkpoint_dir, selected_folds)
    device = resolve_device(args.device)
    if args.validation_cache is not None:
        (
            x_cache,
            y_cache,
            subject_cache,
            cache_metadata,
            source_kind,
            source_handle,
        ) = load_cache(args.validation_cache)
        validation_source = args.validation_cache
    else:
        (
            x_cache,
            y_cache,
            subject_cache,
            cache_metadata,
            source_kind,
            source_handle,
        ) = load_mat(args.validation_mat)
        validation_source = args.validation_mat
    cache_target_names = list(cache_metadata.get("cache_target_names", []))

    prediction_dir = args.output_dir / "predictions"
    metric_dir = args.output_dir / "fold_metrics"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    fold_rows: list[dict] = []
    sim_rows: list[dict] = []
    print(
        f"target={args.target} device={device} folds={len(checkpoints)} "
        f"source={source_kind} rows={len(subject_cache):,} "
        f"output={args.output_dir}"
    )

    for ordinal, (fold, test_id, checkpoint_path) in enumerate(checkpoints, start=1):
        metric_path = metric_dir / f"fold_{fold:03d}_test_id_{test_id}.json"
        prediction_path = (
            prediction_dir / f"fold_{fold:03d}_test_id_{test_id}_predictions.npz"
        )
        can_resume = metric_path.exists() and (
            args.no_save_predictions or prediction_path.exists()
        )
        if can_resume and not args.overwrite:
            with metric_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            saved_target = saved.get(
                "target", saved.get("fold_metrics", {}).get("target")
            )
            if saved_target == args.target:
                fold_rows.append(saved["fold_metrics"])
                sim_rows.extend(saved["sim_metrics"])
                print(
                    f"[{ordinal:03d}/{len(checkpoints):03d}] "
                    f"fold {fold:03d}: cached"
                )
                continue

        fold_started = time.time()
        config, row_index, sim_id, y_true, y_pred = infer_fold(
            checkpoint_path=checkpoint_path,
            expected_test_id=test_id,
            expected_target=args.target,
            x_cache=x_cache,
            y_cache=y_cache,
            subject_cache=subject_cache,
            cache_target_names=cache_target_names,
            source_kind=source_kind,
            device=device,
            batch_size=args.batch_size,
            rows_per_sim=args.rows_per_sim,
        )
        metrics = scalar_metrics(y_true, y_pred)
        fold_row = {
            "target": args.target,
            "fold": fold,
            "test_id": test_id,
            "original_matlab_subject_id": test_id + 1,
            "n_rows": int(y_true.size),
            **metrics,
            "source_checkpoint": str(checkpoint_path.resolve()),
        }
        fold_sim_rows = []
        for current_sim in np.unique(sim_id):
            mask = sim_id == current_sim
            fold_sim_rows.append(
                {
                    "target": args.target,
                    "fold": fold,
                    "test_id": test_id,
                    "original_matlab_subject_id": test_id + 1,
                    "sim_id_0based": int(current_sim),
                    "sim_number_1based": int(current_sim) + 1,
                    "n_rows": int(mask.sum()),
                    **scalar_metrics(y_true[mask], y_pred[mask]),
                }
            )

        if not args.no_save_predictions:
            error = y_pred - y_true
            save_prediction_npz(
                prediction_path,
                fold=np.array(fold, dtype=np.int16),
                test_id=np.array(test_id, dtype=np.int16),
                original_matlab_subject_id=np.array(test_id + 1, dtype=np.int16),
                source_checkpoint=np.array(str(checkpoint_path.resolve())),
                source_validation_data=np.array(str(validation_source.resolve())),
                row_index=row_index,
                sim_id_0based=sim_id,
                y_true=y_true,
                y_pred=y_pred,
                error=error.astype(np.float32),
                abs_error=np.abs(error).astype(np.float32),
            )
        atomic_json(
            metric_path,
            {
                "target": args.target,
                "fold_metrics": fold_row,
                "sim_metrics": fold_sim_rows,
                "checkpoint_config_test_id": int(config.get("test_id", test_id)),
                "source_validation_mat": cache_metadata.get("val_mat_file")
                or cache_metadata.get("mat_file"),
            },
        )
        fold_rows.append(fold_row)
        sim_rows.extend(fold_sim_rows)
        print(
            f"[{ordinal:03d}/{len(checkpoints):03d}] fold {fold:03d}: "
            f"rows={y_true.size:,} MAE={metrics['MAE']:.6f} "
            f"bias={metrics['Bias']:.6f} sec={time.time() - fold_started:.1f}"
        )

    fold_rows.sort(key=lambda row: int(row["fold"]))
    sim_rows.sort(key=lambda row: (int(row["fold"]), int(row["sim_id_0based"])))
    write_csv(args.output_dir / "per_fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "per_sim_metrics.csv", sim_rows)
    summary = {
        "description": (
            "Frozen legacy DRS-only LOSO checkpoints evaluated on each fold's "
            "test-subject rows from the new mu_s validation dataset."
        ),
        "target": args.target,
        "target_column": {"hc": "GM_hc", "sto2": "GM_StO2"}[args.target],
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "validation_source_kind": source_kind,
        "validation_source": str(validation_source.resolve()),
        "source_validation_mat": cache_metadata.get("val_mat_file")
        or cache_metadata.get("mat_file"),
        "device": str(device),
        "batch_size": args.batch_size,
        "folds": [int(row["fold"]) for row in fold_rows],
        "num_folds": len(fold_rows),
        "num_rows": int(sum(int(row["n_rows"]) for row in fold_rows)),
        "rows_per_sim": args.rows_per_sim,
        "predictions_saved": not args.no_save_predictions,
        "fold_metric_summary": summarize_fold_metrics(fold_rows),
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    atomic_json(args.output_dir / "analysis_summary.json", summary)
    if source_handle is not None:
        source_handle.close()
    print(
        f"complete: folds={len(fold_rows)} rows={summary['num_rows']:,} "
        f"fold-mean MAE={summary['fold_metric_summary']['MAE']['fold_mean']:.6f} "
        f"elapsed={summary['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
