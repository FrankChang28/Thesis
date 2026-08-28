#!/usr/bin/env python3
"""Resumable OP robustness analysis using U-centered linear CKA.

The DTOF matrix is always accessed through LazyDTOFArray in acquisition batches.
Each completed fold is reduced to 2700 normalized U-centered Gram vectors, so a
rerun can skip model loading and inference entirely.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch

from .evaluation import (
    build_model,
    infer_embeddings,
    load_checkpoint,
    load_raw_dtof,
    resolve_device,
    subject_position_indices,
)


PARAM_NAMES = (
    "mu_a_scalp",
    "mu_a_skull",
    "mu_a_gm",
    "mu_s_gm",
    "mu_s_skull",
    "mu_s_scalp",
)
PARAM_LEVELS: dict[str, np.ndarray] = {
    "mu_a_scalp": np.asarray([0.12, 0.17, 0.22, 0.27, 0.32]),
    "mu_a_skull": np.asarray([0.10, 0.115, 0.13, 0.145, 0.16]),
    "mu_a_gm": np.asarray([0.12, 0.153333, 0.186667, 0.22]),
    "mu_s_gm": np.asarray([80.0, 160.0, 240.0]),
    "mu_s_skull": np.asarray([100.0, 150.0, 200.0]),
    "mu_s_scalp": np.asarray([130.0, 165.0, 200.0]),
}
PARAM_STRIDES = {
    "mu_a_scalp": 1,
    "mu_a_skull": 5,
    "mu_a_gm": 25,
    "mu_s_gm": 100,
    "mu_s_skull": 300,
    "mu_s_scalp": 900,
}
EXPECTED_FOLDS = 154
EXPECTED_OPS = 2700
EPS = 1e-12


def op_mapping() -> tuple[np.ndarray, np.ndarray]:
    """Return level indices and physical values in the specified mixed radix."""
    indices = np.empty((EXPECTED_OPS, len(PARAM_NAMES)), dtype=np.int16)
    values = np.empty((EXPECTED_OPS, len(PARAM_NAMES)), dtype=np.float64)
    for k in range(EXPECTED_OPS):
        remainder = k
        for column, name in enumerate(PARAM_NAMES):
            radix = len(PARAM_LEVELS[name])
            level = remainder % radix
            remainder //= radix
            indices[k, column] = level
            values[k, column] = PARAM_LEVELS[name][level]
        if remainder:
            raise AssertionError("Mixed-radix OP mapping overflow")
    return indices, values


def adjacent_and_endpoint_levels(name: str) -> list[tuple[int, int, bool]]:
    count = len(PARAM_LEVELS[name])
    return [(i, i + 1, False) for i in range(count - 1)] + [(0, count - 1, True)]


def format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def transition_label(name: str, low: int, high: int) -> str:
    levels = PARAM_LEVELS[name]
    return f"{format_value(levels[low])}→{format_value(levels[high])}"


def matched_transition_pairs(
    name: str, low: int, high: int, level_indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if level_indices is None:
        level_indices, _ = op_mapping()
    column = PARAM_NAMES.index(name)
    left = np.flatnonzero(level_indices[:, column] == low).astype(np.int64)
    right = left + (high - low) * PARAM_STRIDES[name]
    if not np.all(level_indices[right, column] == high):
        raise AssertionError(f"Invalid matched-pair construction for {name}")
    other = [i for i in range(len(PARAM_NAMES)) if i != column]
    if not np.array_equal(level_indices[left][:, other], level_indices[right][:, other]):
        raise AssertionError("A controlled transition changed a nuisance parameter")
    return left, right


def u_centered_cka_vectors(latent: np.ndarray) -> np.ndarray:
    """Convert [OP,subject,latent] rows to normalized U-centered Gram vectors."""
    latent = np.asarray(latent, dtype=np.float64)
    if latent.ndim != 3:
        raise ValueError(f"Expected latent [OP,subject,dim], got {latent.shape}")
    num_subjects = latent.shape[1]
    if num_subjects < 4:
        raise ValueError("U-centering requires at least four subjects")
    gram = latent @ np.swapaxes(latent, 1, 2)
    diagonal = np.arange(num_subjects)
    gram[:, diagonal, diagonal] = 0.0
    row_sum = gram.sum(axis=2, keepdims=True)
    col_sum = gram.sum(axis=1, keepdims=True)
    total = gram.sum(axis=(1, 2), keepdims=True)
    centered = (
        gram
        - row_sum / (num_subjects - 2)
        - col_sum / (num_subjects - 2)
        + total / ((num_subjects - 1) * (num_subjects - 2))
    )
    centered[:, diagonal, diagonal] = 0.0
    flat = centered.reshape(len(centered), -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        bad = np.flatnonzero(norms[:, 0] <= EPS)[:10].tolist()
        raise FloatingPointError(f"Degenerate U-centered Gram vectors at OPs {bad}")
    vectors = (flat / norms).astype(np.float32)
    if not np.isfinite(vectors).all():
        raise FloatingPointError("CKA vectors contain NaN or Inf")
    return vectors


def discover_checkpoints(
    result_root: Path, experiment: str, requested: Sequence[int] | None
) -> list[tuple[int, Path]]:
    requested_set = None if requested is None else set(map(int, requested))
    found: list[tuple[int, Path]] = []
    for path in result_root.glob(f"*/{experiment}/best_model.pt"):
        if re.fullmatch(r"\d+", path.parent.parent.name):
            fold = int(path.parent.parent.name)
            if requested_set is None or fold in requested_set:
                found.append((fold, path.resolve()))
    found.sort()
    if requested_set is not None:
        missing = sorted(requested_set.difference(fold for fold, _ in found))
        if missing:
            raise FileNotFoundError(f"Requested folds have no checkpoint: {missing}")
    return found


def cache_path(output_dir: Path, fold: int) -> Path:
    return output_dir / "cache" / f"fold_{fold}_cka_vectors.npz"


def load_valid_cache(
    path: Path, checkpoint: Path, experiment: str
) -> Mapping[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        payload = np.load(path, allow_pickle=False)
        vectors = payload["cka_vectors"]
        source = str(payload["source_checkpoint"].item())
        cached_experiment = str(payload["experiment"].item())
        if vectors.shape != (EXPECTED_OPS, 121):
            raise ValueError(f"wrong vector shape {vectors.shape}")
        if source != str(checkpoint) or cached_experiment != experiment:
            raise ValueError("checkpoint or experiment metadata changed")
        if not np.isfinite(vectors).all():
            raise ValueError("non-finite vectors")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-5):
            raise ValueError("non-unit CKA vectors")
        return payload
    except Exception as error:
        print(f"Ignoring invalid cache {path}: {error}")
        return None


def infer_fold_cache(
    fold: int,
    checkpoint_path: Path,
    raw_dtof: Any,
    output_dir: Path,
    experiment: str,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    val_subjects = np.asarray(checkpoint.get("val_subjects"), dtype=np.int64).reshape(-1)
    test_subjects = np.asarray(checkpoint.get("test_subjects"), dtype=np.int64).reshape(-1)
    if len(val_subjects) != 10 or len(test_subjects) != 1:
        raise ValueError(
            f"Fold {fold}: expected 10 validation + 1 test subject, got "
            f"{len(val_subjects)} + {len(test_subjects)}"
        )
    if np.intersect1d(val_subjects, test_subjects).size:
        raise ValueError(f"Fold {fold}: validation and test subjects overlap")
    subject_ids = np.sort(np.concatenate([val_subjects, test_subjects]))
    model_config = dict(checkpoint["model_config"])
    latent_dim = int(model_config["latent_dim"])
    if latent_dim <= 0:
        raise ValueError(f"Fold {fold}: latent_dim must be positive, got {latent_dim}")
    if raw_dtof.shape[0] % EXPECTED_FOLDS:
        raise ValueError("DTOF acquisition count is not divisible by 154 subjects")
    op_count = raw_dtof.shape[0] // EXPECTED_FOLDS
    if op_count != EXPECTED_OPS:
        raise ValueError(f"Expected 2700 OPs per subject, got {op_count}")
    if tuple(raw_dtof.shape[1:]) != (
        int(model_config["num_sds"]),
        int(model_config["num_time_bins"]),
    ):
        raise ValueError("DTOF shape does not match checkpoint model_config")

    model = build_model(checkpoint, device)
    positions = np.arange(EXPECTED_OPS, dtype=np.int64)
    global_indices = subject_position_indices(subject_ids, positions, EXPECTED_OPS)
    latent_subject_major = infer_embeddings(
        model,
        raw_dtof,
        global_indices,
        checkpoint,
        device,
        batch_size,
        amp,
    )
    latent = latent_subject_major.reshape(len(subject_ids), EXPECTED_OPS, latent_dim).transpose(1, 0, 2)
    latent_norms = np.linalg.norm(latent, axis=2, keepdims=True)
    if np.any(latent_norms <= EPS):
        raise FloatingPointError(f"Fold {fold}: zero latent row")
    max_norm_error = float(np.max(np.abs(latent_norms - 1.0)))
    latent = latent / latent_norms
    vectors = u_centered_cka_vectors(latent)
    level_indices, values = op_mapping()
    destination = cache_path(output_dir, fold)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        cka_vectors=vectors,
        subject_ids=subject_ids,
        val_subject_ids=np.sort(val_subjects),
        test_subject_ids=np.sort(test_subjects),
        op_values=values,
        op_level_indices=level_indices,
        source_checkpoint=np.asarray(str(checkpoint_path)),
        latent_dim=np.asarray(latent_dim),
        experiment=np.asarray(experiment),
        max_latent_norm_error=np.asarray(max_norm_error),
    )
    temporary.replace(destination)
    del model, latent_subject_major, latent, vectors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "fold": fold,
        "checkpoint": str(checkpoint_path),
        "val_subject_ids": np.sort(val_subjects).tolist(),
        "test_subject_ids": np.sort(test_subjects).tolist(),
        "subject_ids": subject_ids.tolist(),
        "max_latent_norm_error": max_norm_error,
    }


def cache_metadata(fold: int, path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "fold": fold,
            "checkpoint": str(payload["source_checkpoint"].item()),
            "val_subject_ids": payload["val_subject_ids"].astype(int).tolist(),
            "test_subject_ids": payload["test_subject_ids"].astype(int).tolist(),
            "subject_ids": payload["subject_ids"].astype(int).tolist(),
            "latent_dim": int(payload["latent_dim"].item()),
            "max_latent_norm_error": float(payload["max_latent_norm_error"].item()),
        }


def aggregate_full_cka(
    folds: Sequence[int], output_dir: Path, device: torch.device, block_size: int
) -> tuple[Path, float, float]:
    """Reconstruct fold CKAs by row blocks and store online mean/sample SD."""
    cache_vectors = []
    for fold in folds:
        with np.load(cache_path(output_dir, fold), allow_pickle=False) as payload:
            cache_vectors.append(np.asarray(payload["cka_vectors"], dtype=np.float32))
    summary_path = output_dir / "full_cka_summary.h5"
    temporary = summary_path.with_suffix(".tmp.h5")
    level_indices, values = op_mapping()
    with h5py.File(temporary, "w") as handle:
        chunks = (min(block_size, EXPECTED_OPS), EXPECTED_OPS)
        mean_ds = handle.create_dataset(
            "mean_cka", (EXPECTED_OPS, EXPECTED_OPS), dtype="f4", chunks=chunks
        )
        sd_ds = handle.create_dataset(
            "fold_sd", (EXPECTED_OPS, EXPECTED_OPS), dtype="f4", chunks=chunks
        )
        handle.create_dataset("fold_ids", data=np.asarray(folds, dtype=np.int32))
        handle.create_dataset(
            "num_subjects_per_fold", data=np.full(len(folds), 11, dtype=np.int16)
        )
        handle.create_dataset("op_numbers", data=np.arange(1, EXPECTED_OPS + 1))
        handle.create_dataset("op_level_indices", data=level_indices)
        op_group = handle.create_group("op_mapping")
        for column, name in enumerate(PARAM_NAMES):
            op_group.create_dataset(name, data=values[:, column])
            op_group[name].attrs["levels"] = PARAM_LEVELS[name]
            op_group[name].attrs["stride"] = PARAM_STRIDES[name]
        handle.attrs["cka_estimator"] = "U-centered/debiased linear CKA"
        handle.attrs["raw_values_clipped"] = False

        # Accumulate one complete Gram matrix per fold.  Computing separate row
        # blocks on CUDA produced inconsistent values for A[i, j] and A[j, i]
        # on the project workstation.  A full 2700 x 2700 float32 matrix is only
        # about 28 MiB, so forming it once per fold is both safer and faster.
        mean = np.zeros((EXPECTED_OPS, EXPECTED_OPS), dtype=np.float64)
        m2 = np.zeros_like(mean)
        for count, vectors in enumerate(cache_vectors, start=1):
            if device.type == "cuda":
                tensor = torch.from_numpy(vectors).to(device)
                matrix = (tensor @ tensor.T).cpu().numpy()
                del tensor
            else:
                matrix = vectors @ vectors.T
            matrix = (matrix + matrix.T) * 0.5
            np.fill_diagonal(matrix, 1.0)
            delta = matrix - mean
            mean += delta / count
            m2 += delta * (matrix - mean)

        sd = (
            np.sqrt(m2 / (len(folds) - 1))
            if len(folds) > 1
            else np.zeros_like(m2)
        )
        # Remove small direction-dependent numerical drift from repeated GPU
        # matrix multiplications and online updates.
        mean = np.ascontiguousarray((mean + mean.T) * 0.5)
        sd = np.ascontiguousarray((sd + sd.T) * 0.5)
        np.fill_diagonal(mean, 1.0)
        np.fill_diagonal(sd, 0.0)

        # A single write avoids corrupted cross-chunk values observed when this
        # HDF5 file was populated through repeated row-slice assignments.
        mean_ds[...] = mean.astype(np.float32)
        sd_ds[...] = sd.astype(np.float32)
        print(f"  wrote full CKA matrix: {EXPECTED_OPS} x {EXPECTED_OPS}")

    temporary.replace(summary_path)
    with h5py.File(summary_path, "r+") as handle:
        mean_cka = handle["mean_cka"][:]
        if not np.isfinite(mean_cka).all():
            raise FloatingPointError("Aggregated mean CKA contains NaN or Inf")
        symmetry_error = float(np.max(np.abs(mean_cka - mean_cka.T)))
        diagonal_error = float(np.max(np.abs(np.diag(mean_cka) - 1.0)))
        if symmetry_error > 2e-5 or diagonal_error > 2e-5:
            raise AssertionError(
                f"CKA validation failed: symmetry={symmetry_error}, diagonal={diagonal_error}"
            )
        mask = ~np.eye(EXPECTED_OPS, dtype=bool)
        vmin, vmax = np.quantile(mean_cka[mask], [0.01, 0.99]).astype(float)
        handle.attrs["default_heatmap_vmin"] = vmin
        handle.attrs["default_heatmap_vmax"] = vmax
        handle.attrs["symmetry_max_abs_error"] = symmetry_error
        handle.attrs["diagonal_max_abs_error"] = diagonal_error
    return summary_path, float(vmin), float(vmax)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def compute_transition_metrics(
    folds: Sequence[int], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    level_indices, _ = op_mapping()
    vectors_by_fold = {}
    for fold in folds:
        with np.load(cache_path(output_dir, fold), allow_pickle=False) as payload:
            vectors_by_fold[fold] = np.asarray(payload["cka_vectors"], dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    expected_counts = {
        "mu_a_scalp": 540,
        "mu_a_skull": 540,
        "mu_a_gm": 675,
        "mu_s_gm": 900,
        "mu_s_skull": 900,
        "mu_s_scalp": 900,
    }
    for name in PARAM_NAMES:
        for low, high, endpoint in adjacent_and_endpoint_levels(name):
            left, right = matched_transition_pairs(name, low, high, level_indices)
            if len(left) != expected_counts[name]:
                raise AssertionError(f"{name} matched-pair count {len(left)} != {expected_counts[name]}")
            pair_by_fold = []
            for fold in folds:
                vectors = vectors_by_fold[fold]
                pair_values = np.einsum("ij,ij->i", vectors[left], vectors[right])
                if not np.isfinite(pair_values).all():
                    raise FloatingPointError(f"Fold {fold} transition {name} has non-finite CKA")
                group_mean = float(pair_values.mean())
                pair_by_fold.append(pair_values)
                fold_rows.append(
                    {
                        "fold": fold,
                        "parameter": name,
                        "family": "absorption" if name.startswith("mu_a") else "scattering",
                        "transition": transition_label(name, low, high),
                        "low_level_index": low,
                        "high_level_index": high,
                        "is_endpoint": endpoint,
                        "matched_pair_count": len(left),
                        "fold_group_mean_cka": group_mean,
                    }
                )
            pair_array = np.stack(pair_by_fold)
            fold_means = pair_array.mean(axis=1)
            pair_cross_fold_means = pair_array.mean(axis=0)
            row = {
                "parameter": name,
                "family": "absorption" if name.startswith("mu_a") else "scattering",
                "transition": transition_label(name, low, high),
                "low_level_index": low,
                "high_level_index": high,
                "is_endpoint": endpoint,
                "matched_pair_count": len(left),
                "num_folds": len(folds),
                "mean_cka": float(fold_means.mean()),
                "fold_sd": float(fold_means.std(ddof=1)) if len(folds) > 1 else 0.0,
                "fold_median": float(np.median(fold_means)),
                "fold_q05": float(np.quantile(fold_means, 0.05)),
                "fold_q25": float(np.quantile(fold_means, 0.25)),
                "fold_q75": float(np.quantile(fold_means, 0.75)),
                "fold_q95": float(np.quantile(fold_means, 0.95)),
                "pair_cross_fold_q05": float(np.quantile(pair_cross_fold_means, 0.05)),
                "pair_cross_fold_q95": float(np.quantile(pair_cross_fold_means, 0.95)),
            }
            summary_rows.append(row)
            if endpoint:
                ranking.append(
                    {
                        "parameter": name,
                        "transition": row["transition"],
                        "endpoint_mean_cka": row["mean_cka"],
                        "rank_interpretation": "lower CKA indicates greater endpoint sensitivity",
                    }
                )
    if len(summary_rows) != 23:
        raise AssertionError(f"Expected 23 controlled transitions, got {len(summary_rows)}")
    ranking.sort(key=lambda row: row["endpoint_mean_cka"])
    for rank, row in enumerate(ranking, 1):
        row["sensitivity_rank"] = rank
    _write_csv(output_dir / "op_transition_fold_metrics.csv", fold_rows)
    _write_csv(output_dir / "op_transition_summary.csv", summary_rows)
    (output_dir / "endpoint_sensitivity_ranking.json").write_text(
        json.dumps(ranking, indent=2), encoding="utf-8"
    )
    print("Endpoint sensitivity ranking (lowest mean CKA = most sensitive):")
    for row in ranking:
        print(
            f"  {row['sensitivity_rank']}. {row['parameter']}: "
            f"{row['endpoint_mean_cka']:.6f} ({row['transition']})"
        )
    return fold_rows, summary_rows, ranking


def choose_interaction_axes(target: str) -> tuple[str, str]:
    family_prefix = "mu_a" if target.startswith("mu_a") else "mu_s"
    alternatives = [name for name in PARAM_NAMES if name.startswith(family_prefix) and name != target]
    if len(alternatives) != 2:
        raise AssertionError(f"Cannot choose two interaction axes for {target}")
    # Anatomical display preference: scalp on x, GM on y where available.
    x = next((name for name in alternatives if name.endswith("scalp")), alternatives[-1])
    y = next((name for name in alternatives if name.endswith("gm")), alternatives[0])
    if x == y:
        y = next(name for name in alternatives if name != x)
    return x, y


def compute_conditional_summary(
    folds: Sequence[int],
    output_dir: Path,
    target: str,
    axis_x: str | None,
    axis_y: str | None,
    ranking: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str, str, str]:
    if target == "auto":
        target = str(ranking[0]["parameter"])
        axis_x, axis_y = choose_interaction_axes(target)
    if target not in PARAM_NAMES:
        raise ValueError(f"Unknown target_param {target}")
    if axis_x is None or axis_y is None:
        default_x, default_y = choose_interaction_axes(target)
        axis_x = axis_x or default_x
        axis_y = axis_y or default_y
    if axis_x not in PARAM_NAMES or axis_y not in PARAM_NAMES:
        raise ValueError("Unknown interaction axis")
    if len({target, axis_x, axis_y}) != 3:
        raise ValueError("target_param and interaction axes must be distinct")

    level_indices, _ = op_mapping()
    target_col = PARAM_NAMES.index(target)
    x_col = PARAM_NAMES.index(axis_x)
    y_col = PARAM_NAMES.index(axis_y)
    vectors_by_fold = {}
    for fold in folds:
        with np.load(cache_path(output_dir, fold), allow_pickle=False) as payload:
            vectors_by_fold[fold] = np.asarray(payload["cka_vectors"], dtype=np.float32)

    rows: list[dict[str, Any]] = []
    for low, high, endpoint in adjacent_and_endpoint_levels(target):
        for y_level in range(len(PARAM_LEVELS[axis_y])):
            for x_level in range(len(PARAM_LEVELS[axis_x])):
                mask = (
                    (level_indices[:, target_col] == low)
                    & (level_indices[:, x_col] == x_level)
                    & (level_indices[:, y_col] == y_level)
                )
                left = np.flatnonzero(mask)
                right = left + (high - low) * PARAM_STRIDES[target]
                fold_means = []
                all_values = []
                for fold in folds:
                    vectors = vectors_by_fold[fold]
                    values = np.einsum("ij,ij->i", vectors[left], vectors[right])
                    all_values.append(values)
                    fold_means.append(float(values.mean()))
                fold_means_array = np.asarray(fold_means)
                all_values_array = np.concatenate(all_values)
                rows.append(
                    {
                        "target_parameter": target,
                        "transition": transition_label(target, low, high),
                        "low_level_index": low,
                        "high_level_index": high,
                        "is_endpoint": endpoint,
                        "interaction_axis_x": axis_x,
                        "x_level_index": x_level,
                        "x_value": PARAM_LEVELS[axis_x][x_level],
                        "interaction_axis_y": axis_y,
                        "y_level_index": y_level,
                        "y_value": PARAM_LEVELS[axis_y][y_level],
                        "matched_pair_count_per_fold": len(left),
                        "matched_pair_count_total": len(all_values_array),
                        "num_folds": len(folds),
                        "mean_cka": float(all_values_array.mean()),
                        "across_fold_sd": float(fold_means_array.std(ddof=1)) if len(folds) > 1 else 0.0,
                        "fold_median": float(np.median(fold_means_array)),
                        "fold_q05": float(np.quantile(fold_means_array, 0.05)),
                        "fold_q95": float(np.quantile(fold_means_array, 0.95)),
                    }
                )
    _write_csv(output_dir / "conditional_transition_summary.csv", rows)
    return rows, target, axis_x, axis_y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--experiment", default="relat_cons_32")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--matrix_block_size", type=int, default=256)
    parser.add_argument("--heatmap_vmin", type=float, default=None)
    parser.add_argument("--heatmap_vmax", type=float, default=None)
    parser.add_argument("--target_param", default="mu_s_skull")
    parser.add_argument("--interaction_axis_x", default="mu_s_scalp")
    parser.add_argument("--interaction_axis_y", default="mu_s_gm")
    parser.add_argument("--skip_plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    result_root = args.result_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else result_root / "result" / f"op_cka_{args.experiment}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(result_root, args.experiment, args.folds)
    if not checkpoints:
        raise FileNotFoundError("No matching checkpoints found")
    if args.folds is None and len(checkpoints) != EXPECTED_FOLDS and not args.allow_incomplete:
        raise RuntimeError(
            f"Found {len(checkpoints)} folds, expected {EXPECTED_FOLDS}; use --allow_incomplete to proceed"
        )
    device = resolve_device(args.device)
    pending = []
    for fold, checkpoint in checkpoints:
        path = cache_path(output_dir, fold)
        cached = None if args.overwrite_cache else load_valid_cache(path, checkpoint, args.experiment)
        if cached is None:
            pending.append((fold, checkpoint))
        else:
            cached.close()
    print(
        f"Found {len(checkpoints)} fold(s); {len(pending)} require inference; "
        f"device={device}, amp={bool(args.amp and device.type == 'cuda')}"
    )
    raw_dtof = load_raw_dtof(args.data_dir) if pending else None
    try:
        for number, (fold, checkpoint) in enumerate(pending, 1):
            print(f"[{number}/{len(pending)}] inferring fold {fold}: {checkpoint}")
            info = infer_fold_cache(
                fold,
                checkpoint,
                raw_dtof,
                output_dir,
                args.experiment,
                device,
                args.batch_size,
                args.amp,
            )
            print(
                f"  cache complete: shape=(2700,121), subjects={info['subject_ids']}, "
                f"latent norm max error={info['max_latent_norm_error']:.3g}"
            )
    finally:
        if raw_dtof is not None:
            raw_dtof.close()

    folds = [fold for fold, _ in checkpoints]
    print("Aggregating full OP-pair CKA matrix...")
    summary_path, default_vmin, default_vmax = aggregate_full_cka(
        folds, output_dir, device, args.matrix_block_size
    )
    _, _, ranking = compute_transition_metrics(folds, output_dir)
    _, target, axis_x, axis_y = compute_conditional_summary(
        folds,
        output_dir,
        args.target_param,
        args.interaction_axis_x,
        args.interaction_axis_y,
        ranking,
    )
    display_vmin = default_vmin if args.heatmap_vmin is None else args.heatmap_vmin
    display_vmax = default_vmax if args.heatmap_vmax is None else args.heatmap_vmax
    if display_vmin >= display_vmax:
        raise ValueError("heatmap_vmin must be smaller than heatmap_vmax")
    figure_metadata = {
        "heatmap_vmin": display_vmin,
        "heatmap_vmax": display_vmax,
        "default_percentiles": "1st-99th percentile of all off-diagonal mean CKA values",
        "raw_matrix_clipped": False,
        "target_param": target,
        "interaction_axis_x": axis_x,
        "interaction_axis_y": axis_y,
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(figure_metadata, indent=2), encoding="utf-8"
    )
    if not args.skip_plots:
        from .cka_plotting import plot_all

        plot_all(output_dir, display_vmin, display_vmax)

    fold_metadata = [cache_metadata(fold, cache_path(output_dir, fold)) for fold in folds]
    latent_dimensions = sorted({item["latent_dim"] for item in fold_metadata})
    if len(latent_dimensions) != 1:
        raise ValueError(f"Checkpoints use inconsistent latent dimensions: {latent_dimensions}")
    latent_dimension = latent_dimensions[0]
    level_indices, values = op_mapping()
    metadata = {
        "experiment": args.experiment,
        "latent_dimension": latent_dimension,
        "folds": folds,
        "num_folds": len(folds),
        "checkpoint_paths": [str(path) for _, path in checkpoints],
        "inference_subjects_by_fold": fold_metadata,
        "inference_subject_description": "validation and test subjects (held-out inference subjects)",
        "op_mapping": {
            "order": list(PARAM_NAMES),
            "levels": {name: PARAM_LEVELS[name].tolist() for name in PARAM_NAMES},
            "strides": PARAM_STRIDES,
            "zero_based_formula": "i_mu_a_scalp + 5*i_mu_a_skull + 25*i_mu_a_gm + 100*i_mu_s_gm + 300*i_mu_s_skull + 900*i_mu_s_scalp",
            "numbering_in_figures": "one-based k+1",
            "shape": list(values.shape),
        },
        "cka_estimator": "U-centered/debiased linear CKA; diagonal-zero Gram; flattened Frobenius normalization; negative values retained",
        "full_cka_summary": str(summary_path),
        "heatmap_display_range": [display_vmin, display_vmax],
        "heatmap_default_percentile_range": [default_vmin, default_vmax],
        "random_seed": None,
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
            "torch": torch.__version__,
        },
        "device": str(device),
        "amp": bool(args.amp and device.type == "cuda"),
        "batch_size": args.batch_size,
        "started_utc": started_iso,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Analysis complete: {output_dir}")


if __name__ == "__main__":
    main()
