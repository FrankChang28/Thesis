"""MAT-file access and validation-cache preprocessing."""

from __future__ import annotations

import os
import time
from typing import Dict

import h5py
import numpy as np

from .config import CFG
from .utils import ensure_dir, write_json


EPS = 1e-12


def target_cache_tag() -> str:
    """The cache stores both regression targets regardless of active target."""
    return "all_targets"


def cache_paths(split: str) -> Dict[str, str]:
    prefix = split.lower()
    return {
        "x": os.path.join(CFG.cache_dir, f"{prefix}_x_preprocessed.npy"),
        "y": os.path.join(
            CFG.cache_dir,
            f"{prefix}_y_{target_cache_tag()}.npy",
        ),
        "subj": os.path.join(CFG.cache_dir, f"{prefix}_subj_0based.npy"),
    }


def cache_metadata_path() -> str:
    return os.path.join(
        CFG.cache_dir,
        f"cache_metadata_{target_cache_tag()}_logod1.json",
    )


def cache_exists() -> bool:
    """Check that the validation cache exists and has the expected layout."""
    paths = cache_paths("val")
    if not all(os.path.exists(path) for path in paths.values()):
        return False
    if not os.path.exists(cache_metadata_path()):
        return False

    try:
        x_array = np.load(paths["x"], mmap_mode="r")
        y_array = np.load(paths["y"], mmap_mode="r")
        subject_array = np.load(paths["subj"], mmap_mode="r")
    except (OSError, ValueError):
        return False

    return (
        x_array.ndim == 2
        and x_array.shape[1] == CFG.input_dim
        and y_array.ndim == 2
        and y_array.shape[1] == CFG.cache_target_dim
        and subject_array.ndim == 1
        and len(x_array) == len(y_array) == len(subject_array)
    )


def read_scalar(file: h5py.File, name: str) -> int:
    return int(np.asarray(file[name]).squeeze())


def _sorted_rows(rows: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(rows, dtype=np.int64).reshape(-1))


def read_x_rows(dataset, rows: np.ndarray) -> np.ndarray:
    """Read X rows from either MATLAB/HDF5 orientation."""
    rows = _sorted_rows(rows)
    if dataset.shape[1] == CFG.input_dim:
        return np.asarray(dataset[rows, :], dtype=np.float32)
    if dataset.shape[0] == CFG.input_dim:
        return np.asarray(dataset[:, rows], dtype=np.float32).T
    raise ValueError(f"Unexpected X shape: {dataset.shape}")


def read_y_rows(dataset, rows: np.ndarray) -> np.ndarray:
    """Read all target columns from either MATLAB/HDF5 orientation."""
    rows = _sorted_rows(rows)
    if dataset.shape[1] == CFG.cache_target_dim:
        values = np.asarray(dataset[rows, :], dtype=np.float32)
    elif dataset.shape[0] == CFG.cache_target_dim:
        values = np.asarray(dataset[:, rows], dtype=np.float32).T
    else:
        raise ValueError(f"Unexpected Y shape: {dataset.shape}")
    return values[:, : CFG.cache_target_dim]


def select_target_columns(y: np.ndarray) -> np.ndarray:
    """Select the active single target from the shared two-target cache."""
    y = np.asarray(y)
    expected = ("N", CFG.cache_target_dim)
    if y.ndim != 2 or y.shape[1] != CFG.cache_target_dim:
        raise ValueError(f"Expected cached Y shape {expected}, got {y.shape}")
    return y[:, CFG.target_index : CFG.target_index + 1]


def _subject_values(dataset, rows) -> np.ndarray:
    if dataset.shape[1] == 1:
        values = np.asarray(dataset[rows, 0])
    elif dataset.shape[0] == 1:
        values = np.asarray(dataset[0, rows])
    else:
        raise ValueError(f"Unexpected subject shape: {dataset.shape}")
    # MATLAB subject IDs are one-based; all Python code uses zero-based IDs.
    return np.asarray(values, dtype=np.int64).reshape(-1) - 1


def read_subject_rows(dataset, start: int, end: int) -> np.ndarray:
    return _subject_values(dataset, slice(start, end))


def read_subject_selected_rows(dataset, rows: np.ndarray) -> np.ndarray:
    return _subject_values(dataset, _sorted_rows(rows))


def preprocess_drs_x(x_raw: np.ndarray) -> np.ndarray:
    """Convert reflectance to per-sample relative optical density.

    Input columns are ordered as five wavelengths by six SDS positions. Each
    sample is divided by its mean reflectance before applying ``-log10``.
    """
    x_raw = np.asarray(x_raw, dtype=np.float32)
    if x_raw.ndim != 2 or x_raw.shape[1] != CFG.input_dim:
        raise ValueError(
            f"Expected raw DRS shape [N, {CFG.input_dim}], got {x_raw.shape}"
        )

    x = x_raw.reshape(-1, CFG.num_wl, CFG.num_sds)
    sample_mean = np.mean(x, axis=(1, 2), keepdims=True)
    relative_reflectance = np.maximum(x / (sample_mean + EPS), EPS)
    optical_density = -np.log10(relative_reflectance)
    return optical_density.reshape(-1, CFG.input_dim).astype(np.float32)


def build_preprocessed_cache() -> None:
    """Build the full validation cache.

    Training rows are intentionally not cached in full. They are sampled first
    and written as compact per-subject caches by ``compact_cache.py``.
    """
    ensure_dir(CFG.cache_dir)
    if cache_exists() and not CFG.rebuild_preprocessed_cache:
        print("Using existing preprocessed validation cache.")
        return

    print("Building preprocessed validation cache...")
    paths = cache_paths("val")
    with h5py.File(CFG.mat_file, "r") as file:
        n_rows = read_scalar(file, "num_val_rows")
        x_out = np.lib.format.open_memmap(
            paths["x"],
            mode="w+",
            dtype=np.float32,
            shape=(n_rows, CFG.input_dim),
        )
        y_out = np.lib.format.open_memmap(
            paths["y"],
            mode="w+",
            dtype=np.float32,
            shape=(n_rows, CFG.cache_target_dim),
        )
        subject_out = np.lib.format.open_memmap(
            paths["subj"],
            mode="w+",
            dtype=np.int64,
            shape=(n_rows,),
        )

        for start in range(0, n_rows, CFG.cache_chunk_rows):
            end = min(start + CFG.cache_chunk_rows, n_rows)
            rows = np.arange(start, end, dtype=np.int64)
            x_out[start:end] = preprocess_drs_x(
                read_x_rows(file["Val_Dataset_X"], rows)
            )
            y_out[start:end] = read_y_rows(file["Val_Dataset_Y"], rows)
            subject_out[start:end] = read_subject_rows(
                file["Val_Dataset_SubjID"],
                start,
                end,
            )
            print(f"  cached {end}/{n_rows}")

        x_out.flush()
        y_out.flush()
        subject_out.flush()

    write_json(
        cache_metadata_path(),
        {
            "mat_file": CFG.mat_file,
            "rows": n_rows,
            "cache_target_names": list(CFG.cache_target_names),
            "input_dim": CFG.input_dim,
            "preprocessing": "per_sample_mean_then_negative_log10",
            "created_unix_time": time.time(),
            "paths": paths,
        },
    )
    print("Preprocessed validation cache finished.")


def load_cache_split(split: str):
    paths = cache_paths(split)
    return (
        np.load(paths["x"], mmap_mode="r"),
        np.load(paths["y"], mmap_mode="r"),
        np.load(paths["subj"], mmap_mode="r"),
    )
