"""Iterable datasets for cached DRS arrays and one subject-level Z_s per subject."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .cache import select_target_columns
from .config import CFG
from .latent import lookup_subject_features
from .sampling import ranges_to_batches


EPS = 1e-8


def normalize_arrays(
    x: np.ndarray,
    y: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize DRS by min-max and targets by training-fold z-score."""
    x = (x - x_min) / (x_max - x_min + EPS)
    y = (y - y_mean) / (y_std + EPS)
    return x.astype(np.float32), y.astype(np.float32)


def _pack_batch(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    structure_bank: np.ndarray | None,
):
    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    if structure_bank is None:
        return x_t, y_t

    z = lookup_subject_features(subject_ids, structure_bank)
    z_t = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
    return x_t, z_t, y_t


class NIRSDataset(IterableDataset):
    """Iterate arbitrary ranges from memory-mapped validation arrays."""
    def __init__(
        self,
        x_arr,
        y_arr,
        subj_arr,
        ranges,
        x_min,
        x_max,
        y_mean,
        y_std,
        latent_bank=None,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.subj_arr = subj_arr
        self.ranges = list(ranges)
        self.x_min = x_min
        self.x_max = x_max
        self.y_mean = y_mean
        self.y_std = y_std
        self.structure_bank = latent_bank
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        for rows in ranges_to_batches(
            self.ranges,
            CFG.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        ):
            x = np.asarray(self.x_arr[rows], dtype=np.float32)
            y_all = np.asarray(self.y_arr[rows], dtype=np.float32)
            y = select_target_columns(y_all)
            subject_ids = np.asarray(self.subj_arr[rows], dtype=np.int64)

            x, y = normalize_arrays(
                x,
                y,
                self.x_min,
                self.x_max,
                self.y_mean,
                self.y_std,
            )
            yield _pack_batch(x, y, subject_ids, self.structure_bank)


class MultiCompactNIRSDataset(IterableDataset):
    """Stream and mix multiple per-subject compact training caches."""
    def __init__(
        self,
        infos: Sequence[dict],
        x_min,
        x_max,
        y_mean,
        y_std,
        latent_bank=None,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.infos = list(infos)
        self.x_min = x_min
        self.x_max = x_max
        self.y_mean = y_mean
        self.y_std = y_std
        self.structure_bank = latent_bank
        self.shuffle = shuffle
        self.drop_last = drop_last

    def _chunk_index(self) -> list[tuple[int, int, int]]:
        chunk_rows = min(
            CFG.compact_cache_chunk_rows,
            max(CFG.batch_size, 8192),
        )

        chunks = []
        for info_idx, info in enumerate(self.infos):
            n_rows = int(info["n_rows"])
            for start in range(0, n_rows, chunk_rows):
                chunks.append((info_idx, start, min(start + chunk_rows, n_rows)))

        if self.shuffle:
            random.shuffle(chunks)

        return chunks

    def __iter__(self):
        arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        subject_parts: list[np.ndarray] = []
        buffered_rows = 0

        for info_idx, start, end in self._chunk_index():
            if info_idx not in arrays:
                info = self.infos[info_idx]
                arrays[info_idx] = (
                    np.load(info["x"], mmap_mode="r"),
                    np.load(info["y"], mmap_mode="r"),
                    np.load(info["subj"], mmap_mode="r"),
                )

            x_arr, y_arr, subj_arr = arrays[info_idx]
            rows = np.arange(start, end, dtype=np.int64)
            if self.shuffle:
                np.random.shuffle(rows)

            x_parts.append(np.asarray(x_arr[rows], dtype=np.float32))
            y_parts.append(
                select_target_columns(np.asarray(y_arr[rows], dtype=np.float32))
            )
            subject_parts.append(np.asarray(subj_arr[rows], dtype=np.int64))
            buffered_rows += len(rows)

            while buffered_rows >= CFG.batch_size:
                x_all = np.concatenate(x_parts, axis=0)
                y_all = np.concatenate(y_parts, axis=0)
                subject_all = np.concatenate(subject_parts, axis=0)

                x_batch = x_all[: CFG.batch_size]
                y_batch = y_all[: CFG.batch_size]
                subject_batch = subject_all[: CFG.batch_size]

                x_remainder = x_all[CFG.batch_size :]
                y_remainder = y_all[CFG.batch_size :]
                subject_remainder = subject_all[CFG.batch_size :]

                x_parts = [x_remainder] if len(x_remainder) else []
                y_parts = [y_remainder] if len(y_remainder) else []
                subject_parts = [subject_remainder] if len(subject_remainder) else []
                buffered_rows = len(x_remainder)

                if self.shuffle:
                    order = np.random.permutation(CFG.batch_size)
                    x_batch = x_batch[order]
                    y_batch = y_batch[order]
                    subject_batch = subject_batch[order]

                x_batch, y_batch = normalize_arrays(
                    x_batch,
                    y_batch,
                    self.x_min,
                    self.x_max,
                    self.y_mean,
                    self.y_std,
                )
                yield _pack_batch(
                    x_batch,
                    y_batch,
                    subject_batch,
                    self.structure_bank,
                )

        if buffered_rows > 0 and not self.drop_last:
            x_batch = np.concatenate(x_parts, axis=0)
            y_batch = np.concatenate(y_parts, axis=0)
            subject_batch = np.concatenate(subject_parts, axis=0)

            x_batch, y_batch = normalize_arrays(
                x_batch,
                y_batch,
                self.x_min,
                self.x_max,
                self.y_mean,
                self.y_std,
            )
            yield _pack_batch(
                x_batch,
                y_batch,
                subject_batch,
                self.structure_bank,
            )
