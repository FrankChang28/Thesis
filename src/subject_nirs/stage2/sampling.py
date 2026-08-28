"""Subject range construction and train-row sampling utilities."""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from .config import CFG


def fixed_subject_ranges(
    n_rows: int,
    rows_per_subject: int,
    name: str,
) -> Dict[int, List[Tuple[int, int]]]:
    """Map each subject to its contiguous rows in a fixed-layout split."""
    expected = CFG.num_subjects * rows_per_subject
    if n_rows != expected:
        raise ValueError(f"{name} rows mismatch: expected {expected}, got {n_rows}")
    return {
        subject_id: [
            (
                subject_id * rows_per_subject,
                (subject_id + 1) * rows_per_subject,
            )
        ]
        for subject_id in range(CFG.num_subjects)
    }


def train_sim_ranges(subject_id: int) -> List[Tuple[int, int]]:
    base = int(subject_id) * CFG.train_rows_per_subject
    return [
        (
            base + i * CFG.train_rows_per_sim,
            base + (i + 1) * CFG.train_rows_per_sim,
        )
        for i in range(CFG.train_sims_per_subject)
    ]


def allocate_counts(
    total_count: int,
    lengths: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate samples proportionally without exceeding segment lengths."""
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0 or np.any(lengths < 0):
        raise ValueError(
            "lengths must be a non-empty sequence of non-negative values"
        )
    total_capacity = int(lengths.sum())
    if total_count < 0 or total_count > total_capacity:
        raise ValueError("total_count must be between zero and the total capacity")
    if total_capacity == 0:
        return np.zeros_like(lengths)

    raw = lengths / total_capacity * total_count
    alloc = np.floor(raw).astype(np.int64)
    alloc = np.minimum(alloc, lengths)

    leftover = int(total_count - alloc.sum())
    if leftover > 0:
        remainder = raw - np.floor(raw) + rng.random(len(raw)) * 1e-9
        for idx in np.argsort(-remainder):
            if leftover <= 0:
                break
            if alloc[idx] < lengths[idx]:
                alloc[idx] += 1
                leftover -= 1
    return alloc


def sample_windows(
    start: int,
    end: int,
    k: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """Sample approximately uniform contiguous windows inside one simulation."""
    length = end - start
    if k <= 0:
        return []
    if k >= length:
        return [(start, end)]

    n_windows = min(CFG.sim_windows_per_sim, k, length)
    edges = np.linspace(start, end, n_windows + 1).astype(np.int64)
    segment_lengths = np.diff(edges)
    alloc = allocate_counts(k, segment_lengths, rng)

    ranges = []
    for i, win_len in enumerate(alloc):
        win_len = int(win_len)
        if win_len <= 0:
            continue
        seg_start = int(np.floor(edges[i]))
        seg_end = int(np.floor(edges[i + 1]))
        seg_end = max(seg_end, seg_start + 1)
        if seg_end - seg_start <= win_len:
            w_start = seg_start
        else:
            w_start = int(rng.integers(seg_start, seg_end - win_len + 1))
        ranges.append((w_start, min(w_start + win_len, end)))
    return ranges


def sample_train_ranges_for_subject(
    subject_id: int,
    seed: int,
) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    sim_ranges = train_sim_ranges(subject_id)
    lengths = [end - start for start, end in sim_ranges]
    alloc = allocate_counts(CFG.max_train_rows_per_subject, lengths, rng)

    sampled = []
    for (start, end), k in zip(sim_ranges, alloc):
        sampled.extend(sample_windows(start, end, int(k), rng))
    return sampled


def ranges_to_batches(
    ranges: Sequence[Tuple[int, int]],
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> Iterator[np.ndarray]:
    """Yield sorted row-index batches suitable for HDF5 fancy indexing."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ranges = [(int(a), int(b)) for a, b in ranges if b > a]
    if shuffle:
        random.shuffle(ranges)

    parts = []
    n = 0
    for start, end in ranges:
        rows = np.arange(start, end, dtype=np.int64)
        if shuffle:
            np.random.shuffle(rows)
        offset = 0
        while offset < len(rows):
            take = min(batch_size - n, len(rows) - offset)
            parts.append(rows[offset:offset + take])
            n += take
            offset += take
            if n == batch_size:
                batch = np.concatenate(parts)
                if shuffle:
                    np.random.shuffle(batch)
                yield np.sort(batch)
                parts = []
                n = 0

    if n > 0 and not drop_last:
        batch = np.concatenate(parts)
        if shuffle:
            np.random.shuffle(batch)
        yield np.sort(batch)
