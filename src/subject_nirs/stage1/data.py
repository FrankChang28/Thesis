from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

# Preprocessing constants are intentionally fixed to keep the public config small.
_EPS = 1e-10
_PREPROCESS_ROWS = 20_000
_MAX_QUANTILE_VALUES = 2_000_000


@dataclass
class SplitInfo:
    train_subjects: np.ndarray
    val_subjects: np.ndarray
    test_subjects: np.ndarray


@dataclass
class PreprocessStats:
    temporal_low: float
    temporal_high: float
    area_mean: np.ndarray
    area_std: np.ndarray
    condition_min: np.ndarray
    condition_max: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "temporal_low": float(self.temporal_low),
            "temporal_high": float(self.temporal_high),
            "area_mean": self.area_mean.tolist(),
            "area_std": self.area_std.tolist(),
            "condition_min": self.condition_min.tolist(),
            "condition_max": self.condition_max.tolist(),
        }


@dataclass
class DataBundle:
    temporal: torch.Tensor
    relative_area: torch.Tensor
    conditions: torch.Tensor
    raw_conditions: torch.Tensor
    split: SplitInfo
    stats: PreprocessStats
    num_subjects: int
    op_per_subject: int
    num_sds: int
    num_time_bins: int
    train_group_dataset: "SubjectGroupDataset"
    train_group_sampler: "SubjectGroupBatchSampler"
    val_group_loader: DataLoader

    def make_train_group_loader(self, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self.train_group_dataset,
            batch_sampler=self.train_group_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def make_aligned_loader(
        self,
        split_name: str,
        op_positions: Sequence[int],
        batch_size: int,
        num_workers: int = 0,
    ) -> DataLoader:
        subjects = {
            "train": self.split.train_subjects,
            "val": self.split.val_subjects,
            "test": self.split.test_subjects,
            "all": np.arange(self.num_subjects, dtype=np.int64),
        }[split_name]
        dataset = AlignedDataset(
            self.temporal,
            self.relative_area,
            self.conditions,
            self.raw_conditions,
            subjects,
            np.asarray(op_positions, dtype=np.int64),
            self.op_per_subject,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def make_subject_loader(
        self,
        subjects: Sequence[int],
        op_positions: Sequence[int] | None,
        batch_size: int,
        num_workers: int = 0,
    ) -> DataLoader:
        positions = (
            np.arange(self.op_per_subject, dtype=np.int64)
            if op_positions is None
            else np.asarray(op_positions, dtype=np.int64)
        )
        dataset = AlignedDataset(
            self.temporal,
            self.relative_area,
            self.conditions,
            self.raw_conditions,
            np.asarray(subjects, dtype=np.int64),
            positions,
            self.op_per_subject,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )


class SubjectGroupDataset(Dataset):
    """Return K randomly selected OP acquisitions from one subject."""

    def __init__(
        self,
        temporal: torch.Tensor,
        relative_area: torch.Tensor,
        op_per_subject: int,
        samples_per_subject: int,
    ) -> None:
        self.temporal = temporal
        self.relative_area = relative_area
        self.op_per_subject = int(op_per_subject)
        self.samples_per_subject = int(samples_per_subject)
        _validate_group_size(self.samples_per_subject, self.op_per_subject)

    def __len__(self) -> int:
        return self.temporal.shape[0]

    def __getitem__(self, key: Tuple[int, int]):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Expected (subject_id, random_seed)")
        subject_id, seed = int(key[0]), int(key[1])
        rng = np.random.default_rng(seed)
        positions = np.sort(
            rng.choice(
                self.op_per_subject,
                size=self.samples_per_subject,
                replace=False,
            )
        ).astype(np.int64)
        indices = torch.from_numpy(subject_id * self.op_per_subject + positions)
        return (
            self.temporal.index_select(0, indices),
            self.relative_area.index_select(0, indices),
        )


class SubjectGroupBatchSampler(Sampler[List[Tuple[int, int]]]):
    """Sample distinct subjects and one random group seed for each subject."""

    def __init__(
        self,
        subjects: Sequence[int],
        subjects_per_batch: int,
        steps_per_epoch: int,
        seed: int,
    ) -> None:
        self.subjects = np.asarray(subjects, dtype=np.int64)
        self.subjects_per_batch = min(int(subjects_per_batch), len(self.subjects))
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        if self.subjects_per_batch < 2:
            raise ValueError("At least two subjects are required per batch")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        rng = np.random.default_rng(self.seed + 104729 * self.epoch)
        for _ in range(self.steps_per_epoch):
            selected = rng.choice(
                self.subjects,
                size=self.subjects_per_batch,
                replace=False,
            )
            seeds = rng.integers(0, np.iinfo(np.int32).max, size=len(selected))
            yield [(int(subject), int(seed)) for subject, seed in zip(selected, seeds)]

    def __len__(self) -> int:
        return self.steps_per_epoch


class FixedSubjectGroupDataset(Dataset):
    """Deterministic subject groups used for validation loss."""

    def __init__(
        self,
        temporal: torch.Tensor,
        relative_area: torch.Tensor,
        subjects: Sequence[int],
        op_per_subject: int,
        samples_per_subject: int,
        rounds: int,
        seed: int,
    ) -> None:
        self.temporal = temporal
        self.relative_area = relative_area
        self.op_per_subject = int(op_per_subject)
        self.samples_per_subject = int(samples_per_subject)
        _validate_group_size(self.samples_per_subject, self.op_per_subject)

        rng = np.random.default_rng(seed)
        self.groups: List[Tuple[int, np.ndarray]] = []
        for _ in range(int(rounds)):
            for subject in subjects:
                positions = np.sort(
                    rng.choice(
                        self.op_per_subject,
                        size=self.samples_per_subject,
                        replace=False,
                    )
                ).astype(np.int64)
                self.groups.append((int(subject), positions))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int):
        subject, positions = self.groups[index]
        indices = torch.from_numpy(subject * self.op_per_subject + positions)
        return (
            self.temporal.index_select(0, indices),
            self.relative_area.index_select(0, indices),
        )


class AlignedDataset(Dataset):
    """Return the same selected OP positions for every requested subject."""

    def __init__(
        self,
        temporal: torch.Tensor,
        relative_area: torch.Tensor,
        conditions: torch.Tensor,
        raw_conditions: torch.Tensor,
        subjects: np.ndarray,
        op_positions: np.ndarray,
        op_per_subject: int,
    ) -> None:
        self.temporal = temporal
        self.relative_area = relative_area
        self.conditions = conditions
        self.raw_conditions = raw_conditions
        self.subjects = np.asarray(subjects, dtype=np.int64)
        self.positions = np.asarray(op_positions, dtype=np.int64)
        self.op_per_subject = int(op_per_subject)
        self.indices = np.asarray(
            [
                int(subject) * self.op_per_subject + int(position)
                for subject in self.subjects
                for position in self.positions
            ],
            dtype=np.int64,
        )
        self.ids = np.repeat(self.subjects, len(self.positions))
        self.ops = np.tile(self.positions, len(self.subjects))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        global_index = int(self.indices[index])
        return (
            self.temporal[global_index],
            self.relative_area[global_index],
            self.conditions[global_index],
            self.raw_conditions[global_index],
            torch.tensor(int(self.ids[index]), dtype=torch.long),
            torch.tensor(int(self.ops[index]), dtype=torch.long),
        )


def _validate_group_size(samples_per_subject: int, op_per_subject: int) -> None:
    if samples_per_subject < 4 or samples_per_subject % 2 != 0:
        raise ValueError("samples_per_subject must be even and at least four")
    if samples_per_subject > op_per_subject:
        raise ValueError("samples_per_subject exceeds available OP acquisitions")


def _make_split(num_subjects: int, test_id: int, split_seed: int) -> SplitInfo:
    if num_subjects < 12:
        raise ValueError("At least 12 subjects are required")
    if not 0 <= test_id < num_subjects:
        raise ValueError(f"test_id must be in [0, {num_subjects - 1}]")

    all_subjects = np.arange(num_subjects, dtype=np.int64)
    train_val = all_subjects[all_subjects != int(test_id)]
    train, val = train_test_split(
        train_val,
        test_size=10,
        random_state=int(split_seed) + int(test_id),
        shuffle=True,
    )
    return SplitInfo(
        train_subjects=np.sort(np.asarray(train, dtype=np.int64)),
        val_subjects=np.sort(np.asarray(val, dtype=np.int64)),
        test_subjects=np.asarray([int(test_id)], dtype=np.int64),
    )


def _subject_rows(subjects: np.ndarray, op_per_subject: int) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(
                int(subject) * op_per_subject,
                (int(subject) + 1) * op_per_subject,
                dtype=np.int64,
            )
            for subject in subjects
        ]
    )


def _fit_temporal_range(
    temporal: torch.Tensor,
    train_rows: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(train_rows) > _PREPROCESS_ROWS:
        train_rows = np.sort(
            rng.choice(train_rows, size=_PREPROCESS_ROWS, replace=False)
        )

    row_values = temporal.index_select(0, torch.from_numpy(train_rows)).reshape(-1)
    if row_values.numel() > _MAX_QUANTILE_VALUES:
        chosen = rng.choice(
            row_values.numel(),
            size=_MAX_QUANTILE_VALUES,
            replace=False,
        )
        row_values = row_values[torch.from_numpy(chosen)]

    low, high = np.quantile(row_values.numpy(), [0.001, 0.999]).astype(float)
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-8:
        raise RuntimeError("Invalid temporal preprocessing range")
    print(
        f"Temporal clipping range from {len(train_rows)} training acquisitions: "
        f"low={low:.6f}, high={high:.6f}"
    )
    return float(low), float(high)


def _preprocess(
    temporal: torch.Tensor,
    raw_conditions: torch.Tensor,
    split: SplitInfo,
    op_per_subject: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, PreprocessStats]:
    area = temporal.sum(dim=2).clamp_min(_EPS)
    total_area = area.sum(dim=1, keepdim=True).clamp_min(_EPS)
    relative_area = (area / total_area).clamp_min(_EPS).log10_()

    temporal.div_(area.unsqueeze(-1))
    temporal.clamp_min_(_EPS).log10_()

    train_rows = _subject_rows(split.train_subjects, op_per_subject)
    low, high = _fit_temporal_range(temporal, train_rows, seed)
    temporal.clamp_(low, high).sub_(low).div_(high - low)

    train_index = torch.from_numpy(train_rows)
    area_train = relative_area.index_select(0, train_index)
    area_mean = area_train.mean(dim=0)
    area_std = area_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    relative_area.sub_(area_mean).div_(area_std)

    condition_train = raw_conditions.index_select(0, train_index)
    condition_min = condition_train.min(dim=0).values
    condition_max = condition_train.max(dim=0).values
    denominator = (condition_max - condition_min).clamp_min(1e-8)
    conditions = (raw_conditions - condition_min) / denominator
    constant = (condition_max - condition_min) < 1e-8
    if constant.any():
        conditions[:, constant] = 0.0

    stats = PreprocessStats(
        temporal_low=low,
        temporal_high=high,
        area_mean=area_mean.numpy(),
        area_std=area_std.numpy(),
        condition_min=condition_min.numpy(),
        condition_max=condition_max.numpy(),
    )
    return temporal, relative_area, conditions, stats


def load_data_bundle(args) -> DataBundle:
    data_dir = Path(args.data_dir)
    dtof_path = data_dir / "dataset_dtof.mat"
    op_path = data_dir / "dataset_OP.mat"
    if not dtof_path.is_file():
        raise FileNotFoundError(f"Cannot find {dtof_path}")
    if not op_path.is_file():
        raise FileNotFoundError(f"Cannot find {op_path}")

    print(f"Loading DTOFs from {dtof_path}")
    with h5py.File(dtof_path, "r") as handle:
        raw_dtof = handle["dataset_dtof"][:].transpose(2, 1, 0)
    with h5py.File(op_path, "r") as handle:
        raw_op = handle["dataset_OP"][:].transpose(1, 0)

    temporal = torch.from_numpy(np.asarray(raw_dtof, dtype=np.float32))
    raw_conditions = torch.from_numpy(np.asarray(raw_op, dtype=np.float32))
    del raw_dtof, raw_op

    num_acquisitions = temporal.shape[0]
    num_subjects = int(args.num_subjects)
    if num_acquisitions % num_subjects != 0:
        raise ValueError("Acquisition count is not divisible by num_subjects")

    op_per_subject = num_acquisitions // num_subjects
    num_sds, num_time_bins = temporal.shape[1:]

    condition_grid = raw_conditions.reshape(num_subjects, op_per_subject, -1)
    max_deviation = float((condition_grid - condition_grid[:1]).abs().max().item())
    if max_deviation > 1e-6:
        raise ValueError("OP values/order are not shared across subjects")

    split = _make_split(num_subjects, int(args.test_id), int(args.split_seed))
    temporal, relative_area, conditions, stats = _preprocess(
        temporal,
        raw_conditions,
        split,
        op_per_subject,
        int(args.seed),
    )

    train_dataset = SubjectGroupDataset(
        temporal,
        relative_area,
        op_per_subject,
        int(args.samples_per_subject),
    )
    train_sampler = SubjectGroupBatchSampler(
        split.train_subjects,
        int(args.subjects_per_batch),
        int(args.steps_per_epoch),
        int(args.seed),
    )
    val_dataset = FixedSubjectGroupDataset(
        temporal,
        relative_area,
        split.val_subjects,
        op_per_subject,
        int(args.samples_per_subject),
        int(args.val_group_rounds),
        int(args.seed) + 1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(split.val_subjects),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
    )

    print(
        f"Subject split: train={len(split.train_subjects)}, "
        f"val={len(split.val_subjects)}, test={len(split.test_subjects)}"
    )
    print(
        f"Data shape: N={num_acquisitions}, SDS={num_sds}, T={num_time_bins}, "
        f"OP/subject={op_per_subject}, C={raw_conditions.shape[1]}"
    )
    print(
        f"Training groups: {min(int(args.subjects_per_batch), len(split.train_subjects))} "
        f"subjects x {int(args.samples_per_subject)} acquisitions"
    )
    print("Encoder input is DTOF temporal shape plus DTOF-derived relative area.")

    return DataBundle(
        temporal=temporal,
        relative_area=relative_area,
        conditions=conditions,
        raw_conditions=raw_conditions,
        split=split,
        stats=stats,
        num_subjects=num_subjects,
        op_per_subject=op_per_subject,
        num_sds=num_sds,
        num_time_bins=num_time_bins,
        train_group_dataset=train_dataset,
        train_group_sampler=train_sampler,
        val_group_loader=val_loader,
    )


def choose_eval_op_positions(op_per_subject: int, count: int, seed: int) -> np.ndarray:
    count = min(max(int(count), 4), int(op_per_subject))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(op_per_subject, size=count, replace=False)).astype(np.int64)
