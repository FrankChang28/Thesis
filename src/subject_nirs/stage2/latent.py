"""Load one subject-level feature vector per subject.

Supported sources
-----------------
dtof:
    DTOF-derived Z_s loaded from NPZ and standardized with the current
    fold's training subjects only.

metadata:
    Four tissue-thickness features loaded from CSV columns 5:9 and min-max
    normalized with the current fold's training subjects only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CFG


EPS = 1e-8


def _validate_train_subjects(train_subjects: Sequence[int]) -> np.ndarray:
    subject_ids = np.asarray(train_subjects, dtype=np.int64).reshape(-1)

    if subject_ids.size == 0:
        raise ValueError("train_subjects cannot be empty")

    if subject_ids.min() < 0 or subject_ids.max() >= CFG.num_subjects:
        raise ValueError("train_subjects contains an invalid subject ID")

    if np.unique(subject_ids).size != subject_ids.size:
        raise ValueError("train_subjects contains duplicate subject IDs")

    return subject_ids


def resolve_structure_path(test_id: int) -> Path:
    """Resolve a static NPZ path or a fold-specific path template."""
    path = Path(
        str(CFG.structure_feature_path).format(
            test_id=test_id,
            test_id_1based=test_id + 1,
        )
    ).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Structure feature file not found: {path}"
        )

    return path


def _reorder_by_subject_ids(
    features: np.ndarray,
    subject_ids: np.ndarray | None,
) -> np.ndarray:
    """Reorder feature rows to zero-based subject order 0..S-1."""
    if subject_ids is None:
        return features

    subject_ids = np.asarray(subject_ids, dtype=np.int64).reshape(-1)

    if subject_ids.shape[0] != features.shape[0]:
        raise ValueError(
            "subject_ids length does not match the number of feature rows"
        )

    expected = np.arange(CFG.num_subjects, dtype=np.int64)

    if np.array_equal(subject_ids, expected):
        return features

    if not np.array_equal(np.sort(subject_ids), expected):
        raise ValueError(
            "subject_ids must contain every zero-based subject ID exactly once"
        )

    return features[np.argsort(subject_ids)]


def _standardize_from_training_subjects(
    features: np.ndarray,
    train_subjects: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score every feature using training subjects only."""
    train_values = features[train_subjects]

    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)

    normalized = (features - mean) / std

    return (
        normalized.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def _minmax_from_training_subjects(
    features: np.ndarray,
    train_subjects: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalize every feature using training subjects only."""
    train_values = features[train_subjects]

    minimum = train_values.min(axis=0, keepdims=True)
    maximum = train_values.max(axis=0, keepdims=True)
    value_range = maximum - minimum
    value_range = np.where(value_range < EPS, 1.0, value_range)

    normalized = (features - minimum) / value_range

    return (
        normalized.astype(np.float32),
        minimum.astype(np.float32),
        maximum.astype(np.float32),
    )


def _apply_control(
    features: np.ndarray,
    test_id: int,
    train_subjects: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """Apply actual, zero, or leakage-safe shuffled subject features."""
    mode = str(CFG.structure_control).strip().lower()

    if mode == "actual":
        return features, "actual"

    if mode == "zero":
        return np.zeros_like(features), "zero"

    if mode != "shuffle":
        raise ValueError(
            f"Unknown structure_control={CFG.structure_control!r}"
        )

    rng = np.random.default_rng(
        int(CFG.structure_shuffle_seed) + int(test_id)
    )

    shuffled = features.copy()
    all_subjects = np.arange(CFG.num_subjects, dtype=np.int64)
    heldout_subjects = np.setdiff1d(
        all_subjects,
        train_subjects,
        assume_unique=False,
    )

    descriptions = []

    # Shuffle inside each split so validation/test features are never moved
    # into training subjects during the negative-control experiment.
    for name, group in (
        ("train", train_subjects),
        ("heldout", heldout_subjects),
    ):
        group = np.asarray(group, dtype=np.int64)

        if group.size <= 1:
            # A derangement is impossible for a singleton. Zero is safer than
            # leaving the subject's true feature attached to itself.
            shuffled[group] = 0.0
            descriptions.append(f"{name}_zero_singleton")
            continue

        shift = int(rng.integers(1, group.size))
        source_ids = np.roll(group, shift)
        shuffled[group] = features[source_ids]
        descriptions.append(f"{name}_shift_{shift}")

    return shuffled.astype(np.float32), "shuffle:" + ",".join(descriptions)


def load_dtof_bank(
    test_id: int,
    train_subjects: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """Load one DTOF-derived Z_s vector per subject from an NPZ file."""
    path = resolve_structure_path(test_id)

    with np.load(path, allow_pickle=True) as data:
        available = set(data.files)

        requested_key = str(
            getattr(CFG, "structure_feature_key", "latent_raw")
        )

        candidate_keys = list(
            dict.fromkeys(
                [
                    requested_key,
                    "latent_raw",
                    "raw_features",
                    "z",
                    "z_scores",
                ]
            )
        )
        feature_key = next(
            (key for key in candidate_keys if key in available),
            None,
        )

        if feature_key is None:
            raise KeyError(
                f"{path} contains none of {candidate_keys}. "
                f"Available keys: {sorted(available)}"
            )

        features = np.asarray(data[feature_key], dtype=np.float32)
        subject_ids = (
            np.asarray(data["subject_ids"], dtype=np.int64)
            if "subject_ids" in available
            else None
        )
        if "holdout_subject_ids" in available:
            holdout_ids = np.asarray(
                data["holdout_subject_ids"],
                dtype=np.int64,
            ).reshape(-1)
        elif "test_subject_ids" in available:
            holdout_ids = np.asarray(
                data["test_subject_ids"],
                dtype=np.int64,
            ).reshape(-1)
        elif "test_subject_id" in available:
            holdout_ids = np.asarray(
                data["test_subject_id"],
                dtype=np.int64,
            ).reshape(-1)
        else:
            holdout_ids = np.empty(0, dtype=np.int64)

    if features.ndim != 2:
        raise ValueError(
            "Expected DTOF features [subjects, z_dim], "
            f"got {features.shape}"
        )

    if features.shape[0] != CFG.num_subjects:
        raise ValueError(
            f"Expected {CFG.num_subjects} subjects, "
            f"got {features.shape[0]}"
        )

    features = _reorder_by_subject_ids(features, subject_ids)

    requested_dim = int(CFG.structure_feature_dim)
    if requested_dim <= 0:
        raise ValueError("structure_feature_dim must be positive")

    if features.shape[1] < requested_dim:
        raise ValueError(
            f"Requested {requested_dim} DTOF dimensions, "
            f"but {feature_key} only contains {features.shape[1]}"
        )

    features = features[:, :requested_dim]

    if not np.isfinite(features).all():
        raise ValueError("DTOF feature matrix contains NaN or Inf")

    if holdout_ids.size > 0:
        if test_id not in holdout_ids:
            raise ValueError(
                "Wrong fold-specific DTOF file: "
                f"test subject {test_id} is not in "
                f"holdout_subject_ids={holdout_ids.tolist()}"
            )
    else:
        print(
            "WARNING: DTOF NPZ has no declared holdout subject. "
            "It may be an exploratory full-data representation."
        )

    # Prefer loading z_scores and normalize here with the prediction model's
    # current training subjects. This keeps preprocessing consistent per fold.
    features, _, _ = _standardize_from_training_subjects(
        features,
        train_subjects,
    )

    return features, f"{path}:{feature_key}"


def load_metadata_bank(
    train_subjects: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """Load four handcrafted tissue-thickness features per subject."""
    path = Path(CFG.metadata_feature_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

    frame = pd.read_csv(path, header=0)

    if frame.shape[1] < 9:
        raise ValueError(
            "Metadata CSV must contain at least nine columns because "
            "tissue-thickness features are read from columns 5:9. "
            f"Got shape {frame.shape}."
        )

    metadata = frame.iloc[:, 5:9].to_numpy(dtype=np.float32)

    expected_shape = (CFG.num_subjects, 4)
    if metadata.shape != expected_shape:
        raise ValueError(
            f"Expected metadata shape {expected_shape}, "
            f"got {metadata.shape}"
        )

    if not np.isfinite(metadata).all():
        raise ValueError("Metadata contains NaN or Inf")

    thickness = metadata

    normalized_thickness, _, _ = _minmax_from_training_subjects(
        thickness,
        train_subjects,
    )

    # Do not clip held-out thickness values. Values outside [0, 1] correctly
    # indicate that a held-out subject lies outside the training range.
    features = normalized_thickness.astype(np.float32)

    return features, str(path)


def load_structure_bank(
    test_id: int,
    train_subjects: Sequence[int],
) -> Tuple[np.ndarray, str]:
    """Load and normalize exactly one configured subject-feature source."""
    train_subjects_array = _validate_train_subjects(train_subjects)
    source = str(CFG.subject_feature_source).strip().lower()

    if source == "dtof":
        bank, source_path = load_dtof_bank(
            test_id=test_id,
            train_subjects=train_subjects_array,
        )
    elif source == "metadata":
        bank, source_path = load_metadata_bank(
            train_subjects=train_subjects_array,
        )
    else:
        raise ValueError(
            "subject_feature_source must be 'dtof' or 'metadata', "
            f"got {CFG.subject_feature_source!r}"
        )

    if bank.ndim != 2:
        raise ValueError(
            "Expected subject feature bank [subjects, dim], "
            f"got {bank.shape}"
        )

    if bank.shape[0] != CFG.num_subjects:
        raise ValueError(
            f"Expected {CFG.num_subjects} subjects, "
            f"got {bank.shape[0]}"
        )

    if not np.isfinite(bank).all():
        raise ValueError("Subject feature bank contains NaN or Inf")

    bank, applied_control = _apply_control(
        bank,
        test_id=test_id,
        train_subjects=train_subjects_array,
    )

    print(
        "Loaded subject features\n"
        f"  source: {source}\n"
        f"  path: {source_path}\n"
        f"  shape: {bank.shape}\n"
        f"  control: {applied_control}"
    )

    return bank.astype(np.float32), source_path


def lookup_subject_features(
    subject_ids: np.ndarray,
    feature_bank: np.ndarray,
) -> np.ndarray:
    """Return one complete subject vector for every DRS sample."""
    subject_ids = np.asarray(subject_ids, dtype=np.int64).reshape(-1)

    if feature_bank.ndim != 2:
        raise ValueError(
            "Expected feature_bank [subjects, feature_dim], "
            f"got {feature_bank.shape}"
        )

    if subject_ids.size == 0:
        return np.empty(
            (0, feature_bank.shape[1]),
            dtype=np.float32,
        )

    if subject_ids.min() < 0 or subject_ids.max() >= feature_bank.shape[0]:
        raise ValueError(
            "Sample subject ID is outside the subject feature bank"
        )

    return np.asarray(feature_bank[subject_ids], dtype=np.float32)


# Compatibility for older scripts.
load_latent_bank = load_structure_bank
