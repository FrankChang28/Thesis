#!/usr/bin/env python3
"""Extract one raw DTOF descriptor per subject for Stage 2 prediction.

The generated NPZ is compatible with ``load_dtof_bank()`` when:

    CFG.structure_feature_key = "latent_raw"

For each subject, this version selects the observed acquisition embedding
nearest the spherical center of that subject's DTOF representations.  The
saved ``latent_raw`` has not been z-score normalized. Stage 2
should perform fold-specific normalization using only the training subjects
of each prediction fold.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .data import load_data_bundle
from .metrics import collect_embeddings
from .model import DTOFSubjectEncoder
from .training import resolve_device
from .utils import (
    set_seed,
    select_nearest_to_spherical_center,
    standardize_from_train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one raw nearest-to-center DTOF descriptor per subject "
            "for Stage 2 prediction"
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument(
        "--max_samples_per_subject",
        type=int,
        default=0,
        help="0 uses all OP positions",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--amp", action="store_true")

    parser.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Optional CSV containing the raw subject-level latent vectors",
    )
    parser.add_argument(
        "--save_reference_zscore",
        action="store_true",
        help=(
            "Also save a checkpoint-training-set z-scored copy for diagnostics. "
            "Abs_prediction should still use latent_raw and perform its own "
            "fold-specific normalization."
        ),
    )

    return parser


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert checkpoint args into a plain dictionary."""
    if isinstance(value, Mapping):
        return dict(value)

    if hasattr(value, "__dict__"):
        return dict(vars(value))

    raise TypeError(
        "checkpoint['args'] must be a mapping or argparse-like namespace, "
        f"got {type(value).__name__}"
    )


def _subject_array(
    checkpoint: Mapping[str, Any],
    key: str,
) -> np.ndarray:
    """Read a checkpoint subject-ID array safely."""
    values = checkpoint.get(key, [])

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    return np.asarray(values, dtype=np.int64).reshape(-1)


def _validate_subject_partition(
    num_subjects: int,
    train_subjects: np.ndarray,
    val_subjects: np.ndarray,
    test_subjects: np.ndarray,
) -> None:
    """Validate checkpoint train/validation/test subject IDs."""
    for name, values in (
        ("train_subjects", train_subjects),
        ("val_subjects", val_subjects),
        ("test_subjects", test_subjects),
    ):
        if values.size:
            if values.min() < 0 or values.max() >= num_subjects:
                raise ValueError(
                    f"{name} contains an out-of-range subject ID"
                )

        if np.unique(values).size != values.size:
            raise ValueError(f"{name} contains duplicate subject IDs")

    if np.intersect1d(train_subjects, val_subjects).size:
        raise ValueError("train_subjects and val_subjects overlap")

    if np.intersect1d(train_subjects, test_subjects).size:
        raise ValueError("train_subjects and test_subjects overlap")

    if np.intersect1d(val_subjects, test_subjects).size:
        raise ValueError("val_subjects and test_subjects overlap")


def _validate_and_sort_features(
    subject_ids: np.ndarray,
    latent_raw: np.ndarray,
    num_subjects: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and sort subject-level features by subject ID."""
    subject_ids = np.asarray(
        subject_ids,
        dtype=np.int64,
    ).reshape(-1)

    latent_raw = np.asarray(
        latent_raw,
        dtype=np.float32,
    )

    if latent_raw.ndim != 2:
        raise ValueError(
            "Expected latent_raw with shape [subjects, latent_dim], "
            f"got {latent_raw.shape}"
        )

    if latent_raw.shape[0] != subject_ids.shape[0]:
        raise ValueError(
            "Subject ID count does not match latent feature count"
        )

    if not np.isfinite(latent_raw).all():
        raise ValueError("latent_raw contains NaN or Inf")

    if np.unique(subject_ids).size != subject_ids.size:
        raise ValueError("Subject-level output contains duplicate subject IDs")

    order = np.argsort(subject_ids)
    subject_ids = subject_ids[order]
    latent_raw = latent_raw[order]

    expected_subject_ids = np.arange(
        num_subjects,
        dtype=np.int64,
    )

    if not np.array_equal(subject_ids, expected_subject_ids):
        missing = np.setdiff1d(
            expected_subject_ids,
            subject_ids,
        )
        extra = np.setdiff1d(
            subject_ids,
            expected_subject_ids,
        )

        raise ValueError(
            "Subject-level features do not contain every expected subject. "
            f"Missing={missing.tolist()}, extra={extra.tolist()}"
        )

    return subject_ids, latent_raw


def main(args: argparse.Namespace) -> None:
    checkpoint_path = Path(
        args.checkpoint
    ).expanduser().resolve()

    save_path = Path(
        args.save_path
    ).expanduser().resolve()

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must contain a mapping, "
            f"got {type(checkpoint).__name__}"
        )

    required_keys = {
        "args",
        "model_config",
        "model_state_dict",
    }

    missing_keys = sorted(
        required_keys.difference(checkpoint)
    )

    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing required keys: {missing_keys}"
        )

    train_args = _as_dict(checkpoint["args"])
    train_args["data_dir"] = args.data_dir

    namespace = argparse.Namespace(**train_args)

    seed = int(train_args.get("seed", 42))
    set_seed(seed)

    bundle = load_data_bundle(namespace)

    model = DTOFSubjectEncoder(
        **checkpoint["model_config"]
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    device = resolve_device(args.device)
    model.to(device).eval()

    if args.max_samples_per_subject > 0:
        rng = np.random.default_rng(seed + 999)

        sample_count = min(
            args.max_samples_per_subject,
            bundle.op_per_subject,
        )

        positions = np.sort(
            rng.choice(
                bundle.op_per_subject,
                size=sample_count,
                replace=False,
            )
        ).astype(np.int64)
    else:
        positions = np.arange(
            bundle.op_per_subject,
            dtype=np.int64,
        )

    loader = bundle.make_subject_loader(
        subjects=np.arange(
            bundle.num_subjects,
            dtype=np.int64,
        ),
        op_positions=positions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    payload = collect_embeddings(
        model,
        loader,
        device,
        use_amp=args.amp,
    )

    if "z" not in payload or "subject_ids" not in payload:
        raise KeyError(
            "collect_embeddings() must return 'z' and 'subject_ids'"
        )

    subject_ids, latent_raw = select_nearest_to_spherical_center(
        payload["z"],
        payload["subject_ids"],
    )

    subject_ids, latent_raw = _validate_and_sort_features(
        subject_ids=subject_ids,
        latent_raw=latent_raw,
        num_subjects=int(bundle.num_subjects),
    )

    train_subjects = _subject_array(
        checkpoint,
        "train_subjects",
    )
    val_subjects = _subject_array(
        checkpoint,
        "val_subjects",
    )
    test_subjects = _subject_array(
        checkpoint,
        "test_subjects",
    )

    _validate_subject_partition(
        num_subjects=int(bundle.num_subjects),
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )

    holdout_subject_ids = np.unique(
        np.concatenate(
            [
                val_subjects,
                test_subjects,
            ]
        )
    ).astype(np.int64)

    splits = np.full(
        subject_ids.shape[0],
        "unknown",
        dtype="U8",
    )

    splits[
        np.isin(subject_ids, train_subjects)
    ] = "train"

    splits[
        np.isin(subject_ids, val_subjects)
    ] = "val"

    splits[
        np.isin(subject_ids, test_subjects)
    ] = "test"

    save_payload: dict[str, np.ndarray] = {
        # Main feature consumed by Abs_prediction.
        # "Raw" means it has not received checkpoint-train z-score
        # normalization. It is an observed acquisition embedding selected by
        # proximity to the subject's spherical center.
        "latent_raw": latent_raw,

        # Canonical alias for new consumers. ``latent_raw`` remains for
        # compatibility with existing Stage 2 configurations.
        "dtof_descriptor_raw": latent_raw,

        # Alias retained for compatibility and easier inspection.
        "raw_features": latent_raw,

        "subject_ids": subject_ids,
        "matlab_subject_ids": subject_ids + 1,

        "train_subject_ids": train_subjects,
        "val_subject_ids": val_subjects,
        "test_subject_ids": test_subjects,

        # Used by load_dtof_bank() for fold-file validation.
        "holdout_subject_ids": holdout_subject_ids,

        "splits": splits,
        "op_positions": positions,

        "aggregation": np.asarray("nearest_to_spherical_center"),
        "feature_key": np.asarray("latent_raw"),
        "source_checkpoint": np.asarray(
            str(checkpoint_path)
        ),
    }

    if args.save_reference_zscore:
        if train_subjects.size == 0:
            raise ValueError(
                "Cannot compute reference z-score because "
                "train_subjects is empty"
            )

        train_mask = np.isin(
            subject_ids,
            train_subjects,
        )

        (
            latent_reference_zscore,
            normalization_mean,
            normalization_std,
        ) = standardize_from_train(
            latent_raw,
            train_mask,
        )

        save_payload.update(
            {
                "latent_reference_zscore": np.asarray(
                    latent_reference_zscore,
                    dtype=np.float32,
                ),
                "reference_normalization_mean": np.asarray(
                    normalization_mean,
                    dtype=np.float32,
                ),
                "reference_normalization_std": np.asarray(
                    normalization_std,
                    dtype=np.float32,
                ),
            }
        )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        save_path,
        **save_payload,
    )

    if args.save_csv:
        csv_path = Path(
            args.save_csv
        ).expanduser().resolve()

        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)

            writer.writerow(
                [
                    "subject_id",
                    "matlab_subject_id",
                    "split",
                ]
                + [
                    f"latent_raw_{i}"
                    for i in range(latent_raw.shape[1])
                ]
            )

            for subject_id, split_name, row in zip(
                subject_ids,
                splits,
                latent_raw,
            ):
                writer.writerow(
                    [
                        int(subject_id),
                        int(subject_id) + 1,
                        split_name,
                    ]
                    + row.astype(float).tolist()
                )

    print("Saved Abs_prediction-compatible subject features")
    print(f"  path: {save_path}")
    print(f"  latent_raw shape: {latent_raw.shape}")
    print("  aggregation: nearest_to_spherical_center")
    print(
        f"  subject IDs: "
        f"{subject_ids[0]}..{subject_ids[-1]}"
    )
    print(
        f"  train/val/test: "
        f"{len(train_subjects)}/"
        f"{len(val_subjects)}/"
        f"{len(test_subjects)}"
    )
    print(
        f"  holdout_subject_ids: "
        f"{holdout_subject_ids.tolist()}"
    )
    print(
        "  normalization: none in latent_raw; "
        "Abs_prediction performs fold-specific z-score "
        "using its own training subjects"
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
