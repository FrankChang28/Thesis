"""Evaluate frozen Stage 2 fusion checkpoints under DTOF descriptor controls.

The trained model and held-out DRS rows are fixed.  A control replaces exactly
one subject-level descriptor and broadcasts it to every DRS row of that test
subject.  Repeated controls are reduced to one median result per subject before
group-level inference; repeat rows are retained for sensitivity inspection.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .cache import build_preprocessed_cache, load_cache_split
from .config import CFG, DEVICE
from .engine import predict_ranges
from .loso import build_test_ids, global_fold_pairs, split_subjects
from .model import CNN1D, build_baseline_fusion_model
from .sampling import fixed_subject_ranges
from .utils import ensure_dir, save_csv, write_json


EPS = 1e-8


def repeat_rng(base_seed: int, repeat: int, test_id: int) -> np.random.Generator:
    """Create a fold-order-independent random stream."""
    seed = np.random.SeedSequence([int(base_seed), int(repeat), int(test_id)])
    return np.random.default_rng(seed)


def standardize_descriptor_bank(
    raw_features: np.ndarray,
    train_subjects: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit descriptor z-score statistics on training subjects only."""
    raw = np.asarray(raw_features, dtype=np.float32)
    train = np.asarray(train_subjects, dtype=np.int64)
    mean = raw[train].mean(axis=0)
    std = raw[train].std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return ((raw - mean) / std).astype(np.float32), mean, std


def replace_test_descriptor(
    actual_bank: np.ndarray,
    test_id: int,
    mode: str,
    rng: np.random.Generator,
    train_subjects: list[int] | np.ndarray,
    acquisition_raw: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Return a copied bank with one test-subject replacement."""
    bank = np.asarray(actual_bank, dtype=np.float32).copy()
    metadata: dict[str, int | str] = {"control_mode": mode}
    if mode == "zero":
        bank[int(test_id)] = 0.0
    elif mode == "shuffle":
        donors = np.asarray(train_subjects, dtype=np.int64)
        donor_id = int(rng.choice(donors))
        bank[int(test_id)] = bank[donor_id]
        metadata["donor_subject_id"] = donor_id
    elif mode == "random_acquisition":
        if acquisition_raw is None or mean is None or std is None:
            raise ValueError("random_acquisition requires raw acquisitions and scaler")
        acquisitions = np.asarray(acquisition_raw, dtype=np.float32)
        acquisition_index = int(rng.integers(0, acquisitions.shape[0]))
        bank[int(test_id)] = (acquisitions[acquisition_index] - mean) / std
        metadata["acquisition_index"] = acquisition_index
    else:
        raise ValueError(f"Unknown descriptor control mode: {mode}")
    return bank, metadata


def _load_actual_bank(
    test_id: int,
    train_subjects: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    path = Path(
        CFG.structure_feature_path.format(
            test_id=test_id,
            test_id_1based=test_id + 1,
        )
    )
    with np.load(path, allow_pickle=True) as data:
        raw = np.asarray(data[CFG.structure_feature_key], dtype=np.float32)
        subject_ids = np.asarray(
            data.get("subject_ids", np.arange(len(raw))), dtype=np.int64
        )
    if raw.ndim != 2 or raw.shape[0] != CFG.num_subjects:
        raise ValueError(f"Unexpected descriptor shape in {path}: {raw.shape}")
    order = np.argsort(subject_ids)
    if not np.array_equal(subject_ids[order], np.arange(CFG.num_subjects)):
        raise ValueError(f"Invalid subject IDs in {path}")
    raw = raw[order, : int(CFG.structure_feature_dim)]
    actual, mean, std = standardize_descriptor_bank(raw, train_subjects)
    return actual, mean, std, path


def _load_test_acquisitions(test_id: int) -> tuple[np.ndarray, np.ndarray, Path]:
    path = Path(
        CFG.descriptor_control_acquisition_path.format(
            test_id=test_id,
            test_id_1based=test_id + 1,
        )
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing test acquisition export: {path}. Run "
            "scripts/run_stage1_test_acquisition_exports.sh first."
        )
    with np.load(path, allow_pickle=True) as data:
        raw = np.asarray(data["test_acquisition_latent_raw"], dtype=np.float32)
        positions = np.asarray(data["test_acquisition_op_positions"], dtype=np.int64)
        saved_test_id = int(np.asarray(data["test_subject_id"]).reshape(()))
    if saved_test_id != test_id:
        raise ValueError(f"Acquisition export {path} belongs to test_id={saved_test_id}")
    expected = (2700, int(CFG.structure_feature_dim))
    raw = raw[:, : int(CFG.structure_feature_dim)]
    if raw.shape != expected or positions.shape != (2700,):
        raise ValueError(
            f"Expected 2,700 acquisition embeddings with shape {expected}; "
            f"got raw={raw.shape}, positions={positions.shape}"
        )
    if np.unique(positions).size != 2700:
        raise ValueError(f"Acquisition positions in {path} are not unique")
    return raw, positions, path


def _checkpoint_array(config: dict, name: str, length: int) -> np.ndarray:
    if name not in config:
        raise KeyError(f"Checkpoint config is missing {name!r}")
    value = np.asarray(config[name], dtype=np.float32).reshape(-1)
    if value.size not in {1, length}:
        raise ValueError(f"Unexpected {name} shape: {value.shape}")
    return value


def _load_model_and_stats(
    checkpoint_path: Path,
    fold: int,
    test_id: int,
) -> tuple[torch.nn.Module, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    checks = {
        "test_id": test_id,
        "target_mode": CFG.target_mode,
        "fusion_method": CFG.fusion_method,
        "structure_dim": int(CFG.structure_feature_dim),
    }
    for name, expected in checks.items():
        if name in config and config[name] != expected:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has {name}={config[name]!r}, "
                f"expected {expected!r} for fold {fold}"
            )
    baseline = CNN1D(output_dim=CFG.target_dim, latent_dim=0)
    model = build_baseline_fusion_model(
        CFG.fusion_method,
        baseline=baseline,
        latent_dim=int(CFG.structure_feature_dim),
        output_dim=CFG.target_dim,
        freeze_baseline=CFG.freezes_baseline,
        residual_use_gate=CFG.residual_use_gate,
        residual_gate_init_logit=CFG.residual_gate_init_logit,
    )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.to(DEVICE).eval()
    return (
        model,
        config,
        _checkpoint_array(config, "x_min", CFG.input_dim),
        _checkpoint_array(config, "x_max", CFG.input_dim),
        _checkpoint_array(config, "y_mean", 1),
        _checkpoint_array(config, "y_std", 1),
    )


def _active(metrics: dict[str, float], suffix: str) -> float:
    return float(metrics[f"{CFG.target_name}_{suffix}"])


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _subject_summary(rows: list[dict]) -> list[dict]:
    output = []
    for test_id in sorted({int(row["test_id"]) for row in rows}):
        current = [row for row in rows if int(row["test_id"]) == test_id]
        delta = np.asarray([float(row["delta_RMSE"]) for row in current])
        output.append(
            {
                "target": CFG.target_mode,
                "control_mode": CFG.descriptor_control_mode,
                "fold": int(current[0]["fold"]),
                "test_id": test_id,
                "original_matlab_subject_id": test_id + 1,
                "repeats": len(current),
                "actual_RMSE": float(current[0]["actual_RMSE"]),
                "control_RMSE_median": float(
                    np.median([float(row["control_RMSE"]) for row in current])
                ),
                "delta_RMSE_median": float(np.median(delta)),
                "delta_RMSE_q05": float(np.quantile(delta, 0.05)),
                "delta_RMSE_q95": float(np.quantile(delta, 0.95)),
            }
        )
    return output


def run_descriptor_controls() -> None:
    """Run the configured frozen-checkpoint descriptor control over LOSO folds."""
    CFG.validate()
    ensure_dir(CFG.output_dir)
    fold_dir = Path(CFG.output_dir) / "fold_metrics"
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        str(Path(CFG.output_dir) / "experiment_config.json"),
        {**asdict(CFG), "output_dir": CFG.output_dir},
    )
    build_preprocessed_cache()
    val_x, val_y, val_subj = load_cache_split("val")
    ranges = fixed_subject_ranges(
        val_x.shape[0], CFG.val_rows_per_subject, "val"
    )
    all_subjects = list(range(CFG.num_subjects))
    fold_pairs = global_fold_pairs(all_subjects, build_test_ids(all_subjects))
    all_rows: list[dict] = []
    repeats = 1 if CFG.descriptor_control_mode == "zero" else int(
        CFG.descriptor_control_repeats
    )

    for ordinal, (fold, test_id) in enumerate(fold_pairs, start=1):
        result_path = fold_dir / f"fold_{fold:03d}_test_id_{test_id}.json"
        if CFG.resume and result_path.is_file():
            with result_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            if (
                saved.get("control_mode") == CFG.descriptor_control_mode
                and int(saved.get("repeats", -1)) == repeats
                and int(saved.get("base_seed", -1)) == CFG.descriptor_control_seed
            ):
                all_rows.extend(saved["repeat_metrics"])
                print(f"[{ordinal:03d}/{len(fold_pairs):03d}] fold {fold:03d}: cached")
                continue

        train_subjects, _ = split_subjects(all_subjects, test_id)
        actual_bank, mean, std, descriptor_path = _load_actual_bank(
            test_id, train_subjects
        )
        acquisition_raw = None
        acquisition_positions = None
        acquisition_path = None
        if CFG.descriptor_control_mode == "random_acquisition":
            acquisition_raw, acquisition_positions, acquisition_path = (
                _load_test_acquisitions(test_id)
            )

        checkpoint_path = Path(
            str(CFG.descriptor_control_checkpoint_path).format(
                fold=fold,
                test_id=test_id,
                test_id_1based=test_id + 1,
            )
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing frozen fusion checkpoint: {checkpoint_path}")
        model, checkpoint_config, x_min, x_max, y_mean, y_std = (
            _load_model_and_stats(checkpoint_path, fold, test_id)
        )
        criterion = torch.nn.MSELoss()
        actual_prediction = predict_ranges(
            model,
            val_x,
            val_y,
            val_subj,
            ranges[test_id],
            x_min,
            x_max,
            y_mean,
            y_std,
            criterion,
            latent_bank=actual_bank,
        )
        actual_metrics = actual_prediction["metrics"]
        repeat_rows = []
        for repeat in range(repeats):
            rng = repeat_rng(CFG.descriptor_control_seed, repeat, test_id)
            control_bank, selection = replace_test_descriptor(
                actual_bank,
                test_id,
                CFG.descriptor_control_mode,
                rng,
                train_subjects,
                acquisition_raw=acquisition_raw,
                mean=mean,
                std=std,
            )
            control_prediction = predict_ranges(
                model,
                val_x,
                val_y,
                val_subj,
                ranges[test_id],
                x_min,
                x_max,
                y_mean,
                y_std,
                criterion,
                latent_bank=control_bank,
            )
            control_metrics = control_prediction["metrics"]
            row = {
                "target": CFG.target_mode,
                "control_mode": CFG.descriptor_control_mode,
                "fold": fold,
                "test_id": test_id,
                "original_matlab_subject_id": test_id + 1,
                "repeat": repeat,
                "base_seed": int(CFG.descriptor_control_seed),
                "n_rows": int(len(actual_prediction["y_true"])),
                "actual_RMSE": _active(actual_metrics, "RMSE"),
                "control_RMSE": _active(control_metrics, "RMSE"),
                "delta_RMSE": (
                    _active(control_metrics, "RMSE")
                    - _active(actual_metrics, "RMSE")
                ),
                "actual_Bias": _active(actual_metrics, "Bias"),
                "control_Bias": _active(control_metrics, "Bias"),
                "delta_Bias": (
                    _active(control_metrics, "Bias")
                    - _active(actual_metrics, "Bias")
                ),
                **selection,
            }
            if "acquisition_index" in selection:
                row["op_position"] = int(
                    acquisition_positions[int(selection["acquisition_index"])]
                )
            repeat_rows.append(row)

        payload = {
            "target": CFG.target_mode,
            "control_mode": CFG.descriptor_control_mode,
            "repeats": repeats,
            "base_seed": int(CFG.descriptor_control_seed),
            "fold": fold,
            "test_id": test_id,
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_descriptor": str(descriptor_path.resolve()),
            "source_acquisitions": (
                None if acquisition_path is None else str(acquisition_path.resolve())
            ),
            "checkpoint_best_epoch": checkpoint_config.get("best_epoch"),
            "repeat_metrics": repeat_rows,
        }
        _atomic_json(result_path, payload)
        all_rows.extend(repeat_rows)
        print(
            f"[{ordinal:03d}/{len(fold_pairs):03d}] fold {fold:03d}: "
            f"actual={repeat_rows[0]['actual_RMSE']:.6f} "
            f"control-median={np.median([r['control_RMSE'] for r in repeat_rows]):.6f}"
        )

    all_rows.sort(key=lambda row: (int(row["fold"]), int(row["repeat"])))
    subject_rows = _subject_summary(all_rows)
    save_csv(str(Path(CFG.output_dir) / "per_subject_repeat_metrics.csv"), all_rows)
    save_csv(str(Path(CFG.output_dir) / "per_subject_metrics.csv"), subject_rows)
    delta = np.asarray([row["delta_RMSE_median"] for row in subject_rows])
    write_json(
        str(Path(CFG.output_dir) / "analysis_summary.json"),
        {
            "target": CFG.target_mode,
            "control_mode": CFG.descriptor_control_mode,
            "num_subjects": len(subject_rows),
            "repeats_per_subject": repeats,
            "base_seed": int(CFG.descriptor_control_seed),
            "delta_definition": "control_RMSE - actual_descriptor_RMSE",
            "delta_RMSE_median": float(np.median(delta)),
            "delta_RMSE_q1": float(np.quantile(delta, 0.25)),
            "delta_RMSE_q3": float(np.quantile(delta, 0.75)),
            "worsened_fraction": float(np.mean(delta > 0)),
            "device": str(DEVICE),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
    )
    print(f"complete: {len(subject_rows)} subjects -> {CFG.output_dir}")
