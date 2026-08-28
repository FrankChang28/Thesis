"""Leave-One-Subject-Out orchestration for DRS regression experiments."""

from __future__ import annotations

import csv
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .cache import build_preprocessed_cache, load_cache_split
from .compact_cache import (
    build_or_load_subject_compact_cache_from_mat,
    compute_stats_from_compact,
)
from .config import CFG, DEVICE
from .datasets import MultiCompactNIRSDataset, NIRSDataset
from .engine import (
    evaluate,
    evaluate_subjects,
    make_loader,
    predict_ranges,
    save_predictions_npz,
    train_one_epoch,
)
from .latent import load_structure_bank
from .model import CNN1D, build_baseline_fusion_model
from .sampling import fixed_subject_ranges
from .utils import (
    append_csv_row,
    count_rows,
    ensure_dir,
    save_csv,
    set_seed,
    write_json,
)

def fusion_name() -> str:
    """Human-readable fusion label saved with every fold."""
    if not CFG.use_structure_features:
        return "none"
    fusion = CFG.fusion_method
    if fusion == "residual":
        fusion += "_gated" if CFG.residual_use_gate else "_ungated"
    return f"{CFG.baseline_init}_{CFG.baseline_train}_{fusion}"


def config_to_dict() -> dict:
    """Serialize explicit fields plus important derived experiment values."""
    values = asdict(CFG)
    values.update(
        {
            "experiment_name": CFG.experiment_name,
            "target_name": CFG.target_name,
            "target_index": CFG.target_index,
            "fusion": fusion_name(),
        }
    )
    return values


def split_subjects(
    all_subjects: Sequence[int],
    test_id: int,
) -> Tuple[List[int], List[int]]:
    train_val = np.asarray(
        [subject_id for subject_id in all_subjects if subject_id != test_id],
        dtype=np.int64,
    )

    train_subjects, val_subjects = train_test_split(
        train_val,
        test_size=CFG.val_subject_count,
        random_state=CFG.split_random_state_base + test_id,
        shuffle=True,
    )

    return (
        sorted(map(int, train_subjects)),
        sorted(map(int, val_subjects)),
    )


def build_test_ids(all_subjects: Sequence[int]) -> List[int]:
    if CFG.test_subject_start is not None or CFG.test_subject_end is not None:
        if CFG.test_subject_start is None or CFG.test_subject_end is None:
            raise ValueError(
                "test_subject_start and test_subject_end must both be set"
            )

        start = int(CFG.test_subject_start) - 1
        end = int(CFG.test_subject_end) - 1
        ids = [subject_id for subject_id in all_subjects if start <= subject_id <= end]
    elif CFG.run_all_loso:
        ids = list(all_subjects)
    else:
        ids = [int(CFG.test_subject_id)]

    return ids if CFG.max_folds is None else ids[: CFG.max_folds]


def global_fold_pairs(
    all_subjects: Sequence[int],
    test_ids: Sequence[int],
) -> list[tuple[int, int]]:
    """Pair each test ID with its stable fold number in the full LOSO order."""
    fold_by_test_id = {
        int(subject_id): fold
        for fold, subject_id in enumerate(all_subjects, start=1)
    }
    missing = sorted(set(map(int, test_ids)) - set(fold_by_test_id))
    if missing:
        raise ValueError(f"Unknown test subject IDs: {missing}")
    return [(fold_by_test_id[int(test_id)], int(test_id)) for test_id in test_ids]


def _backup_once(path: str) -> None:
    source = Path(path)
    backup = source.with_name(f"{source.stem}.pre_resume_backup{source.suffix}")
    if source.exists() and not backup.exists():
        shutil.copy2(source, backup)


def completed_test_ids(
    results_path: str,
    fold_by_test_id: dict[int, int],
) -> set[int]:
    """Validate, normalize, and return completed folds from loso_results.csv.

    A fold is complete only when its global fold number is correct, test loss is
    finite, and the recorded best-model file exists. Truncated, duplicate, or
    old mis-numbered rows are removed after a one-time backup.
    """
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        if os.path.exists(results_path):
            _backup_once(results_path)
            os.remove(results_path)
        return set()

    with open(results_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        source_rows = list(reader)

    valid_by_test_id: dict[int, dict[str, str]] = {}
    for row in source_rows:
        try:
            test_id = int(row["test_id"])
            fold = int(row["fold"])
            test_loss = float(row["test_loss"])
            model_path = row["model_path"]
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(test_loss):
            continue
        if fold_by_test_id.get(test_id) != fold:
            continue
        if not model_path or not os.path.isfile(model_path):
            continue
        valid_by_test_id[test_id] = row

    normalized_rows = [
        valid_by_test_id[test_id]
        for test_id in sorted(valid_by_test_id, key=fold_by_test_id.__getitem__)
    ]
    needs_rewrite = len(normalized_rows) != len(source_rows)
    if needs_rewrite:
        _backup_once(results_path)
        if normalized_rows and fieldnames:
            temporary = f"{results_path}.resume_tmp"
            with open(temporary, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(normalized_rows)
            os.replace(temporary, results_path)
        else:
            os.remove(results_path)
        print(
            "Resume audit: retained "
            f"{len(normalized_rows)}/{len(source_rows)} valid result row(s)."
        )
    return set(valid_by_test_id)


def retain_completed_epoch_rows(path: str, completed: set[int]) -> None:
    """Remove epoch-log rows belonging to interrupted folds."""
    if not os.path.exists(path):
        return
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    retained = []
    for row in rows:
        try:
            if int(row["test_id"]) in completed:
                retained.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    if len(retained) == len(rows):
        return
    _backup_once(path)
    if retained and fieldnames:
        temporary = f"{path}.resume_tmp"
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(retained)
        os.replace(temporary, path)
    else:
        os.remove(path)


def active_metric(metrics: Dict[str, float], suffix: str) -> float:
    return float(metrics[f"{CFG.target_name}_{suffix}"])


def print_run_summary(
    test_id: int,
    train_subjects: Sequence[int],
    val_subjects: Sequence[int],
    n_train: int,
    n_val: int,
    n_test: int,
    structure_path: str | None,
    structure_dim: int,
) -> None:
    print("\nRun summary")
    print(
        f"  target: {CFG.target_name} "
        f"(source column {CFG.target_index})"
    )
    print(f"  mode: {CFG.experiment_mode}")
    print(f"  test: {test_id} (MATLAB SubjID {test_id + 1})")
    print(
        f"  subjects: train={len(train_subjects)}, "
        f"val={len(val_subjects)}, test=1"
    )
    print(f"  rows: train={n_train}, val={n_val}, test={n_test}")
    print(f"  compact stages: {CFG.compact_stage_count}")
    print(f"  structure: {structure_path or 'none'}")
    if CFG.use_structure_features:
        print(
            f"  structure dim/control: {structure_dim}/{CFG.structure_control}"
        )
        print(
            f"  baseline: init={CFG.baseline_init}, "
            f"train={CFG.baseline_train}"
        )
        print(f"  fusion: {CFG.fusion_method}")
    print(
        f"  learning_rate={CFG.learning_rate}, "
        f"baseline_learning_rate={CFG.baseline_optimizer_lr}, "
        f"epochs={CFG.epochs}, batch_size={CFG.batch_size}"
    )


def print_epoch_header() -> None:
    target = CFG.target_mode
    print("\n" + "-" * 118)
    print(
        f"{'ep':>3} {'train_loss':>12} {'val_loss':>12} "
        f"{f'{target}_mae':>12} {f'{target}_rmse':>12} "
        f"{f'{target}_bias':>12} {'main_lr':>10} {'base_lr':>10} "
        f"{'sec':>8} {'best':>6}"
    )
    print("-" * 118)


def print_epoch_row(
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: Dict[str, float],
    primary_learning_rate: float,
    baseline_learning_rate: float | None,
    seconds: float,
    improved: bool,
) -> None:
    print(
        f"{epoch:3d} {train_loss:12.6e} {val_loss:12.6e} "
        f"{active_metric(metrics, 'MAE'):12.6f} "
        f"{active_metric(metrics, 'RMSE'):12.6f} "
        f"{active_metric(metrics, 'Bias'):12.6f} "
        f"{primary_learning_rate:10.2e} "
        f"{(baseline_learning_rate if baseline_learning_rate is not None else float('nan')):10.2e} "
        f"{seconds:8.1f} "
        f"{('*' if improved else ''):>6}"
    )


def resolve_baseline_checkpoint(fold_idx: int, test_id: int) -> str:
    """Resolve the configured per-fold baseline checkpoint template."""
    if not CFG.baseline_checkpoint_path:
        raise ValueError("baseline_checkpoint_path is not configured")
    path = Path(
        CFG.baseline_checkpoint_path.format(
            fold=fold_idx,
            test_id=test_id,
            test_id_1based=test_id + 1,
        )
    ).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {path}")
    return str(path)


def build_model_and_optimizer(
    structure_dim: int,
    fold_idx: int,
    test_id: int,
):
    """Build one fold's model, optimizer, and resolved checkpoint path."""
    if structure_dim == 0:
        model = CNN1D(output_dim=CFG.target_dim, latent_dim=0).to(DEVICE)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.parameters(),
                    "lr": CFG.learning_rate,
                    "name": "model",
                }
            ],
            weight_decay=CFG.weight_decay,
        )
        print("Training a DRS-only baseline from scratch.")
        _print_trainable_parameters(model)
        return model, optimizer, None

    baseline = CNN1D(output_dim=CFG.target_dim, latent_dim=0).to(DEVICE)
    baseline_path = None
    if CFG.uses_pretrained_baseline:
        baseline_path = resolve_baseline_checkpoint(fold_idx, test_id)
        checkpoint = torch.load(
            baseline_path,
            map_location=DEVICE,
            weights_only=False,
        )
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_test_id = checkpoint_config.get("test_id")
        if checkpoint_test_id is not None and int(checkpoint_test_id) != test_id:
            raise ValueError(
                "Baseline checkpoint test_id does not match this LOSO fold: "
                f"checkpoint={checkpoint_test_id}, current={test_id}, "
                f"path={baseline_path}"
            )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        baseline.load_state_dict(state_dict, strict=True)

    model = build_baseline_fusion_model(
        CFG.fusion_method,
        baseline=baseline,
        latent_dim=structure_dim,
        output_dim=CFG.target_dim,
        freeze_baseline=CFG.freezes_baseline,
        residual_use_gate=CFG.residual_use_gate,
        residual_gate_init_logit=CFG.residual_gate_init_logit,
    ).to(DEVICE)

    fusion_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("baseline.") and parameter.requires_grad
    ]
    parameter_groups = [
        {
            "params": fusion_parameters,
            "lr": CFG.learning_rate,
            "name": "fusion",
        }
    ]
    if not CFG.freezes_baseline:
        parameter_groups.append(
            {
                "params": model.baseline.parameters(),
                "lr": CFG.baseline_optimizer_lr,
                "name": "baseline",
            }
        )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=CFG.weight_decay,
    )

    print(
        "Training subject-feature fusion\n"
        f"  baseline init: {CFG.baseline_init}\n"
        f"  baseline train: {CFG.baseline_train}\n"
        f"  fusion method: {CFG.fusion_method}"
    )
    if baseline_path is not None:
        print(f"  baseline checkpoint: {baseline_path}")
    if not CFG.freezes_baseline:
        print(f"  baseline learning rate: {CFG.baseline_optimizer_lr}")
    if CFG.fusion_method == "residual":
        print(f"  residual gate: {'on' if CFG.residual_use_gate else 'off'}")

    _print_trainable_parameters(model)
    return model, optimizer, baseline_path


def _print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"  trainable parameters: {trainable:,}/{total:,}")


def current_learning_rates(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def build_compact_infos(
    mat_file: h5py.File,
    train_subjects: Sequence[int],
):
    infos_by_stage = []

    for stage_idx in range(CFG.compact_stage_count):
        stage_infos = [
            build_or_load_subject_compact_cache_from_mat(
                mat_file,
                subject_id,
                stage_idx,
            )
            for subject_id in train_subjects
        ]
        infos_by_stage.append(stage_infos)

    return infos_by_stage


def run_loso() -> None:
    CFG.validate()
    use_structure = CFG.use_structure_features

    set_seed(CFG.seed)
    ensure_dir(CFG.output_dir)
    ensure_dir(os.path.join(CFG.output_dir, "predictions"))
    ensure_dir(os.path.join(CFG.output_dir, "configs"))

    write_json(
        os.path.join(CFG.output_dir, "experiment_config.json"),
        config_to_dict(),
    )

    build_preprocessed_cache()
    val_x, val_y, val_subj = load_cache_split("val")
    val_ranges_by_subj = fixed_subject_ranges(
        val_x.shape[0],
        CFG.val_rows_per_subject,
        "val",
    )

    all_subjects = list(range(CFG.num_subjects))
    test_ids = build_test_ids(all_subjects)
    fold_pairs = global_fold_pairs(all_subjects, test_ids)
    fold_by_test_id = {
        test_id: fold_idx
        for fold_idx, test_id in global_fold_pairs(all_subjects, all_subjects)
    }

    results_path = os.path.join(CFG.output_dir, "loso_results.csv")
    all_epoch_log_path = os.path.join(CFG.output_dir, "all_epoch_logs.csv")
    if CFG.resume:
        completed = completed_test_ids(results_path, fold_by_test_id)
        if CFG.save_epoch_log:
            retain_completed_epoch_rows(all_epoch_log_path, completed)
        selected_completed = completed.intersection(test_ids)
        print(
            "Resume enabled: "
            f"{len(selected_completed)}/{len(test_ids)} selected fold(s) complete."
        )
    else:
        completed = set()
        if os.path.exists(results_path):
            os.remove(results_path)
        if CFG.save_epoch_log and os.path.exists(all_epoch_log_path):
            os.remove(all_epoch_log_path)

    with h5py.File(CFG.mat_file, "r") as mat_file:
        for fold_idx, test_id in fold_pairs:
            if test_id in completed:
                print(
                    f"Skipping fold {fold_idx:03d}, test_id={test_id}: "
                    "valid result row and model checkpoint already exist."
                )
                continue
            set_seed(CFG.seed + int(test_id))
            train_subjects, val_subjects = split_subjects(all_subjects, test_id)

            val_ranges = [
                subject_range
                for subject_id in val_subjects
                for subject_range in val_ranges_by_subj[subject_id]
            ]
            test_ranges = val_ranges_by_subj[test_id]

            structure_bank = None
            structure_path = None
            structure_dim = 0

            if use_structure:
                structure_bank, structure_path = load_structure_bank(
                    test_id=test_id,
                    train_subjects=train_subjects,
                )
                structure_dim = int(structure_bank.shape[1])

            compact_infos_by_stage = build_compact_infos(
                mat_file,
                train_subjects,
            )
            compact_infos = [
                info
                for stage_infos in compact_infos_by_stage
                for info in stage_infos
            ]

            n_train = sum(int(info["n_rows"]) for info in compact_infos)
            n_val = count_rows(val_ranges)
            n_test = count_rows(test_ranges)

            print_run_summary(
                test_id,
                train_subjects,
                val_subjects,
                n_train,
                n_val,
                n_test,
                structure_path,
                structure_dim,
            )

            x_min, x_max, y_mean, y_std = compute_stats_from_compact(
                compact_infos
            )
            print(f"  y_mean={y_mean}, y_std={y_std}")

            train_loaders = [
                make_loader(
                    MultiCompactNIRSDataset(
                        stage_infos,
                        x_min,
                        x_max,
                        y_mean,
                        y_std,
                        latent_bank=structure_bank,
                        shuffle=True,
                        drop_last=True,
                    )
                )
                for stage_infos in compact_infos_by_stage
            ]

            val_loader = make_loader(
                NIRSDataset(
                    val_x,
                    val_y,
                    val_subj,
                    val_ranges,
                    x_min,
                    x_max,
                    y_mean,
                    y_std,
                    latent_bank=structure_bank,
                    shuffle=False,
                )
            )

            model, optimizer, baseline_model_path = build_model_and_optimizer(
                structure_dim,
                fold_idx,
                test_id,
            )
            criterion = torch.nn.MSELoss()
            scheduler_min_lrs = [
                (
                    CFG.baseline_optimizer_min_lr
                    if group.get("name") == "baseline"
                    else CFG.min_learning_rate
                )
                for group in optimizer.param_groups
            ]
            scheduler_min_lr = (
                scheduler_min_lrs[0]
                if len(scheduler_min_lrs) == 1
                else scheduler_min_lrs
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=CFG.scheduler_factor,
                patience=CFG.scheduler_patience,
                min_lr=scheduler_min_lr,
            )

            model_path = os.path.join(
                CFG.output_dir,
                f"best_model_fold_{fold_idx:03d}_test_id_{test_id}.pth",
            )
            epoch_log_path = os.path.join(
                CFG.output_dir,
                f"epoch_log_fold_{fold_idx:03d}_test_id_{test_id}.csv",
            )
            if CFG.save_epoch_log and os.path.exists(epoch_log_path):
                os.remove(epoch_log_path)

            fold_config = config_to_dict()
            fold_config.update(
                {
                    "fold": fold_idx,
                    "test_id": test_id,
                    "original_matlab_subj_id": test_id + 1,
                    "train_subjects": train_subjects,
                    "val_subjects": val_subjects,
                    "structure_path": structure_path,
                    "structure_dim": structure_dim,
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "fusion": fusion_name(),
                    "baseline_model_path": baseline_model_path,
                    "train_compact_cache_dirs": [
                        info["dir"] for info in compact_infos
                    ],
                }
            )

            fold_config_path = os.path.join(
                CFG.output_dir,
                "configs",
                f"config_fold_{fold_idx:03d}_test_id_{test_id}.json",
            )
            write_json(fold_config_path, fold_config)

            best_loss = float("inf")
            best_epoch = 0
            patience = 0

            print_epoch_header()
            for epoch in range(1, CFG.epochs + 1):
                start_time = time.time()

                stage_losses = [
                    train_one_epoch(
                        model,
                        loader,
                        optimizer,
                        criterion,
                    )
                    for loader in train_loaders
                ]
                train_loss = float(np.mean(stage_losses))

                val_loss, val_metrics = evaluate(
                    model,
                    val_loader,
                    criterion,
                    y_mean,
                    y_std,
                )

                scheduler.step(val_loss)
                learning_rates = current_learning_rates(optimizer)
                primary_learning_rate = (
                    learning_rates["fusion"]
                    if "fusion" in learning_rates
                    else learning_rates["model"]
                )
                baseline_learning_rate = learning_rates.get("baseline")
                improved = val_loss < best_loss

                if improved:
                    best_loss = val_loss
                    best_epoch = epoch
                    patience = 0
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": fold_config,
                        },
                        model_path,
                    )
                else:
                    patience += 1

                epoch_seconds = time.time() - start_time
                print_epoch_row(
                    epoch,
                    train_loss,
                    val_loss,
                    val_metrics,
                    primary_learning_rate,
                    baseline_learning_rate,
                    epoch_seconds,
                    improved,
                )

                if CFG.save_epoch_log:
                    row = {
                        "fold": fold_idx,
                        "test_id": test_id,
                        "experiment_mode": CFG.experiment_mode,
                        "structure_control": (
                            CFG.structure_control if use_structure else "none"
                        ),
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val_loss": best_loss,
                        "best_epoch": best_epoch,
                        "learning_rate": primary_learning_rate,
                        "primary_learning_rate": primary_learning_rate,
                        "baseline_learning_rate": baseline_learning_rate,
                        "epoch_seconds": epoch_seconds,
                        "improved": int(improved),
                        "patience": patience,
                        **{
                            f"val_{key}": value
                            for key, value in val_metrics.items()
                        },
                    }
                    append_csv_row(epoch_log_path, row)
                    append_csv_row(all_epoch_log_path, row)

                if patience >= CFG.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            checkpoint = torch.load(
                model_path,
                map_location=DEVICE,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            val_prediction = predict_ranges(
                model,
                val_x,
                val_y,
                val_subj,
                val_ranges,
                x_min,
                x_max,
                y_mean,
                y_std,
                criterion,
                latent_bank=structure_bank,
                save_x=CFG.save_prediction_x,
            )
            test_prediction = predict_ranges(
                model,
                val_x,
                val_y,
                val_subj,
                test_ranges,
                x_min,
                x_max,
                y_mean,
                y_std,
                criterion,
                latent_bank=structure_bank,
                save_x=CFG.save_prediction_x,
            )

            val_metrics = val_prediction["metrics"]
            test_metrics = test_prediction["metrics"]

            val_prediction_path = None
            test_prediction_path = None
            if CFG.save_predictions:
                val_prediction_path = os.path.join(
                    CFG.output_dir,
                    "predictions",
                    f"fold_{fold_idx:03d}_test_id_{test_id}_val_predictions.npz",
                )
                test_prediction_path = os.path.join(
                    CFG.output_dir,
                    "predictions",
                    f"fold_{fold_idx:03d}_test_id_{test_id}_test_predictions.npz",
                )
                save_predictions_npz(
                    val_prediction_path,
                    "val",
                    fold_idx,
                    test_id,
                    val_prediction,
                )
                save_predictions_npz(
                    test_prediction_path,
                    "test",
                    fold_idx,
                    test_id,
                    test_prediction,
                )

            per_subject_eval_path = None
            if CFG.save_per_subject_eval:
                subject_rows = evaluate_subjects(
                    model,
                    val_x,
                    val_y,
                    val_subj,
                    val_ranges_by_subj,
                    val_subjects,
                    x_min,
                    x_max,
                    y_mean,
                    y_std,
                    criterion,
                    structure_bank,
                    "val_full",
                )
                subject_rows.extend(
                    evaluate_subjects(
                        model,
                        val_x,
                        val_y,
                        val_subj,
                        val_ranges_by_subj,
                        [test_id],
                        x_min,
                        x_max,
                        y_mean,
                        y_std,
                        criterion,
                        structure_bank,
                        "test_full",
                    )
                )

                per_subject_eval_path = os.path.join(
                    CFG.output_dir,
                    f"per_subject_eval_fold_{fold_idx:03d}_test_id_{test_id}.csv",
                )
                save_csv(per_subject_eval_path, subject_rows)

            result = {
                "fold": fold_idx,
                "test_id": test_id,
                "original_matlab_subj_id": test_id + 1,
                "experiment_mode": CFG.experiment_mode,
                "structure_control": (
                    CFG.structure_control if use_structure else "none"
                ),
                "fusion": fusion_name(),
                "baseline_model_path": baseline_model_path,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "test_loss": test_prediction["loss"],
                "n_train_all_stages": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "model_path": model_path,
                "epoch_log_path": epoch_log_path,
                "fold_config_path": fold_config_path,
                "val_prediction_path": val_prediction_path,
                "test_prediction_path": test_prediction_path,
                "per_subject_eval_path": per_subject_eval_path,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            append_csv_row(results_path, result)

            print("\nBest result")
            print(
                f"  best_epoch={best_epoch}, best_val_loss={best_loss:.6e}"
            )
            print(
                f"  val: MAE={active_metric(val_metrics, 'MAE'):.6f}, "
                f"RMSE={active_metric(val_metrics, 'RMSE'):.6f}, "
                f"Bias={active_metric(val_metrics, 'Bias'):.6f}"
            )
            print(
                f"  test: MAE={active_metric(test_metrics, 'MAE'):.6f}, "
                f"RMSE={active_metric(test_metrics, 'RMSE'):.6f}, "
                f"Bias={active_metric(test_metrics, 'Bias'):.6f}"
            )

    print("\nTraining finished.")
    print(f"Results saved to {results_path}")
