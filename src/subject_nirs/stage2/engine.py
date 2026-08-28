"""Training, evaluation, prediction, and result-saving routines."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import select_target_columns
from .config import CFG, DEVICE
from .datasets import NIRSDataset, normalize_arrays
from .latent import lookup_subject_features
from .metrics import compute_metrics
from .sampling import ranges_to_batches
from .utils import count_rows, ensure_dir


EPS = 1e-8


def make_loader(dataset) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def unpack_batch(batch):
    if len(batch) == 2:
        x, y = batch
        z = None
    elif len(batch) == 3:
        x, z, y = batch
    else:
        raise RuntimeError(f"Unexpected batch structure with {len(batch)} elements")

    x = x.to(DEVICE, non_blocking=False)
    y = y.to(DEVICE, non_blocking=False)
    z = None if z is None else z.to(DEVICE, non_blocking=False)
    return x, z, y


def _assert_finite(name: str, tensor: torch.Tensor, step: int) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite {name} at step {step}")


def train_one_epoch(model, loader, optimizer, criterion) -> float:
    """Train one pass over an iterable loader and return mean sample loss."""
    model.train()
    total_loss = 0.0
    total_n = 0

    for step, batch in enumerate(loader):
        x, z, y = unpack_batch(batch)
        _assert_finite("x", x, step)
        _assert_finite("y", y, step)
        if z is not None:
            _assert_finite("z", z, step)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x, z)
        _assert_finite("prediction", pred, step)

        loss = criterion(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=CFG.gradient_clip_norm,
        )
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_n += batch_size

    if total_n == 0:
        raise RuntimeError("Training loader produced no samples")
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, y_mean, y_std):
    """Evaluate normalized loss and metrics in original target units."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    predictions = []
    targets = []

    for batch in loader:
        x, z, y = unpack_batch(batch)
        pred = model(x, z)
        loss = criterion(pred, y)

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_n += batch_size

        predictions.append(pred.cpu().numpy() * (y_std + EPS) + y_mean)
        targets.append(y.cpu().numpy() * (y_std + EPS) + y_mean)

    if total_n == 0:
        raise RuntimeError("Evaluation loader produced no samples")

    pred_all = np.concatenate(predictions, axis=0)
    true_all = np.concatenate(targets, axis=0)
    return total_loss / total_n, compute_metrics(true_all, pred_all)


@torch.no_grad()
def predict_ranges(
    model,
    x_arr,
    y_arr,
    subj_arr,
    ranges,
    x_min,
    x_max,
    y_mean,
    y_std,
    criterion,
    latent_bank=None,
    save_x: bool = False,
):
    """Evaluate row ranges and return sample-level predictions."""
    model.eval()
    total_loss = 0.0
    total_n = 0

    y_true_list = []
    y_pred_list = []
    subject_list = []
    row_list = []
    x_list = [] if save_x else None

    for rows in ranges_to_batches(
        ranges,
        CFG.batch_size,
        shuffle=False,
        drop_last=False,
    ):
        x_np = np.asarray(x_arr[rows], dtype=np.float32)
        y_all = np.asarray(y_arr[rows], dtype=np.float32)
        y_np = select_target_columns(y_all)
        subject_np = np.asarray(subj_arr[rows], dtype=np.int64)

        x_norm, y_norm = normalize_arrays(
            x_np,
            y_np,
            x_min,
            x_max,
            y_mean,
            y_std,
        )

        x_t = torch.from_numpy(x_norm.astype(np.float32)).to(DEVICE)
        y_t = torch.from_numpy(y_norm.astype(np.float32)).to(DEVICE)

        z_t = None
        if latent_bank is not None:
            z_np = lookup_subject_features(subject_np, latent_bank)
            z_t = torch.from_numpy(z_np).to(DEVICE)

        pred_t = model(x_t, z_t)
        loss = criterion(pred_t, y_t)

        batch_size = x_t.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_n += batch_size

        pred_np = pred_t.cpu().numpy() * (y_std + EPS) + y_mean
        true_np = y_t.cpu().numpy() * (y_std + EPS) + y_mean

        y_pred_list.append(pred_np.astype(np.float32))
        y_true_list.append(true_np.astype(np.float32))
        subject_list.append(subject_np)
        row_list.append(np.asarray(rows, dtype=np.int64))

        if save_x:
            assert x_list is not None
            x_list.append(x_np)

    if total_n == 0:
        raise RuntimeError("Prediction ranges produced no samples")

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    subject_id = np.concatenate(subject_list, axis=0)
    row_index = np.concatenate(row_list, axis=0)
    error = y_pred - y_true

    result = {
        "loss": total_loss / total_n,
        "metrics": compute_metrics(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "error": error.astype(np.float32),
        "abs_error": np.abs(error).astype(np.float32),
        "subject_id": subject_id,
        "row_index": row_index,
    }

    if save_x:
        assert x_list is not None
        result["x"] = np.concatenate(x_list, axis=0).astype(np.float32)

    return result


def save_predictions_npz(path, split, fold_idx, test_id, pred_result) -> None:
    ensure_dir(os.path.dirname(path))

    payload = {
        "split": np.array(split),
        "fold": np.array(fold_idx, dtype=np.int64),
        "test_id": np.array(test_id, dtype=np.int64),
        "original_matlab_subj_id": np.array(test_id + 1, dtype=np.int64),
        "target_mode": np.array(CFG.target_mode),
        "target_names": np.array(CFG.target_names),
        "source_target_index": np.array(CFG.target_index, dtype=np.int64),
        "prediction_target_index": np.array(0, dtype=np.int64),
        "cache_target_names": np.array(CFG.cache_target_names),
        "loss": np.array(pred_result["loss"], dtype=np.float32),
        "y_true": pred_result["y_true"].astype(np.float32),
        "y_pred": pred_result["y_pred"].astype(np.float32),
        "error": pred_result["error"].astype(np.float32),
        "abs_error": pred_result["abs_error"].astype(np.float32),
        "subject_id": pred_result["subject_id"].astype(np.int64),
        "row_index": pred_result["row_index"].astype(np.int64),
    }

    for key, value in pred_result["metrics"].items():
        payload[key] = np.array(value, dtype=np.float32)

    if "x" in pred_result:
        payload["x"] = pred_result["x"].astype(np.float32)

    np.savez_compressed(path, **payload)


def evaluate_subjects(
    model,
    val_x,
    val_y,
    val_subj,
    val_ranges_by_subj,
    subject_ids,
    x_min,
    x_max,
    y_mean,
    y_std,
    criterion,
    latent_bank,
    prefix,
):
    rows = []

    for subject_id in subject_ids:
        subject_id = int(subject_id)
        dataset = NIRSDataset(
            val_x,
            val_y,
            val_subj,
            val_ranges_by_subj[subject_id],
            x_min,
            x_max,
            y_mean,
            y_std,
            latent_bank=latent_bank,
            shuffle=False,
        )

        loss, metrics = evaluate(
            model,
            make_loader(dataset),
            criterion,
            y_mean,
            y_std,
        )

        row = {
            "split": prefix,
            "subject_id_0based": subject_id,
            "original_matlab_subj_id": subject_id + 1,
            "loss": loss,
            "n_rows": count_rows(val_ranges_by_subj[subject_id]),
        }
        row.update(metrics)
        rows.append(row)

    return rows
