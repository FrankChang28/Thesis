from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import subject_group_loss

_MIN_DELTA = 1e-4


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _run_group_epoch(
    model,
    loader,
    device: torch.device,
    args,
    optimizer=None,
    scaler=None,
) -> Dict[str, float]:
    training = optimizer is not None
    if training and scaler is None:
        raise ValueError("A GradScaler is required for training")

    model.train(training)
    totals = defaultdict(float)
    batches = 0
    amp_enabled = bool(args.amp and device.type == "cuda")

    for temporal, area in loader:
        if temporal.ndim != 4 or area.ndim != 3:
            raise ValueError(
                "Group loader must return temporal [G,K,S,T] and area [G,K,S]"
            )

        groups, samples, num_sds, num_time_bins = temporal.shape
        temporal = temporal.to(device, non_blocking=True)
        area = area.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                z = model.encode(
                    temporal.reshape(groups * samples, num_sds, num_time_bins),
                    area.reshape(groups * samples, num_sds),
                ).reshape(groups, samples, -1)
                loss, metrics = subject_group_loss(
                    z,
                    temperature=args.temperature,
                    compact_weight=args.compact_weight,
                    randomize_split=training,
                )

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

        for key, value in metrics.items():
            totals[key] += float(value.item())
        batches += 1

    if batches == 0:
        raise RuntimeError("Group loader returned no batches")
    return {key: value / batches for key, value in totals.items()}


def train_model(model, bundle, args):
    min_checkpoint_epoch = int(args.min_checkpoint_epoch)
    smooth_window = int(args.checkpoint_smooth_window)
    if not 1 <= min_checkpoint_epoch <= int(args.epochs):
        raise ValueError(
            "min_checkpoint_epoch must be between 1 and epochs, "
            f"got {min_checkpoint_epoch} for {args.epochs} epochs"
        )
    if smooth_window < 1:
        raise ValueError("checkpoint_smooth_window must be at least one")

    device = resolve_device(args.device)
    model = model.to(device)
    train_loader = bundle.make_train_group_loader(args.num_workers)
    val_loader = bundle.val_group_loader

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.05,
    )
    scaler = _make_scaler(bool(args.amp and device.type == "cuda"))

    history = defaultdict(list)
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0

    print(f"Using device: {device}")
    print(
        f"{'Epoch':>5} | {'Train':>8} {'Val':>8} | "
        f"{'VProto':>8} {'VComp':>8} | {'Smooth':>8} {'LR':>10}"
    )
    print("-" * 78)

    for epoch in range(args.epochs):
        bundle.train_group_sampler.set_epoch(epoch)
        train_metrics = _run_group_epoch(
            model,
            train_loader,
            device,
            args,
            optimizer=optimizer,
            scaler=scaler,
        )

        with torch.inference_mode():
            val_group_metrics = _run_group_epoch(
                model,
                val_loader,
                device,
                args,
            )

        scheduler.step()

        for key, value in train_metrics.items():
            history[f"train_{key}"].append(value)
        for key, value in val_group_metrics.items():
            history[f"val_{key}"].append(value)
        recent_losses = history["val_total"][-smooth_window:]
        selection_loss = float(sum(recent_losses) / len(recent_losses))
        history["val_selection_loss"].append(selection_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"{epoch + 1:5d} | {train_metrics['total']:8.4f} "
            f"{val_group_metrics['total']:8.4f} | "
            f"{val_group_metrics['prototype']:8.4f} "
            f"{val_group_metrics['compactness']:8.4f} | "
            f"{selection_loss:8.4f} {optimizer.param_groups[0]['lr']:10.2e}"
        )

        current_epoch = epoch + 1
        can_checkpoint = (
            current_epoch >= min_checkpoint_epoch
            and len(history["val_total"]) >= smooth_window
        )
        if can_checkpoint and best_loss - selection_loss > _MIN_DELTA:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
            print(
                f"    [Save] smoothed_val_loss={selection_loss:.6f}, "
                f"val_loss={val_group_metrics['total']:.6f}"
            )
        elif can_checkpoint:
            patience += 1
            print(
                f"    [Patience {patience}/{args.patience}] "
                f"best={best_loss:.6f}, smoothed={selection_loss:.6f}, "
                f"val_loss={val_group_metrics['total']:.6f}"
            )
        else:
            print(
                f"    [Checkpoint warmup {current_epoch}/"
                f"{min_checkpoint_epoch}] val_loss={val_group_metrics['total']:.6f}"
            )

        if can_checkpoint and patience >= args.patience:
            print(f"Early stopping. Best epoch: {best_epoch + 1}")
            break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_epoch = len(history["train_total"]) - 1

    model.load_state_dict(best_state)
    return model, dict(history), best_epoch, device
