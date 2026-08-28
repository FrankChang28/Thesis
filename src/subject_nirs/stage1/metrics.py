from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from .utils import unit_rows


@torch.inference_mode()
def collect_embeddings(
    model,
    loader,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    z_parts, c_parts, raw_c_parts, sid_parts, op_parts = [], [], [], [], []
    amp_enabled = bool(use_amp and device.type == "cuda")

    for temporal, area, conditions, raw_conditions, subject_ids, op_positions in loader:
        temporal = temporal.to(device, non_blocking=True)
        area = area.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            z = model.encode(temporal, area)

        z_parts.append(z.float().cpu().numpy())
        c_parts.append(conditions.numpy())
        raw_c_parts.append(raw_conditions.numpy())
        sid_parts.append(subject_ids.numpy())
        op_parts.append(op_positions.numpy())

    return {
        "z": np.concatenate(z_parts).astype(np.float32),
        "c": np.concatenate(c_parts).astype(np.float32),
        "raw_c": np.concatenate(raw_c_parts).astype(np.float32),
        "subject_ids": np.concatenate(sid_parts).astype(np.int64),
        "op_positions": np.concatenate(op_parts).astype(np.int64),
    }


def _grid(payload: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(payload["z"], dtype=np.float32)
    ids = np.asarray(payload["subject_ids"], dtype=np.int64)
    ops = np.asarray(payload["op_positions"], dtype=np.int64)
    subjects = np.sort(np.unique(ids))
    positions = np.sort(np.unique(ops))
    if len(z) != len(subjects) * len(positions):
        raise ValueError("Payload is not a complete subject x OP grid")
    return z.reshape(len(subjects), len(positions), -1), subjects, positions


def evaluate_aligned_representation(payload: Dict[str, np.ndarray]) -> Dict[str, float]:
    z_grid, subjects, positions = _grid(payload)
    z_unit = unit_rows(z_grid.reshape(-1, z_grid.shape[-1])).reshape(z_grid.shape)
    num_subjects, num_positions, _ = z_grid.shape

    within_distances = []
    triangle = np.triu_indices(num_positions, k=1)
    for subject_index in range(num_subjects):
        distance = 1.0 - z_unit[subject_index] @ z_unit[subject_index].T
        within_distances.append(distance[triangle])
    within = float(np.mean(np.concatenate(within_distances)))

    if num_subjects > 1:
        between_distances = []
        subject_triangle = np.triu_indices(num_subjects, k=1)
        for op_index in range(num_positions):
            distance = 1.0 - z_unit[:, op_index] @ z_unit[:, op_index].T
            between_distances.append(distance[subject_triangle])
        between = float(np.mean(np.concatenate(between_distances)))
        ratio = within / (between + 1e-12)
    else:
        between = float("nan")
        ratio = float("nan")

    return {
        "num_subjects": int(num_subjects),
        "num_op_positions": int(num_positions),
        "same_subject_distance": within,
        "different_subject_distance": between,
        "intervention_ratio": float(ratio),
    }


def evaluate_global_subject_gallery(
    payloads: Dict[str, Dict[str, np.ndarray]],
    query_split: str,
) -> Dict[str, float]:
    if query_split not in payloads:
        raise KeyError(f"Missing query split: {query_split}")

    grids: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        name: _grid(payload) for name, payload in payloads.items()
    }
    position_counts = {len(positions) for _, _, positions in grids.values()}
    if len(position_counts) != 1:
        raise ValueError("All gallery payloads must use the same OP positions")
    num_positions = position_counts.pop()
    if num_positions < 4:
        raise ValueError("At least four OP positions are required")
    split = num_positions // 2

    reference_vectors, reference_subjects = [], []
    for grid, subjects, _ in grids.values():
        reference_vectors.append(grid[:, split:].mean(axis=1))
        reference_subjects.append(subjects)
    reference_vectors = unit_rows(np.concatenate(reference_vectors, axis=0))
    reference_subjects = np.concatenate(reference_subjects)

    query_grid, query_subjects, _ = grids[query_split]

    acquisition_queries = unit_rows(
        query_grid[:, :split].reshape(-1, query_grid.shape[-1])
    )
    acquisition_ids = np.repeat(query_subjects, split)
    acquisition_similarity = acquisition_queries @ reference_vectors.T
    acquisition_order = np.argsort(-acquisition_similarity, axis=1)
    acquisition_top1 = float(
        np.mean(reference_subjects[acquisition_order[:, 0]] == acquisition_ids)
    )
    return {
        "query_subjects": int(len(query_subjects)),
        "gallery_subjects": int(len(reference_subjects)),
        "global_acquisition_top1": acquisition_top1,
    }
