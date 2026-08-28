#!/usr/bin/env python3
"""Evaluate every Stage-1 LOSO checkpoint on the complete OP grid.

For each fold, the 2700 OP positions are deterministically divided into two
disjoint halves.  The reference half forms one spherical centroid per subject;
the query half evaluates the held-out subject against the 154-subject gallery.
The script is resumable and writes both per-fold JSON files and one aggregate
CSV used by the thesis plotting notebooks.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from .model import DTOFSubjectEncoder
from .utils import unit_rows


EPS = 1e-10


class LazyDTOFArray:
    """Read MATLAB DTOFs from HDF5 by acquisition batch instead of into RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = h5py.File(path, "r")
        self._dataset = self._handle["dataset_dtof"]
        if self._dataset.ndim != 3:
            self.close()
            raise ValueError(
                f"Expected stored DTOF [T,S,N], got {self._dataset.shape}"
            )
        time_bins, num_sds, acquisitions = self._dataset.shape
        self.shape = (int(acquisitions), int(num_sds), int(time_bins))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return np.empty((0, self.shape[1], self.shape[2]), dtype=np.float32)
        if indices.min() < 0 or indices.max() >= self.shape[0]:
            raise IndexError("DTOF acquisition index is out of range")

        # h5py requires monotonically increasing fancy indices. Preserve the
        # caller's order in case a future evaluation uses shuffled batches.
        order = np.argsort(indices, kind="stable")
        sorted_indices = indices[order]
        if np.any(np.diff(sorted_indices) == 0):
            raise ValueError("Duplicate DTOF indices in one inference batch")
        stored = self._dataset[:, :, sorted_indices]
        batch = np.asarray(stored.transpose(2, 1, 0), dtype=np.float32)
        if not np.array_equal(order, np.arange(len(order))):
            batch = batch[np.argsort(order, kind="stable")]
        return batch

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--experiment", default="relat_cons_32")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: <result_root>/result/full_grid_<experiment>",
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Optional zero-based fold IDs. Omit to evaluate every available fold.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_raw_dtof(data_dir: Path) -> LazyDTOFArray:
    path = data_dir.expanduser().resolve() / "dataset_dtof.mat"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}")
    raw = LazyDTOFArray(path)
    print(f"Using batched DTOF access: {path}; logical shape={raw.shape}")
    return raw


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    required = {"model_state_dict", "model_config", "preprocess_stats"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"{path} is missing {sorted(missing)}")
    return checkpoint


def build_model(checkpoint: Mapping[str, Any], device: torch.device):
    model = DTOFSubjectEncoder(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def _stats_arrays(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    stats = checkpoint["preprocess_stats"]
    return {
        "low": float(stats["temporal_low"]),
        "high": float(stats["temporal_high"]),
        "area_mean": torch.as_tensor(stats["area_mean"], dtype=torch.float32),
        "area_std": torch.as_tensor(stats["area_std"], dtype=torch.float32),
    }


def preprocess_batch(raw_batch: np.ndarray, stats: Mapping[str, Any]):
    temporal = torch.from_numpy(np.asarray(raw_batch, dtype=np.float32))
    area = temporal.sum(dim=2).clamp_min(EPS)
    total_area = area.sum(dim=1, keepdim=True).clamp_min(EPS)
    relative_area = (area / total_area).clamp_min(EPS).log10()
    relative_area = (relative_area - stats["area_mean"]) / stats["area_std"]

    temporal = (temporal / area.unsqueeze(-1)).clamp_min(EPS).log10()
    temporal = temporal.clamp(stats["low"], stats["high"])
    temporal = (temporal - stats["low"]) / (stats["high"] - stats["low"])
    return temporal, relative_area


@torch.inference_mode()
def infer_embeddings(
    model,
    raw_dtof: LazyDTOFArray,
    global_indices: np.ndarray,
    checkpoint: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    stats = _stats_arrays(checkpoint)
    outputs: list[np.ndarray] = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    for start in range(0, len(global_indices), batch_size):
        indices = global_indices[start : start + batch_size]
        temporal, area = preprocess_batch(raw_dtof[indices], stats)
        temporal = temporal.to(device, non_blocking=True)
        area = area.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            z = model.encode(temporal, area)
        outputs.append(z.float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def make_op_split(op_per_subject: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if op_per_subject < 4:
        raise ValueError("At least four OP positions are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(op_per_subject).astype(np.int64)
    split = op_per_subject // 2
    query = np.sort(shuffled[:split])
    reference = np.sort(shuffled[split:])
    return query, reference


def subject_position_indices(
    subjects: Sequence[int], positions: np.ndarray, op_per_subject: int
) -> np.ndarray:
    subjects = np.asarray(subjects, dtype=np.int64)
    return (subjects[:, None] * op_per_subject + positions[None, :]).reshape(-1)


def reference_centroids(
    model,
    raw_dtof: LazyDTOFArray,
    checkpoint: Mapping[str, Any],
    num_subjects: int,
    op_per_subject: int,
    positions: np.ndarray,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    subjects = np.arange(num_subjects, dtype=np.int64)
    indices = subject_position_indices(subjects, positions, op_per_subject)
    z = infer_embeddings(
        model, raw_dtof, indices, checkpoint, device, batch_size, use_amp
    )
    z = z.reshape(num_subjects, len(positions), -1)
    return unit_rows(z.mean(axis=1)).astype(np.float32)


def query_metrics(
    query_z: np.ndarray,
    test_subject: int,
    centroids: np.ndarray,
) -> dict[str, Any]:
    query_z = unit_rows(query_z)
    similarities = query_z @ centroids.T
    own_similarity = similarities[:, test_subject]

    wrong = similarities.copy()
    wrong[:, test_subject] = -np.inf
    nearest_wrong_subject = np.argmax(wrong, axis=1)
    nearest_wrong_similarity = wrong[np.arange(len(wrong)), nearest_wrong_subject]

    intra_distance = 1.0 - own_similarity
    nearest_inter_distance = 1.0 - nearest_wrong_similarity
    margin = own_similarity - nearest_wrong_similarity
    ratio = intra_distance / np.maximum(nearest_inter_distance, 1e-12)

    ranks = 1 + np.sum(similarities > own_similarity[:, None], axis=1)
    top1 = float(np.mean(ranks == 1))
    nearest_counts = np.bincount(nearest_wrong_subject, minlength=len(centroids))
    most_confused = int(np.argmax(nearest_counts))

    test_centroid_similarity = centroids @ centroids[test_subject]
    test_centroid_similarity[test_subject] = -np.inf
    nearest_centroid_subject = int(np.argmax(test_centroid_similarity))

    return {
        "query_acquisitions": int(len(query_z)),
        "gallery_subjects": int(len(centroids)),
        "top1": top1,
        "mrr": float(np.mean(1.0 / ranks)),
        "intra_distance_mean": float(np.mean(intra_distance)),
        "intra_distance_median": float(np.median(intra_distance)),
        "nearest_inter_distance_mean": float(np.mean(nearest_inter_distance)),
        "nearest_inter_distance_median": float(np.median(nearest_inter_distance)),
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "margin_p05": float(np.quantile(margin, 0.05)),
        "distance_ratio_mean": float(np.mean(ratio)),
        "distance_ratio_median": float(np.median(ratio)),
        "most_confused_subject_id": most_confused,
        "most_confused_fraction": float(nearest_counts[most_confused] / len(query_z)),
        "nearest_centroid_subject_id": nearest_centroid_subject,
        "nearest_centroid_distance": float(
            1.0 - test_centroid_similarity[nearest_centroid_subject]
        ),
    }


def discover_checkpoints(
    result_root: Path, experiment: str, requested_folds: Sequence[int] | None
) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    requested = None if requested_folds is None else set(requested_folds)
    for path in result_root.glob(f"*/{experiment}/best_model.pt"):
        match = re.fullmatch(r"\d+", path.parent.parent.name)
        if not match:
            continue
        fold = int(match.group())
        if requested is None or fold in requested:
            found.append((fold, path))
    return sorted(found)


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_existing_rows(
    checkpoints: Sequence[tuple[int, Path]], experiment: str
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for fold, checkpoint_path in checkpoints:
        path = checkpoint_path.parent / "full_grid_evaluation.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") == experiment:
            rows[fold] = payload
    return rows


def write_aggregate(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "full_grid_metrics_by_fold.csv"
    keys = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main() -> None:
    args = build_parser().parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else result_root / "result" / f"full_grid_{args.experiment}"
    )
    checkpoints = discover_checkpoints(result_root, args.experiment, args.folds)
    if not checkpoints:
        raise FileNotFoundError("No matching best_model.pt checkpoints were found")

    existing = {} if args.overwrite else load_existing_rows(checkpoints, args.experiment)
    pending = [(fold, path) for fold, path in checkpoints if fold not in existing]
    print(f"Found {len(checkpoints)} folds; {len(pending)} require evaluation")

    raw_dtof = load_raw_dtof(args.data_dir) if pending else None
    device = resolve_device(args.device)
    rows = dict(existing)

    for number, (fold, checkpoint_path) in enumerate(pending, start=1):
        print(f"\n[{number}/{len(pending)}] fold={fold}: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        model_config = dict(checkpoint["model_config"])
        num_subjects = int(
            dict(checkpoint.get("args", {})).get("num_subjects", 154)
        )
        if raw_dtof.shape[0] % num_subjects:
            raise ValueError("Acquisition count is not divisible by num_subjects")
        op_per_subject = raw_dtof.shape[0] // num_subjects
        if tuple(raw_dtof.shape[1:]) != (
            int(model_config["num_sds"]),
            int(model_config["num_time_bins"]),
        ):
            raise ValueError("DTOF shape does not match checkpoint model_config")

        query_positions, reference_positions = make_op_split(
            op_per_subject, args.split_seed
        )
        model = build_model(checkpoint, device)
        centroids = reference_centroids(
            model,
            raw_dtof,
            checkpoint,
            num_subjects,
            op_per_subject,
            reference_positions,
            device,
            args.batch_size,
            args.amp,
        )
        test_ids = np.asarray(checkpoint.get("test_subjects", [fold])).reshape(-1)
        if len(test_ids) != 1:
            raise ValueError(f"Fold {fold} does not contain exactly one test subject")
        test_subject = int(test_ids[0])
        query_indices = subject_position_indices(
            [test_subject], query_positions, op_per_subject
        )
        query_z = infer_embeddings(
            model,
            raw_dtof,
            query_indices,
            checkpoint,
            device,
            args.batch_size,
            args.amp,
        )
        metrics = query_metrics(query_z, test_subject, centroids)
        row = {
            "experiment": args.experiment,
            "fold": fold,
            "subject_id": test_subject,
            "matlab_subject_id": test_subject + 1,
            "op_per_subject": op_per_subject,
            "query_positions": len(query_positions),
            "reference_positions": len(reference_positions),
            "op_split_seed": args.split_seed,
            **metrics,
        }
        save_json(row, checkpoint_path.parent / "full_grid_evaluation.json")
        rows[fold] = row
        print(
            f"Top-1={row['top1']:.4f}, MRR={row['mrr']:.4f}, "
            f"margin={row['margin_mean']:.4f}"
        )
        del model, centroids, query_z
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ordered_rows = [rows[fold] for fold, _ in checkpoints if fold in rows]
    csv_path = write_aggregate(ordered_rows, output_dir)
    split_payload = {
        "op_per_subject": int(ordered_rows[0]["op_per_subject"]),
        "split_seed": int(args.split_seed),
        "query_positions": make_op_split(
            int(ordered_rows[0]["op_per_subject"]), args.split_seed
        )[0].tolist(),
        "reference_positions": make_op_split(
            int(ordered_rows[0]["op_per_subject"]), args.split_seed
        )[1].tolist(),
    }
    save_json(split_payload, output_dir / "op_split.json")
    print(f"\nSaved {len(ordered_rows)} fold rows to {csv_path}")


if __name__ == "__main__":
    main()
