from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(data: Dict, path: str | Path) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value)}")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=convert)


def save_history_csv(history: Dict[str, list], path: str | Path) -> None:
    keys = list(history)
    rows = max(len(history[key]) for key in keys)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for index in range(rows):
            writer.writerow([index + 1, *[history[key][index] for key in keys]])


def unit_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def spherical_subject_features(
    embeddings: np.ndarray,
    subject_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    subjects = np.sort(np.unique(subject_ids))
    unit = unit_rows(np.asarray(embeddings, dtype=np.float32))
    features = []

    for subject in subjects:
        z = unit[subject_ids == subject]

        center = unit_rows(z.mean(axis=0, keepdims=True))[0]

        sim = z @ center

        idx = np.argmax(sim)

        features.append(z[idx])

    return subjects.astype(np.int64), np.asarray(features, dtype=np.float32)


def standardize_from_train(
    features: np.ndarray,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train_mask].mean(axis=0, keepdims=True)
    std = features[train_mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (features - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)
