"""Replot selected-fold t-SNE figures from saved coordinates.

Subject IDs are zero-based, matching LOSO fold IDs. Metrics describe the test
subject in the original latent space, not distances in the t-SNE projection.
This module performs no model inference or t-SNE fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_fold_tsne(coordinates: pd.DataFrame, selected: pd.Series):
    """Preserve the legacy point geometry while clarifying IDs and metrics."""
    required = {"subject_id", "split", "tsne_1", "tsne_2"}
    missing = required.difference(coordinates.columns)
    if missing:
        raise ValueError(f"Missing t-SNE columns: {sorted(missing)}")
    test_ids = coordinates.loc[coordinates["split"] == "test", "subject_id"].unique()
    fold = int(selected["fold"])
    if len(test_ids) != 1 or int(test_ids[0]) != int(selected["subject_id"]):
        raise ValueError("Saved test subject does not match the fold metrics")
    test_subject = int(test_ids[0])

    ids = coordinates["subject_id"].to_numpy()
    split_labels = coordinates["split"].to_numpy()
    xy = coordinates[["tsne_1", "tsne_2"]].to_numpy()
    val_subjects = np.sort(np.unique(ids[split_labels == "val"]))

    fig, axis = plt.subplots(figsize=(8.2, 6.2))
    train_mask = split_labels == "train"
    axis.scatter(
        xy[train_mask, 0], xy[train_mask, 1],
        color="lightgray", alpha=0.20, s=12, linewidths=0,
        rasterized=True, label="Training subjects",
    )
    color_map = plt.get_cmap("tab10")
    for index, subject in enumerate(val_subjects):
        mask = (split_labels == "val") & (ids == subject)
        axis.scatter(
            xy[mask, 0], xy[mask, 1],
            color=color_map(index % 10), alpha=0.72, s=19, linewidths=0,
            rasterized=True, label=f"Validation subject {int(subject)}",
        )
    test_mask = split_labels == "test"
    axis.scatter(
        xy[test_mask, 0], xy[test_mask, 1],
        color="#C0392B", marker="*", edgecolors="black", linewidths=0.25,
        s=70, alpha=0.82, label=f"Test subject {test_subject}", zorder=3,
    )
    metric_text = (
        f"Test subject {test_subject} / fold {fold}\n"
        f"Top-1 = {selected['top1']:.3f}\n"
        f"$D_{{intra}}$ = {selected['intra_distance_mean']:.3f}\n"
        f"$D_{{nearest}}$ = {selected['nearest_inter_distance_mean']:.3f}"
    )
    axis.text(
        1.02, 0.30, metric_text, transform=axis.transAxes, va="top", ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.92},
    )
    axis.set_title(f"Subject latent-space distribution (fold {fold})")
    axis.legend(
        title="Subject ID (0-based)", title_fontsize=7,
        bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=7,
    )
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, axis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates_csv", type=Path, required=True)
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    coordinates = pd.read_csv(args.coordinates_csv)
    metrics = pd.read_csv(args.metrics_csv)
    selected = metrics.loc[metrics["fold"] == args.fold]
    if len(selected) != 1:
        raise ValueError(f"Expected one metrics row for fold {args.fold}")
    fig, _ = plot_fold_tsne(coordinates, selected.iloc[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.coordinates_csv.stem
    for extension in ("png", "pdf"):
        output = args.output_dir / f"{stem}.{extension}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
