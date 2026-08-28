from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import unit_rows

def plot_training_history(history: Dict[str, list], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_total"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, history["train_total"], label="train")
    axes[0, 0].plot(epochs, history["val_total"], label="validation")
    axes[0, 0].plot(
        epochs,
        history["val_selection_loss"],
        label="validation moving average",
        linewidth=2,
    )
    axes[0, 0].set_title("Subject-group loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["train_prototype"], label="train")
    axes[0, 1].plot(epochs, history["val_prototype"], label="validation")
    axes[0, 1].set_title("Prototype loss")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history["train_compactness"], label="train")
    axes[1, 0].plot(epochs, history["val_compactness"], label="validation")
    axes[1, 0].set_title("Compactness loss")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history["lr"], color="tab:purple")
    axes[1, 1].set_title("Learning rate")

    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output / "training_history.png", dpi=180)
    plt.close(fig)


def plot_global_tsne(
    payloads: Dict[str, Dict[str, np.ndarray]],
    output_dir: str | Path,
    perplexity: float = 40.0,
    samples_per_subject: int = 100,
    seed: int = 42,
) -> None:
    split_names = [name for name in ("train", "val", "test") if name in payloads]
    if not split_names:
        return

    z_all = np.concatenate([payloads[name]["z"] for name in split_names])
    c_all = np.concatenate([payloads[name]["raw_c"] for name in split_names])
    id_all = np.concatenate([payloads[name]["subject_ids"] for name in split_names])
    split_all = np.concatenate(
        [np.full(len(payloads[name]["z"]), name, dtype="U5") for name in split_names]
    )

    rng = np.random.default_rng(seed)
    selected = []
    for subject in np.sort(np.unique(id_all)):
        indices = np.where(id_all == subject)[0]
        if samples_per_subject > 0:
            indices = rng.choice(
                indices,
                size=min(samples_per_subject, len(indices)),
                replace=False,
            )
        selected.extend(indices.tolist())
    selected = np.asarray(selected, dtype=np.int64)

    z = z_all[selected]
    c = c_all[selected]
    ids = id_all[selected]
    splits = split_all[selected]
    effective_perplexity = min(float(perplexity), max(2.0, (len(z) - 1) / 3.0))
    embedded = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=effective_perplexity,
        random_state=seed,
        max_iter=1000,
    ).fit_transform(z)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for axis, condition_index, label in zip(
        axes[:3],
        [-3, -2, -1],
        ["Scalp", "Skull", "GM"],
    ):
        scatter = axis.scatter(
            embedded[:, 0],
            embedded[:, 1],
            c=c[:, condition_index],
            cmap="viridis",
            s=10,
            alpha=0.55,
            linewidths=0,
        )
        fig.colorbar(scatter, ax=axis).set_label(rf"$\mu_{{s,{label}}}$")
        axis.set_title(rf"$\mu_{{s,{label}}}$")

    axis = axes[3]
    train_mask = splits == "train"
    axis.scatter(
        embedded[train_mask, 0],
        embedded[train_mask, 1],
        c="lightgray",
        alpha=0.20,
        s=14,
        linewidths=0,
        label="Train subjects",
    )
    color_map = plt.get_cmap("tab10")
    for index, subject in enumerate(np.sort(np.unique(ids[splits == "val"]))):
        mask = (splits == "val") & (ids == subject)
        axis.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            color=color_map(index % 10),
            alpha=0.75,
            s=24,
            linewidths=0,
            label=f"Val {int(subject)}",
        )
    for subject in np.sort(np.unique(ids[splits == "test"])):
        mask = (splits == "test") & (ids == subject)
        axis.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c="red",
            marker="*",
            edgecolors="black",
            linewidths=0.4,
            s=120,
            alpha=0.85,
            label=f"Test {int(subject)}",
        )
    axis.set_title("Train / validation / test")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    for axis in axes:
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "global_tsne_visualization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_global_umap(
    payloads: Dict[str, Dict[str, np.ndarray]],
    output_dir: str | Path,
    n_neighbors: int = 15,    
    min_dist: float = 0.1, 
    samples_per_subject: int = 100,
    seed: int = 42,
) -> None:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "plot_global_umap requires the optional umap-learn package"
        ) from exc

    split_names = [name for name in ("train", "val", "test") if name in payloads]
    if not split_names:
        return

    z_all = np.concatenate([payloads[name]["z"] for name in split_names])
    c_all = np.concatenate([payloads[name]["raw_c"] for name in split_names])
    id_all = np.concatenate([payloads[name]["subject_ids"] for name in split_names])
    split_all = np.concatenate(
        [np.full(len(payloads[name]["z"]), name, dtype="U5") for name in split_names]
    )

    rng = np.random.default_rng(seed)
    selected = []
    for subject in np.sort(np.unique(id_all)):
        indices = np.where(id_all == subject)[0]
        if samples_per_subject > 0:
            indices = rng.choice(
                indices,
                size=min(samples_per_subject, len(indices)),
                replace=False,
            )
        selected.extend(indices.tolist())
    selected = np.asarray(selected, dtype=np.int64)

    z = z_all[selected]
    c = c_all[selected]
    ids = id_all[selected]
    splits = split_all[selected]
    
    effective_neighbors = min(int(n_neighbors), len(z) - 1)
    if effective_neighbors < 2:
        effective_neighbors = 2

    embedded = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric="cosine",  
        random_state=seed,
    ).fit_transform(z)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for axis, condition_index, label in zip(
        axes[:3],
        [-3, -2, -1],
        ["Scalp", "Skull", "GM"],
    ):
        scatter = axis.scatter(
            embedded[:, 0],
            embedded[:, 1],
            c=c[:, condition_index],
            cmap="viridis",
            s=10,
            alpha=0.55,
            linewidths=0,
        )
        fig.colorbar(scatter, ax=axis).set_label(rf"$\mu_{{s,{label}}}$")
        axis.set_title(rf"$\mu_{{s,{label}}}$")

    axis = axes[3]
    train_mask = splits == "train"
    axis.scatter(
        embedded[train_mask, 0],
        embedded[train_mask, 1],
        c="lightgray",
        alpha=0.20,
        s=14,
        linewidths=0,
        label="Train subjects",
    )
    color_map = plt.get_cmap("tab10")
    for index, subject in enumerate(np.sort(np.unique(ids[splits == "val"]))):
        mask = (splits == "val") & (ids == subject)
        axis.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            color=color_map(index % 10),
            alpha=0.75,
            s=24,
            linewidths=0,
            label=f"Val {int(subject)}",
        )
    for subject in np.sort(np.unique(ids[splits == "test"])):
        mask = (splits == "test") & (ids == subject)
        axis.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c="red",
            marker="*",
            edgecolors="black",
            linewidths=0.4,
            s=120,
            alpha=0.85,
            label=f"Test {int(subject)}",
        )
    axis.set_title("Train / validation / test")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    for axis in axes:
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "global_umap_visualization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_subject_centroid_pca(
    payloads: Dict[str, Dict[str, np.ndarray]],
    output_dir: str | Path,
) -> None:
    centroids, labels, subjects = [], [], []
    for split_name, payload in payloads.items():
        z = unit_rows(payload["z"])
        for subject in np.sort(np.unique(payload["subject_ids"])):
            center = z[payload["subject_ids"] == subject].mean(axis=0, keepdims=True)
            centroids.append(unit_rows(center)[0])
            labels.append(split_name)
            subjects.append(int(subject))

    embedded = PCA(n_components=2).fit_transform(np.asarray(centroids))
    labels_array = np.asarray(labels)

    fig, axis = plt.subplots(figsize=(9, 7))
    markers = {"train": "o", "val": "s", "test": "*"}
    for split_name in ("train", "val", "test"):
        mask = labels_array == split_name
        if mask.any():
            axis.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                marker=markers[split_name],
                s=80 if split_name == "test" else 35,
                alpha=0.75,
                label=split_name,
            )
    for point, subject, split_name in zip(embedded, subjects, labels):
        if split_name != "train":
            axis.annotate(str(subject), point, fontsize=7, alpha=0.8)

    axis.set_title("Subject spherical centroids (PCA)")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "subject_centroid_pca.png", dpi=180)
    plt.close(fig)
