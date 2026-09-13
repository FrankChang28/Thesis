from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .config import get_args
from .data import choose_eval_op_positions, load_data_bundle
from .metrics import (
    collect_embeddings,
    evaluate_aligned_representation,
    evaluate_global_subject_gallery,
)
from .model import DTOFSubjectEncoder
from .training import train_model
from .utils import (
    ensure_dir,
    save_history_csv,
    save_json,
    set_seed,
    select_nearest_to_spherical_center,
    standardize_from_train,
)
from .visualization import (
    plot_global_tsne,
    plot_global_umap,
    plot_subject_centroid_pca,
    plot_training_history,
)


def main() -> None:
    args = get_args()
    set_seed(args.seed)
    output_dir = ensure_dir(
        args.output_dir
        or Path("artifacts") / "stage1" / str(args.test_id) / args.experiment_name
    )

    bundle = load_data_bundle(args)
    model = DTOFSubjectEncoder(
        num_sds=bundle.num_sds,
        num_time_bins=bundle.num_time_bins,
        latent_dim=args.latent_dim,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
    )

    model, history, best_epoch, device = train_model(model, bundle, args)
    best_validation_loss = float(history["val_selection_loss"][best_epoch])
    if args.save_csv:
        save_history_csv(history, output_dir / "history.csv")
    plot_training_history(history, output_dir)

    eval_positions = choose_eval_op_positions(
        bundle.op_per_subject,
        args.eval_op_positions,
        args.seed + 77,
    )

    payloads = {}
    split_metrics = {}
    for split_name in ("train", "val", "test"):
        loader = bundle.make_aligned_loader(
            split_name,
            eval_positions,
            args.eval_batch_size,
            args.num_workers,
        )
        payload = collect_embeddings(model, loader, device, use_amp=args.amp)
        payloads[split_name] = payload
        split_metrics[split_name] = evaluate_aligned_representation(payload)
        print(f"\n[{split_name}] representation metrics")
        for key, value in split_metrics[split_name].items():
            print(f"  {key}: {value}")

    global_metrics = {
        name: evaluate_global_subject_gallery(payloads, name)
        for name in ("train", "val", "test")
    }
    for split_name, metrics in global_metrics.items():
        print(f"\n[{split_name}] global-gallery metrics")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    if not args.skip_tsne:
        plot_global_tsne(
            payloads,
            output_dir,
            perplexity=args.tsne_perplexity,
            samples_per_subject=args.tsne_samples_per_subject,
            seed=args.seed,
        )
        # plot_global_umap(
        #     payloads,
        #     output_dir,
        #     n_neighbors = 15, 
        #     min_dist = 0.1,  
        #     seed=args.seed,
        # )
    plot_subject_centroid_pca(payloads, output_dir)

    model_config = {
        "num_sds": bundle.num_sds,
        "num_time_bins": bundle.num_time_bins,
        "latent_dim": args.latent_dim,
        "feature_dim": args.feature_dim,
        "dropout": args.dropout,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "args": vars(args),
        "best_epoch": int(best_epoch),
        "best_validation_loss": best_validation_loss,
        "preprocess_stats": bundle.stats.to_dict(),
        "train_subjects": bundle.split.train_subjects,
        "val_subjects": bundle.split.val_subjects,
        "test_subjects": bundle.split.test_subjects,
        "eval_op_positions": eval_positions,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")

    report = {
        "best_epoch_zero_based": int(best_epoch),
        "model_selection": {
            "criterion": "moving_average_validation_total_loss",
            "best_epoch_one_based": int(best_epoch) + 1,
            "best_validation_loss": best_validation_loss,
            "smooth_window": int(args.checkpoint_smooth_window),
            "min_checkpoint_epoch": int(args.min_checkpoint_epoch),
            "patience": int(args.patience),
            "min_delta": 1e-4,
        },
        "split_metrics": split_metrics,
        "global_gallery_metrics": global_metrics,
        "split_subjects": {
            "train": bundle.split.train_subjects,
            "val": bundle.split.val_subjects,
            "test": bundle.split.test_subjects,
        },
        "eval_op_positions": eval_positions,
        "preprocess_stats": bundle.stats.to_dict(),
    }
    save_json(report, output_dir / "evaluation_metrics.json")
    save_json(
        {"args": vars(args), "model_config": model_config},
        output_dir / "run_config.json",
    )

    all_z = np.concatenate([payloads[name]["z"] for name in ("train", "val", "test")])
    all_ids = np.concatenate(
        [payloads[name]["subject_ids"] for name in ("train", "val", "test")]
    )
    subject_ids, raw_features = select_nearest_to_spherical_center(all_z, all_ids)

    splits = np.full(len(subject_ids), "unknown", dtype="U8")
    splits[np.isin(subject_ids, bundle.split.train_subjects)] = "train"
    splits[np.isin(subject_ids, bundle.split.val_subjects)] = "val"
    splits[np.isin(subject_ids, bundle.split.test_subjects)] = "test"
    train_mask = splits == "train"
    features, mean, std = standardize_from_train(raw_features, train_mask)

    np.savez_compressed(
        output_dir / "subject_features_eval_grid.npz",
        features=features,
        latent_raw=raw_features,
        dtof_descriptor_raw=raw_features,
        subject_ids=subject_ids,
        matlab_subject_ids=subject_ids + 1,
        splits=splits,
        normalization_mean=mean,
        normalization_std=std,
        op_positions=eval_positions,
        aggregation=np.asarray("nearest_to_spherical_center"),
    )

    if args.save_csv:
        with open(
            output_dir / "subject_features_eval_grid.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["subject_id", "matlab_subject_id", "split"]
                + [f"z{i}" for i in range(features.shape[1])]
            )
            for subject, split_name, row in zip(subject_ids, splits, features):
                writer.writerow(
                    [int(subject), int(subject) + 1, split_name, *row.astype(float)]
                )

    print(f"\nSaved all outputs to: {output_dir}")


if __name__ == "__main__":
    main()
