from __future__ import annotations

import argparse

from ..common.config import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a stable DTOF subject embedding with a shared temporal CNN, "
            "fixed-order SDS fusion, prototype contrastive learning, and "
            "within-subject compactness."
        )
    )

    # Data and subject split
    parser.add_argument("--config", type=str, default=None, help="Optional YAML defaults.")
    parser.add_argument("--data_dir", type=str, default="./data/raw/stage1")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="relat_cons_32")
    parser.add_argument("--num_subjects", type=int, default=154)
    parser.add_argument("--test_id", type=int, default=0, help="Zero-based test subject ID.")
    parser.add_argument("--split_seed", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Subject-group sampling
    parser.add_argument("--subjects_per_batch", type=int, default=16)
    parser.add_argument("--samples_per_subject", type=int, default=16)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--val_group_rounds", type=int, default=24)

    # Encoder
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.10)

    # Optimization
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--min_checkpoint_epoch",
        type=int,
        default=20,
        help="Do not select checkpoints or early-stop before this epoch.",
    )
    parser.add_argument(
        "--checkpoint_smooth_window",
        type=int,
        default=5,
        help="Number of recent validation losses averaged for checkpoint selection.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--compact_weight", type=float, default=0.5)

    # Validation and export
    parser.add_argument("--eval_op_positions", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip_tsne", action="store_true")
    parser.add_argument("--tsne_perplexity", type=float, default=40.0)
    parser.add_argument("--tsne_samples_per_subject", type=int, default=100)
    parser.add_argument("--save_csv", action="store_true")

    return parser


def get_args() -> argparse.Namespace:
    parser = build_parser()
    config_only = argparse.ArgumentParser(add_help=False)
    config_only.add_argument("--config", type=str, default=None)
    known, _ = config_only.parse_known_args()
    if known.config:
        values = load_yaml(known.config)
        allowed = {action.dest for action in parser._actions}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise KeyError(f"Unknown Stage 1 configuration keys: {', '.join(unknown)}")
        parser.set_defaults(**values)
    return parser.parse_args()
