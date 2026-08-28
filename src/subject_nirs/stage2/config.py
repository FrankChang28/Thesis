"""Central configuration for NIRS LOSO regression experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional

import torch

TargetMode = Literal["hc", "sto2"]
ExperimentMode = Literal["baseline", "fusion"]
BaselineInit = Literal["scratch", "pretrained"]
BaselineTrain = Literal["frozen", "finetune"]
FusionMethod = Literal["residual", "film"]
StructureControl = Literal["actual", "zero", "shuffle"]
SubjectFeatureSource = Literal["dtof", "metadata"]


@dataclass
class Config:
    """All experiment settings and validated derived values.

    The model configuration is deliberately separated into three independent
    choices:

    * ``baseline_init``: initialize the DRS network from scratch or checkpoint.
    * ``baseline_train``: freeze or fine-tune the DRS network during fusion.
    * ``fusion_method``: choose how the subject feature modifies DRS features.
    """

    # ------------------------------------------------------------------
    # Experiment definition
    # ------------------------------------------------------------------
    target_mode: TargetMode = "hc"
    experiment_mode: ExperimentMode = "fusion"

    baseline_init: BaselineInit = "pretrained"
    baseline_train: BaselineTrain = "finetune"
    fusion_method: FusionMethod = "residual"

    baseline_checkpoint_path: Optional[str] = (
        "./artifacts/stage2/hc_baseline_scratch_main/"
        "best_model_fold_{fold:03d}_test_id_{test_id}.pth"
    )

    residual_use_gate: bool = True
    residual_gate_init_logit: float = 0.0

    run_name_suffix: str = "main"

    # ------------------------------------------------------------------
    # Data and output paths
    # ------------------------------------------------------------------
    mat_file: str = "./data/raw/stage2/NIRS_Absolute_Dataset_grid.mat"
    output_root: str = "./artifacts/stage2"
    cache_root: str = "./cache/stage2"

    output_dir_override: Optional[str] = None
    cache_dir_override: Optional[str] = None
    compact_cache_dir_name_override: Optional[str] = None

    # ------------------------------------------------------------------
    # Subject-level structure features
    # ------------------------------------------------------------------
    subject_feature_source: SubjectFeatureSource = "dtof"
    structure_feature_path: str = (
        "./artifacts/stage1/{test_id}/relat_cons_32/subject_features_all_op.npz"
    )
    structure_feature_key: str = "latent_raw"
    structure_feature_dim: int = 32
    metadata_feature_path: str = "./data/temp_metadata_v2.csv"

    structure_control: StructureControl = "actual"
    structure_shuffle_seed: int = 2026

    # ------------------------------------------------------------------
    # LOSO split
    # ------------------------------------------------------------------
    num_subjects: int = 154
    val_subject_count: int = 10
    split_random_state_base: int = 100

    run_all_loso: bool = True
    resume: bool = True
    test_subject_id: int = 4  # zero-based
    max_folds: Optional[int] = None

    # Optional one-based inclusive MATLAB subject range. When set, this takes
    # precedence over run_all_loso and test_subject_id.
    test_subject_start: Optional[int] = 1
    test_subject_end: Optional[int] = 154

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    seed: int = 42
    batch_size: int = 4096
    epochs: int = 100

    # Fusion adapters and scratch baselines use the primary learning rate.
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6

    # Only a pretrained, fine-tuned baseline uses this smaller LR pair.
    pretrained_baseline_learning_rate: float = 1e-6
    pretrained_baseline_min_learning_rate: float = 1e-8

    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    early_stop_patience: int = 10
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5

    dropout: float = 0.10
    group_norm_groups: int = 8

    # ------------------------------------------------------------------
    # Dataset layout
    # ------------------------------------------------------------------
    num_wl: int = 5
    num_sds: int = 6

    train_sims_per_subject: int = 64
    train_rows_per_sim: int = 20 * 20 * 10 * 20

    val_sims_per_subject: int = 3
    val_rows_per_sim: int = 12 * 12 * 90

    cache_target_names: ClassVar[tuple[str, str]] = ("GM_hc", "GM_StO2")

    # ------------------------------------------------------------------
    # Training subset and cache strategy
    # ------------------------------------------------------------------
    max_train_rows_per_subject: int = 50_000
    sim_windows_per_sim: int = 16
    subset_seed: int = 42

    rebuild_preprocessed_cache: bool = False
    rebuild_compact_cache: bool = False

    compact_stage_count: int = 1
    compact_stage_seed_stride: int = 10_000

    cache_chunk_rows: int = 1_048_576
    compact_cache_chunk_rows: int = 65_536

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------
    save_epoch_log: bool = False
    save_per_subject_eval: bool = False
    save_predictions: bool = False
    save_prediction_x: bool = False

    # ------------------------------------------------------------------
    # Derived model properties
    # ------------------------------------------------------------------
    @property
    def use_structure_features(self) -> bool:
        return self.experiment_mode == "fusion"

    @property
    def uses_pretrained_baseline(self) -> bool:
        return self.use_structure_features and self.baseline_init == "pretrained"

    @property
    def freezes_baseline(self) -> bool:
        return self.use_structure_features and self.baseline_train == "frozen"

    @property
    def baseline_optimizer_lr(self) -> float:
        if self.baseline_init == "pretrained":
            return self.pretrained_baseline_learning_rate
        return self.learning_rate

    @property
    def baseline_optimizer_min_lr(self) -> float:
        if self.baseline_init == "pretrained":
            return self.pretrained_baseline_min_learning_rate
        return self.min_learning_rate

    @property
    def input_dim(self) -> int:
        return self.num_wl * self.num_sds

    @property
    def train_rows_per_subject(self) -> int:
        return self.train_sims_per_subject * self.train_rows_per_sim

    @property
    def val_rows_per_subject(self) -> int:
        return self.val_sims_per_subject * self.val_rows_per_sim

    @property
    def cache_target_dim(self) -> int:
        return len(self.cache_target_names)

    @property
    def target_index(self) -> int:
        return {"hc": 0, "sto2": 1}[self.target_mode]

    @property
    def target_name(self) -> str:
        return self.cache_target_names[self.target_index]

    @property
    def target_names(self) -> list[str]:
        return [self.target_name]

    @property
    def target_dim(self) -> int:
        return 1

    @property
    def train_rows_tag(self) -> str:
        rows = self.max_train_rows_per_subject
        return f"{rows // 1000}k" if rows % 1000 == 0 else str(rows)

    @property
    def experiment_name(self) -> str:
        parts = [self.target_mode, self.experiment_mode]

        if self.experiment_mode == "baseline":
            parts.append("scratch")
        else:
            parts.extend(
                [
                    self.subject_feature_source,
                    self.structure_control,
                    self.baseline_init,
                    self.baseline_train,
                    self.fusion_method,
                ]
            )
            if self.fusion_method == "residual":
                parts.append("gated" if self.residual_use_gate else "ungated")

        if self.run_name_suffix.strip():
            parts.append(self.run_name_suffix.strip())

        return "_".join(parts)

    @property
    def output_dir(self) -> str:
        if self.output_dir_override is not None:
            return os.path.normpath(os.path.expanduser(self.output_dir_override))
        return os.path.normpath(
            os.path.join(os.path.expanduser(self.output_root), self.experiment_name)
        )

    @property
    def cache_dir(self) -> str:
        if self.cache_dir_override is not None:
            return os.path.normpath(os.path.expanduser(self.cache_dir_override))
        return os.path.normpath(
            os.path.join(
                os.path.expanduser(self.cache_root),
                f"preprocessed_cache_grid_{self.train_rows_tag}",
            )
        )

    @property
    def compact_cache_dir_name(self) -> str:
        if self.compact_cache_dir_name_override is not None:
            return self.compact_cache_dir_name_override
        return f"compact_subsets_{self.train_rows_tag}"

    @property
    def compact_cache_dir(self) -> str:
        return os.path.join(self.cache_dir, self.compact_cache_dir_name)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        self._validate_experiment()
        self._validate_subject_split()
        self._validate_positive_values()
        self._validate_optimization()

    def _validate_experiment(self) -> None:
        if self.target_mode not in {"hc", "sto2"}:
            raise ValueError("target_mode must be 'hc' or 'sto2'")
        if self.experiment_mode not in {"baseline", "fusion"}:
            raise ValueError("experiment_mode must be 'baseline' or 'fusion'")
        if self.baseline_init not in {"scratch", "pretrained"}:
            raise ValueError("baseline_init must be 'scratch' or 'pretrained'")
        if self.baseline_train not in {"frozen", "finetune"}:
            raise ValueError("baseline_train must be 'frozen' or 'finetune'")
        if self.fusion_method not in {"residual", "film"}:
            raise ValueError("fusion_method must be 'residual' or 'film'")
        if self.subject_feature_source not in {"dtof", "metadata"}:
            raise ValueError("subject_feature_source must be 'dtof' or 'metadata'")
        if self.structure_control not in {"actual", "zero", "shuffle"}:
            raise ValueError("structure_control must be 'actual', 'zero', or 'shuffle'")

        if self.experiment_mode == "baseline":
            if self.baseline_init != "scratch":
                raise ValueError("baseline experiments must use baseline_init='scratch'")
            if self.baseline_train != "finetune":
                raise ValueError("baseline experiments require baseline_train='finetune'")
        elif self.baseline_init == "scratch" and self.baseline_train == "frozen":
            raise ValueError("a scratch baseline cannot be frozen")

        if self.uses_pretrained_baseline and not self.baseline_checkpoint_path:
            raise ValueError(
                "baseline_checkpoint_path is required when baseline_init='pretrained'"
            )

    def _validate_subject_split(self) -> None:
        if self.num_subjects < 2:
            raise ValueError("num_subjects must be at least 2")
        if not 0 <= self.test_subject_id < self.num_subjects:
            raise ValueError("test_subject_id is out of range")
        if not 1 <= self.val_subject_count < self.num_subjects - 1:
            raise ValueError(
                "val_subject_count must leave at least one training subject"
            )

        start_is_set = self.test_subject_start is not None
        end_is_set = self.test_subject_end is not None
        if start_is_set != end_is_set:
            raise ValueError(
                "test_subject_start and test_subject_end must both be set or both be None"
            )
        if start_is_set:
            assert self.test_subject_start is not None
            assert self.test_subject_end is not None
            if not (
                1
                <= self.test_subject_start
                <= self.test_subject_end
                <= self.num_subjects
            ):
                raise ValueError("invalid one-based test subject range")

    def _validate_positive_values(self) -> None:
        positive_int_fields = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stop_patience": self.early_stop_patience,
            "num_wl": self.num_wl,
            "num_sds": self.num_sds,
            "train_sims_per_subject": self.train_sims_per_subject,
            "val_sims_per_subject": self.val_sims_per_subject,
            "train_rows_per_sim": self.train_rows_per_sim,
            "val_rows_per_sim": self.val_rows_per_sim,
            "max_train_rows_per_subject": self.max_train_rows_per_subject,
            "sim_windows_per_sim": self.sim_windows_per_sim,
            "compact_stage_count": self.compact_stage_count,
            "cache_chunk_rows": self.cache_chunk_rows,
            "compact_cache_chunk_rows": self.compact_cache_chunk_rows,
        }
        for name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.structure_feature_dim <= 0:
            raise ValueError("structure_feature_dim must be positive")
        if self.max_train_rows_per_subject > self.train_rows_per_subject:
            raise ValueError(
                "max_train_rows_per_subject cannot exceed train_rows_per_subject"
            )
        if self.max_folds is not None and self.max_folds <= 0:
            raise ValueError("max_folds must be positive or None")

    def _validate_optimization(self) -> None:
        learning_rates = {
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "pretrained_baseline_learning_rate": (
                self.pretrained_baseline_learning_rate
            ),
            "pretrained_baseline_min_learning_rate": (
                self.pretrained_baseline_min_learning_rate
            ),
        }
        for name, value in learning_rates.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if (
            self.pretrained_baseline_min_learning_rate
            > self.pretrained_baseline_learning_rate
        ):
            raise ValueError(
                "pretrained_baseline_min_learning_rate cannot exceed "
                "pretrained_baseline_learning_rate"
            )
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience cannot be negative")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be between 0 and 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if self.group_norm_groups <= 0:
            raise ValueError("group_norm_groups must be positive")


CFG = Config()
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
