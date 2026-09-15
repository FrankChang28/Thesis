from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from subject_nirs.common.config import apply_dataclass_config, load_yaml
from subject_nirs.stage2.config import Config
from subject_nirs.stage2.config import CFG
from subject_nirs.stage2.latent import _apply_control, load_metadata_bank


def test_metadata_bank_uses_training_subject_zscore(tmp_path, monkeypatch):
    thickness = np.asarray(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
            [3.0, 30.0, 300.0, 3000.0],
            [5.0, 50.0, 500.0, 5000.0],
        ],
        dtype=np.float32,
    )
    table = np.zeros((len(thickness), 9), dtype=np.float32)
    table[:, 5:9] = thickness
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(table).to_csv(metadata_path, index=False)

    monkeypatch.setattr(CFG, "num_subjects", len(thickness))
    monkeypatch.setattr(CFG, "metadata_feature_path", str(metadata_path))

    train_subjects = np.asarray([0, 1, 2], dtype=np.int64)
    features, source = load_metadata_bank(train_subjects)

    train_values = thickness[train_subjects]
    expected = (thickness - train_values.mean(axis=0)) / train_values.std(axis=0)

    np.testing.assert_allclose(features, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(features[train_subjects].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(features[train_subjects].std(axis=0), 1.0, atol=1e-6)
    assert np.all(features[3] > 1.0)
    assert source == str(metadata_path)


def test_stage2_configs_cover_complete_thesis_matrix():
    config_dir = Path(__file__).parents[1] / "configs" / "stage2"
    fusion_cells = set()
    baseline_targets = set()
    training_controls = set()

    for config_path in sorted(config_dir.rglob("*.yaml")):
        config = Config()
        apply_dataclass_config(config, load_yaml(config_path))
        config.validate()

        if config.execution_mode == "descriptor_control":
            expected_relative_dir = (
                Path("descriptor_controls")
                / config.descriptor_control_mode
                / config.target_mode
            )
            assert Path(config.output_dir) == Path(config.output_root) / expected_relative_dir
            continue

        if config.structure_control != "actual":
            training_controls.add((config.target_mode, config.structure_control))
            expected_relative_dir = (
                Path(config.fusion_method)
                / config.training_strategy
                / f"{config.subject_feature_source}_{config.structure_control}"
                / config.target_mode
            )
            assert Path(config.output_dir) == Path(config.output_root) / expected_relative_dir
            continue

        if config.experiment_mode == "baseline":
            baseline_targets.add(config.target_mode)
            expected_relative_dir = Path("baseline") / config.target_mode
            assert Path(config.output_dir) == Path(config.output_root) / expected_relative_dir
            continue

        strategy = "scratch" if config.baseline_init == "scratch" else "finetune"
        cell = (
            config.target_mode,
            config.subject_feature_source,
            config.fusion_method,
            strategy,
        )
        assert cell not in fusion_cells, f"duplicate thesis config cell: {cell}"
        fusion_cells.add(cell)
        expected_relative_dir = (
            Path(config.fusion_method)
            / strategy
            / config.subject_feature_source
            / config.target_mode
        )
        assert Path(config.output_dir) == Path(config.output_root) / expected_relative_dir

    expected_cells = set(
        product(
            ("hc", "sto2"),
            ("dtof", "metadata"),
            ("concatenation", "film", "residual"),
            ("scratch", "finetune"),
        )
    )
    assert fusion_cells == expected_cells
    assert baseline_targets == {"hc", "sto2"}
    assert training_controls == {
        (target, control)
        for target in ("hc", "sto2")
        for control in ("zero", "shuffle")
    }


def test_training_shuffle_is_split_safe_and_deterministic(monkeypatch):
    features = np.arange(6, dtype=np.float32).reshape(-1, 1)
    train_subjects = np.asarray([0, 1, 2], dtype=np.int64)
    val_subjects = np.asarray([3, 4], dtype=np.int64)

    monkeypatch.setattr(CFG, "num_subjects", 6)
    monkeypatch.setattr(CFG, "structure_control", "shuffle")
    monkeypatch.setattr(CFG, "structure_shuffle_seed", 20260913)

    first, first_description = _apply_control(
        features,
        test_id=5,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
    )
    second, second_description = _apply_control(
        features,
        test_id=5,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
    )

    np.testing.assert_array_equal(first, second)
    assert first_description == second_description
    assert set(first[train_subjects, 0]) == set(features[train_subjects, 0])
    assert set(first[val_subjects, 0]) == set(features[val_subjects, 0])
    assert np.all(first[train_subjects, 0] != features[train_subjects, 0])
    assert np.all(first[val_subjects, 0] != features[val_subjects, 0])
    assert first[5, 0] in set(features[train_subjects, 0])
