from __future__ import annotations

import numpy as np

from subject_nirs.stage2.descriptor_controls import (
    repeat_rng,
    replace_test_descriptor,
    standardize_descriptor_bank,
)


def test_zero_control_is_applied_after_training_only_zscore() -> None:
    raw = np.asarray([[1.0], [3.0], [9.0]], dtype=np.float32)
    actual, _, _ = standardize_descriptor_bank(raw, [0, 1])
    controlled, metadata = replace_test_descriptor(
        actual, 2, "zero", repeat_rng(7, 0, 2), [0, 1]
    )
    np.testing.assert_allclose(controlled[2], 0.0)
    np.testing.assert_allclose(controlled[:2], actual[:2])
    assert metadata == {"control_mode": "zero"}


def test_shuffle_uses_one_training_subject_descriptor() -> None:
    actual = np.arange(12, dtype=np.float32).reshape(4, 3)
    controlled, metadata = replace_test_descriptor(
        actual, 3, "shuffle", repeat_rng(11, 0, 3), [0, 1, 2]
    )
    donor = int(metadata["donor_subject_id"])
    assert donor in {0, 1, 2}
    np.testing.assert_allclose(controlled[3], actual[donor])
    np.testing.assert_allclose(controlled[:3], actual[:3])


def test_random_acquisition_is_reproducible_and_standardized() -> None:
    actual = np.zeros((3, 2), dtype=np.float32)
    acquisitions = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    kwargs = dict(
        actual_bank=actual,
        test_id=2,
        mode="random_acquisition",
        train_subjects=[0, 1],
        acquisition_raw=acquisitions,
        mean=np.asarray([10.0, 10.0], dtype=np.float32),
        std=np.asarray([2.0, 5.0], dtype=np.float32),
    )
    first, first_meta = replace_test_descriptor(rng=repeat_rng(13, 4, 2), **kwargs)
    second, second_meta = replace_test_descriptor(rng=repeat_rng(13, 4, 2), **kwargs)
    assert first_meta == second_meta
    index = int(first_meta["acquisition_index"])
    expected = (acquisitions[index] - kwargs["mean"]) / kwargs["std"]
    np.testing.assert_allclose(first[2], expected)
    np.testing.assert_allclose(second, first)
