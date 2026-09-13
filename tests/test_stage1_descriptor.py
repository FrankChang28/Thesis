from __future__ import annotations

import numpy as np

from subject_nirs.stage1.utils import (
    select_nearest_to_spherical_center,
    spherical_subject_features,
)


def test_descriptor_is_an_observed_embedding_nearest_the_spherical_center() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-0.8, -0.6],
        ],
        dtype=np.float32,
    )
    subject_ids = np.array([0, 0, 0, 1, 1])

    subjects, descriptors = select_nearest_to_spherical_center(
        embeddings,
        subject_ids,
    )

    np.testing.assert_array_equal(subjects, np.array([0, 1]))
    for subject, descriptor in zip(subjects, descriptors):
        observed = embeddings[subject_ids == subject]
        assert any(np.allclose(descriptor, row) for row in observed)


def test_legacy_function_name_is_a_compatible_alias() -> None:
    embeddings = np.eye(2, dtype=np.float32)
    subject_ids = np.array([0, 1])
    expected = select_nearest_to_spherical_center(embeddings, subject_ids)
    actual = spherical_subject_features(embeddings, subject_ids)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])
