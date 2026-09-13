from __future__ import annotations

import numpy as np

from subject_nirs.stage2.unseen_scattering_plot import (
    condition_tick_labels,
    rank_biserial,
    scale_detail_for_display,
)


def test_condition_tick_labels_use_thesis_notation() -> None:
    assert condition_tick_labels("unseen_scattering", [1, 2, 3, 4]) == [
        "MK-1", "MK-2", "MK-3", "MK-4"
    ]
    assert condition_tick_labels("unseen_wm", [1, 2, 3]) == [
        "G2-1", "G2-2", "G2-3"
    ]


def test_rank_biserial_positive_means_improvement() -> None:
    assert rank_biserial(np.array([-3.0, -2.0, -1.0])) == 1.0
    assert rank_biserial(np.array([3.0, 2.0, 1.0])) == -1.0


def test_sto2_errors_are_displayed_on_percentage_scale() -> None:
    rows = [
        {
            "original_RMSE": 0.04,
            "new_RMSE": 0.05,
            "delta_RMSE": 0.01,
        }
    ]
    scaled = scale_detail_for_display(rows, "sto2")
    assert scaled[0]["original_RMSE"] == 4.0
    assert scaled[0]["new_RMSE"] == 5.0
    assert scaled[0]["delta_RMSE"] == 1.0


def test_thb_errors_keep_native_units() -> None:
    rows = [
        {
            "original_RMSE": 12.0,
            "new_RMSE": 10.0,
            "delta_RMSE": -2.0,
        }
    ]
    scaled = scale_detail_for_display(rows, "hc")
    assert scaled[0]["delta_RMSE"] == -2.0
