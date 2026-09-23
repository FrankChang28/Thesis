from __future__ import annotations

import numpy as np

from subject_nirs.stage2.unseen_scattering_plot import (
    condition_tick_labels,
    paired_deltas,
    parse_parent_conditions,
    rank_biserial,
    scale_detail_for_display,
)


def test_condition_tick_labels_use_thesis_notation() -> None:
    assert condition_tick_labels("unseen_scattering", [1, 2, 3, 4]) == [
        "MK-1", "MK-2", "MK-3", "MK-4"
    ]
    assert condition_tick_labels("unseen_wm", [1, 2, 3]) == [
        "G2-1′", "G2-2′", "G2-3′"
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


def test_parse_parent_conditions() -> None:
    assert parse_parent_conditions("1,2,2,3", 4, 3) == (1, 2, 2, 3)


def test_paired_deltas_use_matched_parent_condition(tmp_path) -> None:
    path = tmp_path / "new.csv"
    path.write_text(
        "fold,test_id,sim_number_1based,RMSE\n"
        "1,0,1,11\n"
        "1,0,2,22\n"
        "1,0,3,23\n"
        "1,0,4,34\n",
        encoding="utf-8",
    )
    original = {
        (1, 1): (0, 10.0),
        (1, 2): (0, 20.0),
        (1, 3): (0, 30.0),
    }
    detail = paired_deltas(
        path,
        original,
        "unseen_scattering",
        "Modified-K conditions",
        4,
        (1, 2, 2, 3),
    )
    assert [row["reference_condition"] for row in detail] == [1, 2, 2, 3]
    assert [row["original_RMSE"] for row in detail] == [10.0, 20.0, 20.0, 30.0]
    assert [row["delta_RMSE"] for row in detail] == [1.0, 2.0, 3.0, 4.0]
