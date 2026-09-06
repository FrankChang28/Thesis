from __future__ import annotations

import numpy as np
import pandas as pd

from subject_nirs.stage2.comparison import (
    build_final_experiment_table,
    format_final_experiment_table,
    holm_adjust,
    paired_comparisons,
    paired_rank_biserial,
)


def _frame() -> pd.DataFrame:
    rows = []
    for subject_id, baseline_rmse, experiment_rmse, bias in (
        (1, 3.0, 2.0, -0.2),
        (2, 4.0, 3.5, 0.1),
        (3, 5.0, 5.5, 0.3),
        (4, 6.0, 4.0, -0.1),
    ):
        rows.append(
            {
                "target": "hc", "experiment": "DRS only", "training_mode": "baseline",
                "feature_source": "baseline", "fusion_method": "none",
                "training_strategy": "scratch", "structure_control": "none",
                "subject_id": subject_id, "RMSE": baseline_rmse, "MAE": baseline_rmse - 0.2,
                "Bias": bias, "AbsBias": abs(bias), "ErrorStd": baseline_rmse,
            }
        )
        rows.append(
            {
                "target": "hc", "experiment": "DTOF latent | Residual | Fine-tune",
                "training_mode": "residual", "feature_source": "latent",
                "fusion_method": "residual", "training_strategy": "finetune",
                "structure_control": "actual", "subject_id": subject_id,
                "RMSE": experiment_rmse, "MAE": experiment_rmse - 0.2,
                "Bias": bias / 2, "AbsBias": abs(bias / 2), "ErrorStd": experiment_rmse,
            }
        )
    return pd.DataFrame(rows)


def test_rank_biserial_orientation() -> None:
    assert paired_rank_biserial(np.array([-3.0, -2.0, -1.0])) == 1.0
    assert paired_rank_biserial(np.array([3.0, 2.0, 1.0])) == -1.0


def test_holm_adjustment_is_monotone() -> None:
    adjusted = holm_adjust(pd.Series([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted.to_numpy(), [0.03, 0.06, 0.06])


def test_final_table_contains_requested_statistics() -> None:
    frame = _frame()
    detail, paired = paired_comparisons(frame, "DRS only")
    table = build_final_experiment_table(frame, paired)
    formatted = format_final_experiment_table(table)
    fusion = table[table["feature_source"] == "latent"].iloc[0]

    assert len(detail) == 4
    assert fusion["n_paired_subjects"] == 4
    assert fusion["delta_rmse_median"] == -0.75
    assert fusion["delta_rmse_q1"] == -1.25
    assert fusion["delta_rmse_q3"] == -0.25
    assert fusion["improved_subjects_pct"] == 75.0
    assert "RMSE median [Q1, Q3]" in formatted.columns
    assert "Holm-adjusted p" in formatted.columns


if __name__ == "__main__":
    test_rank_biserial_orientation()
    test_holm_adjustment_is_monotone()
    test_final_table_contains_requested_statistics()
    print("stage2 comparison tests passed")
