from __future__ import annotations

import numpy as np
import pandas as pd

from subject_nirs.stage2.baseline_analysis import (
    compute_subject_metrics,
    summarize_metrics,
)


def test_subject_summary_reports_r2_without_nrmse() -> None:
    predictions = pd.DataFrame(
        {
            "target": ["HC"] * 6,
            "subject_id": [1, 1, 1, 2, 2, 2],
            "y_true": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "y_pred": [0.0, 1.0, 2.0, 0.0, 1.0, 1.0],
        }
    )
    predictions["error"] = predictions["y_pred"] - predictions["y_true"]

    subject_metrics = compute_subject_metrics(predictions)
    summary = summarize_metrics(predictions, subject_metrics)

    assert "R2" in subject_metrics
    assert "NRMSE_percent" not in subject_metrics
    assert np.isclose(subject_metrics.loc[subject_metrics.subject_id == 1, "R2"].iloc[0], 1.0)
    assert "subject_R2_median" in summary
    assert not any("NRMSE" in column for column in summary.columns)
