from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from subject_nirs.stage2.subject_failure import (
    compute_failure_layer_thickness_summary,
    compute_feature_failure_comparison_contrasts,
    compute_failure_comparison_contrasts,
    compute_failure_contrasts,
    compute_subject_signatures,
    load_baseline_performance,
    load_layer_thickness_metadata,
    make_failure_layer_thickness_figure,
    make_subject_failure_comparison_figure,
    make_subject_failure_feature_comparison_figure,
    make_subject_failure_figure,
)


def _performance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": np.arange(8),
            "baseline_rmse": np.arange(1.0, 9.0),
            "baseline_bias": [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 1.0, -1.0],
        }
    )


def _signatures() -> tuple[np.ndarray, np.ndarray]:
    subject_ids = np.arange(8)
    base = np.linspace(-1.0, 1.0, 30)
    signatures = np.vstack(
        [base * (1.0 + 0.1 * subject_id) for subject_id in subject_ids]
    )
    return subject_ids, signatures


def _thickness() -> pd.DataFrame:
    subject_ids = np.arange(8)
    return pd.DataFrame(
        {
            "subject_id": subject_ids,
            "scalp": 2.0 + 0.1 * subject_ids,
            "skull": 4.0 + 0.2 * subject_ids,
            "csf": 1.0 + 0.3 * subject_ids,
            "sinus": 0.5 + 0.4 * subject_ids,
        }
    )


def test_target_specific_metric_loading(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "test_id": [0, 1],
            "test_GM_hc_RMSE": [10.0, 11.0],
            "test_GM_hc_Bias": [1.0, -1.0],
            "test_GM_StO2_RMSE": [0.04, 0.05],
            "test_GM_StO2_Bias": [0.01, -0.01],
        }
    )
    frame.to_csv(tmp_path / "loso_results.csv", index=False)

    hc = load_baseline_performance(tmp_path, "hc")
    sto2 = load_baseline_performance(tmp_path, "sto2")

    assert hc["baseline_rmse"].tolist() == [10.0, 11.0]
    assert sto2["baseline_rmse"].tolist() == [0.04, 0.05]


def test_layer_thickness_loading_uses_raw_mm(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "subject": ["A", "B"],
            "sbj": [0, 1],
            "scalp": [2.5, 3.5],
            "skull": [4.5, 5.5],
            "csf": [1.5, 2.5],
            "sinus": [0.5, 1.5],
        }
    )
    path = tmp_path / "metadata.csv"
    raw.to_csv(path, index=False)
    loaded = load_layer_thickness_metadata(path)

    assert loaded["subject_id"].tolist() == [0, 1]
    assert loaded["scalp"].tolist() == [2.5, 3.5]


def test_failure_groups_and_contrasts() -> None:
    subject_ids, signatures = _signatures()
    contrasts, summary = compute_failure_contrasts(
        _performance(), subject_ids, signatures
    )

    centered = signatures - signatures.mean(axis=1, keepdims=True)
    reference_center = np.median(centered[:2], axis=0)
    reference_scale = 1.4826 * np.median(
        np.abs(centered[:2] - reference_center), axis=0
    )
    expected_overestimation = (
        (centered[6] - reference_center) / reference_scale
    ).reshape(5, 6)

    assert contrasts["overestimation"].shape == (5, 6)
    assert contrasts["underestimation"].shape == (5, 6)
    np.testing.assert_allclose(contrasts["overestimation"], expected_overestimation)
    assert summary["common_n"] == 8
    assert summary["accurate_n"] == 2
    assert summary["difficult_n"] == 2
    assert summary["overestimation_n"] == 1
    assert summary["underestimation_n"] == 1


def test_pooled_signature_requires_equal_target_pair_counts(tmp_path: Path) -> None:
    x = np.arange(8 * 30, dtype=np.float32).reshape(8, 30)
    y = np.array([[0, 0], [0, 0], [1, 1], [1, 1]] * 2, dtype=np.float32)
    subject_ids = np.repeat([0, 1], 4)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", y)
    np.save(tmp_path / "subject_ids.npy", subject_ids)

    ids, signatures = compute_subject_signatures(
        tmp_path / "x.npy",
        tmp_path / "y.npy",
        tmp_path / "subject_ids.npy",
    )

    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(signatures[0], x[:4].mean(axis=0))
    np.testing.assert_allclose(signatures[1], x[4:].mean(axis=0))


def test_target_figures_use_distinct_filenames(tmp_path: Path) -> None:
    subject_ids, signatures = _signatures()
    hc = make_subject_failure_figure(
        _performance(), subject_ids, signatures, tmp_path, "hc", dpi=72
    )
    sto2 = make_subject_failure_figure(
        _performance(), subject_ids, signatures, tmp_path, "sto2", dpi=72
    )

    assert Path(hc["figure"]).name == "subject_failure_hc.png"
    assert Path(sto2["figure"]).name == "subject_failure_sto2.png"
    assert "overestimation_iqr_consistent_channels" in hc
    assert "underestimation_iqr_consistent_channels" in sto2
    assert (tmp_path / "subject_failure_hc.png").is_file()
    assert (tmp_path / "subject_failure_sto2.png").is_file()


def test_model_comparison_uses_one_reference_and_distinct_output(tmp_path: Path) -> None:
    subject_ids, signatures = _signatures()
    performance = _performance()
    contrasts, summaries = compute_failure_comparison_contrasts(
        performance,
        performance.copy(),
        subject_ids,
        signatures,
    )

    for group_name in ("overestimation", "underestimation"):
        np.testing.assert_allclose(
            contrasts["drs_only"][group_name],
            contrasts["residual_finetune_dtof"][group_name],
        )
    assert summaries["drs_only"]["contrast_reference"] == (
        "drs_only_q1_median_and_mad"
    )
    assert summaries["drs_only"]["overestimation_mean_abs_contrast"] >= 0
    rows = make_subject_failure_comparison_figure(
        performance,
        performance.copy(),
        subject_ids,
        signatures,
        tmp_path,
        "hc",
        dpi=72,
    )
    assert len(rows) == 2
    assert Path(rows[0]["figure"]).name == "subject_failure_comparison_hc.png"
    assert rows[0]["shared_color_limit"] == rows[1]["shared_color_limit"]
    assert "overestimation_iqr_consistent_channels" in rows[0]
    assert (tmp_path / "subject_failure_comparison_hc.png").is_file()


def test_feature_comparison_adds_thickness_on_the_same_scale(tmp_path: Path) -> None:
    subject_ids, signatures = _signatures()
    performance = _performance()
    contrasts, summaries = compute_feature_failure_comparison_contrasts(
        performance,
        performance.copy(),
        performance.copy(),
        subject_ids,
        signatures,
    )

    for model_name in (
        "residual_finetune_dtof",
        "residual_finetune_thickness",
    ):
        for group_name in ("overestimation", "underestimation"):
            np.testing.assert_allclose(
                contrasts["drs_only"][group_name],
                contrasts[model_name][group_name],
            )
        assert summaries[model_name]["difficult_overlap_n"] == 2

    rows = make_subject_failure_feature_comparison_figure(
        performance,
        performance.copy(),
        performance.copy(),
        subject_ids,
        signatures,
        tmp_path,
        "sto2",
        dpi=72,
    )
    assert len(rows) == 3
    assert len({row["shared_color_limit"] for row in rows}) == 1
    assert all("underestimation_iqr_consistent_channels" in row for row in rows)
    assert Path(rows[0]["figure"]).name == (
        "subject_failure_feature_comparison_sto2.png"
    )
    assert (tmp_path / "subject_failure_feature_comparison_sto2.png").is_file()


def test_failure_layer_thickness_uses_baseline_groups_and_raw_mm(
    tmp_path: Path,
) -> None:
    performance = _performance()
    thickness = _thickness()
    summary = compute_failure_layer_thickness_summary(
        performance,
        thickness,
        n_bootstrap=200,
        seed=7,
    )

    assert len(summary) == 8
    assert set(summary["group"]) == {"overestimation", "underestimation"}
    assert set(summary["layer"]) == {"scalp", "skull", "csf", "sinus"}
    assert set(summary["reference_n"]) == {2}
    assert set(summary["group_n"]) == {1}
    scalp_over = summary[
        (summary["group"] == "overestimation")
        & (summary["layer"] == "scalp")
    ].iloc[0]
    np.testing.assert_allclose(scalp_over["reference_median_mm"], 2.05)
    np.testing.assert_allclose(scalp_over["group_median_mm"], 2.6)
    np.testing.assert_allclose(scalp_over["median_difference_mm"], 0.55)

    plotted = make_failure_layer_thickness_figure(
        {"hc": performance, "sto2": performance.copy()},
        thickness,
        tmp_path,
        n_bootstrap=200,
        seed=7,
        dpi=72,
    )
    assert len(plotted) == 16
    assert Path(plotted["figure"].iloc[0]).name == (
        "subject_failure_layer_thickness.png"
    )
    assert (tmp_path / "subject_failure_layer_thickness.png").is_file()


if __name__ == "__main__":
    test_failure_groups_and_contrasts()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        test_target_specific_metric_loading(root)
        test_layer_thickness_loading_uses_raw_mm(root)
        test_target_figures_use_distinct_filenames(root)
        test_model_comparison_uses_one_reference_and_distinct_output(root)
        test_feature_comparison_adds_thickness_on_the_same_scale(root)
        test_failure_layer_thickness_uses_baseline_groups_and_raw_mm(root)
    print("stage2 subject-failure tests passed")
