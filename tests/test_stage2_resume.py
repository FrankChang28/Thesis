from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from subject_nirs.stage2.loso import completed_test_ids, global_fold_pairs


def test_subrange_keeps_global_fold_numbers() -> None:
    all_subjects = list(range(154))
    assert global_fold_pairs(all_subjects, [19, 20, 153]) == [
        (20, 19),
        (21, 20),
        (154, 153),
    ]


def test_resume_keeps_only_complete_correctly_numbered_rows() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "complete.pth"
        model.touch()
        results = root / "loso_results.csv"
        fieldnames = ["fold", "test_id", "test_loss", "model_path"]
        rows = [
            {"fold": 1, "test_id": 0, "test_loss": 0.1, "model_path": model},
            # Wrong fold number: this is the historical subrange-resume bug.
            {"fold": 1, "test_id": 19, "test_loss": 0.2, "model_path": model},
            # Missing model means the fold was not safely completed.
            {"fold": 3, "test_id": 2, "test_loss": 0.3, "model_path": root / "missing.pth"},
        ]
        with results.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        fold_by_test_id = {subject_id: subject_id + 1 for subject_id in range(154)}
        assert completed_test_ids(str(results), fold_by_test_id) == {0}
        with results.open("r", newline="", encoding="utf-8") as handle:
            normalized = list(csv.DictReader(handle))
        assert len(normalized) == 1
        assert int(normalized[0]["test_id"]) == 0
        assert results.with_name("loso_results.pre_resume_backup.csv").exists()


def test_resume_accepts_checkpoint_relocated_with_experiment_directory() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        relocated_model = root / "best_model_fold_001_test_id_0.pth"
        relocated_model.touch()
        results = root / "loso_results.csv"
        historical_model = (
            Path("artifacts/stage2/legacy_long_experiment_name")
            / relocated_model.name
        )
        with results.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["fold", "test_id", "test_loss", "model_path"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "fold": 1,
                    "test_id": 0,
                    "test_loss": 0.1,
                    "model_path": historical_model,
                }
            )

        assert completed_test_ids(str(results), {0: 1}) == {0}


if __name__ == "__main__":
    test_subrange_keeps_global_fold_numbers()
    test_resume_keeps_only_complete_correctly_numbered_rows()
    test_resume_accepts_checkpoint_relocated_with_experiment_directory()
    print("Stage 2 resume tests: OK")
