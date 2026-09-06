from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_remaining_stage2.py"
SPEC = importlib.util.spec_from_file_location("run_remaining_stage2", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_selected_test_ids_respects_range_and_limit() -> None:
    config = MODULE.Config()
    config.num_subjects = 5
    config.test_subject_start = 2
    config.test_subject_end = 4
    config.max_folds = 2
    assert MODULE.selected_test_ids(config) == frozenset({1, 2})


def test_audit_accepts_checkpoint_relocated_with_output_directory() -> None:
    with TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        checkpoint = output_dir / "best_model_fold_001_test_id_0.pth"
        checkpoint.touch()
        results = output_dir / "loso_results.csv"
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
                    "model_path": (
                        "artifacts/stage2/legacy_long_name/"
                        "best_model_fold_001_test_id_0.pth"
                    ),
                }
            )
        assert MODULE.completed_test_ids(results, output_dir) == frozenset({0})


if __name__ == "__main__":
    test_selected_test_ids_respects_range_and_limit()
    test_audit_accepts_checkpoint_relocated_with_output_directory()
    print("Remaining Stage 2 runner tests: OK")
