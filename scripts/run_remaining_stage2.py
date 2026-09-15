#!/usr/bin/env python3
"""Run only incomplete Stage 2 LOSO configurations, sequentially and resumably."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from subject_nirs.common.config import apply_dataclass_config, load_yaml  # noqa: E402
from subject_nirs.stage2.config import Config  # noqa: E402


@dataclass(frozen=True)
class ExperimentStatus:
    config_path: Path
    output_dir: Path
    completed_ids: frozenset[int]
    expected_ids: frozenset[int]

    @property
    def is_complete(self) -> bool:
        return self.expected_ids.issubset(self.completed_ids)

    @property
    def completed_count(self) -> int:
        return len(self.expected_ids.intersection(self.completed_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every Stage 2 YAML and sequentially run only incomplete "
            "LOSO experiments. Existing valid folds are resumed by train_stage2.py."
        )
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/stage2"),
        help="Directory recursively containing Stage 2 YAML files.",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=("hc", "sto2"),
        help="Run only this target. May be supplied more than once.",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("SUBJECT_NIRS_PYTHON", sys.executable),
        help="Python executable used to launch scripts/train_stage2.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the audit and pending queue without starting training.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later configs if one training command fails.",
    )
    return parser.parse_args()


def resolve_from_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def selected_test_ids(config: Config) -> frozenset[int]:
    all_ids = list(range(config.num_subjects))
    if config.test_subject_start is not None or config.test_subject_end is not None:
        if config.test_subject_start is None or config.test_subject_end is None:
            raise ValueError(
                "test_subject_start and test_subject_end must both be set"
            )
        start = int(config.test_subject_start) - 1
        end = int(config.test_subject_end) - 1
        selected = [subject_id for subject_id in all_ids if start <= subject_id <= end]
    elif config.run_all_loso:
        selected = all_ids
    else:
        selected = [int(config.test_subject_id)]

    if config.max_folds is not None:
        selected = selected[: config.max_folds]
    return frozenset(selected)


def checkpoint_exists(recorded: str, output_dir: Path) -> bool:
    if not recorded:
        return False
    recorded_path = Path(recorded)
    historical_path = resolve_from_repo(recorded_path)
    relocated_path = output_dir / recorded_path.name
    return historical_path.is_file() or relocated_path.is_file()


def completed_test_ids(results_path: Path, output_dir: Path) -> frozenset[int]:
    if not results_path.is_file() or results_path.stat().st_size == 0:
        return frozenset()

    valid: dict[int, dict[str, str]] = {}
    with results_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                test_id = int(row["test_id"])
                fold = int(row["fold"])
                test_loss = float(row["test_loss"])
                model_path = row["model_path"]
            except (KeyError, TypeError, ValueError):
                continue
            if fold != test_id + 1 or not math.isfinite(test_loss):
                continue
            if not checkpoint_exists(model_path, output_dir):
                continue
            valid[test_id] = row
    return frozenset(valid)


def load_status(config_path: Path) -> tuple[Config, ExperimentStatus]:
    config = Config()
    apply_dataclass_config(config, load_yaml(config_path))
    output_dir = resolve_from_repo(Path(config.output_dir))
    expected = selected_test_ids(config)
    completed = completed_test_ids(output_dir / "loso_results.csv", output_dir)
    return config, ExperimentStatus(config_path, output_dir, completed, expected)


def discover_statuses(config_root: Path, targets: set[str] | None) -> list[ExperimentStatus]:
    root = resolve_from_repo(config_root)
    config_paths = sorted(root.rglob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No Stage 2 YAML files found under {root}")

    statuses: list[tuple[int, str, ExperimentStatus]] = []
    seen_outputs: dict[Path, Path] = {}
    for config_path in config_paths:
        config, status = load_status(config_path)
        if config.execution_mode != "train":
            # Descriptor controls are explicit frozen-checkpoint evaluations,
            # not part of the remaining training matrix.
            continue
        if config.structure_control != "actual":
            # Training-time negative controls are launched explicitly with
            # run_stage2_loso.sh; do not mix them into the primary matrix queue.
            continue
        if targets is not None and config.target_mode not in targets:
            continue
        if not status.is_complete and not config.resume:
            raise ValueError(
                f"Pending config must set resume: true before batch execution: "
                f"{config_path}"
            )
        previous = seen_outputs.get(status.output_dir)
        if previous is not None:
            raise ValueError(
                f"Duplicate output directory {status.output_dir}: "
                f"{previous} and {config_path}"
            )
        seen_outputs[status.output_dir] = config_path
        priority = 0 if config.experiment_mode == "baseline" else 1
        statuses.append((priority, str(config_path), status))
    return [status for _, _, status in sorted(statuses)]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def print_audit(statuses: list[ExperimentStatus]) -> list[ExperimentStatus]:
    pending = []
    for status in statuses:
        state = "COMPLETE" if status.is_complete else "PENDING "
        print(
            f"[{state}] {status.completed_count:3d}/{len(status.expected_ids):3d}  "
            f"{display_path(status.config_path)} -> {display_path(status.output_dir)}"
        )
        if not status.is_complete:
            pending.append(status)
    print(
        f"\nAudit summary: {len(statuses) - len(pending)} complete, "
        f"{len(pending)} pending."
    )
    return pending


def run_pending(
    pending: list[ExperimentStatus], python: str, continue_on_error: bool
) -> int:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SRC_DIR)
        if not existing_pythonpath
        else os.pathsep.join((str(SRC_DIR), existing_pythonpath))
    )
    failures = []
    total = len(pending)
    for index, status in enumerate(pending, start=1):
        config_display = display_path(status.config_path)
        print(f"\n[{index}/{total}] Starting {config_display}", flush=True)
        command = [python, "scripts/train_stage2.py", "--config", config_display]
        result = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
        if result.returncode != 0:
            failures.append((config_display, result.returncode))
            print(
                f"FAILED ({result.returncode}): {config_display}",
                file=sys.stderr,
                flush=True,
            )
            if not continue_on_error:
                break
            continue

        _, refreshed = load_status(status.config_path)
        if not refreshed.is_complete:
            failures.append((config_display, 1))
            print(
                f"FAILED audit after exit 0: {refreshed.completed_count}/"
                f"{len(refreshed.expected_ids)} folds complete for {config_display}",
                file=sys.stderr,
                flush=True,
            )
            if not continue_on_error:
                break
        else:
            print(f"COMPLETE: {config_display}", flush=True)

    if failures:
        print("\nFailed configs:", file=sys.stderr)
        for config_path, returncode in failures:
            print(f"  returncode={returncode}: {config_path}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    targets = set(args.target) if args.target else None
    statuses = discover_statuses(args.config_root, targets)
    pending = print_audit(statuses)
    if args.dry_run or not pending:
        if args.dry_run:
            print("Dry run only; no training command was started.")
        return 0
    return run_pending(pending, args.python, args.continue_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
