"""Run a Stage 2 LOSO experiment from one strict YAML configuration."""

from __future__ import annotations

import argparse

from subject_nirs.common.config import apply_dataclass_config, load_yaml
from subject_nirs.stage2.config import CFG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Stage 2 YAML file")
    args = parser.parse_args()
    apply_dataclass_config(CFG, load_yaml(args.config))

    # Import after applying the config so the workflow sees one authoritative CFG.
    from subject_nirs.stage2.loso import run_loso

    run_loso()


if __name__ == "__main__":
    main()
