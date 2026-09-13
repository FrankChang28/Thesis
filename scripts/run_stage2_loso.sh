#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${SUBJECT_NIRS_PYTHON:-python}"
CONFIG="${1:-./configs/stage2/residual/scratch/metadata_sto2.yaml}"

export PYTHONPATH="./src"

"$PYTHON" scripts/train_stage2.py --config "$CONFIG"
