#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${SUBJECT_NIRS_PYTHON:-python}"
START_ID="${1:-0}"
END_ID="${2:-153}"

export PYTHONPATH="./src"

for TEST_ID in $(seq "$START_ID" "$END_ID"); do
  FOLD_DIR="artifacts/stage1/${TEST_ID}/relat_cons_32"
  CHECKPOINT="${FOLD_DIR}/best_model.pt"
  OUTPUT="${FOLD_DIR}/test_acquisition_embeddings.npz"
  if [[ -s "$OUTPUT" ]]; then
    echo "test_id=${TEST_ID}: cached"
    continue
  fi
  "$PYTHON" scripts/export_subject_features.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir data/raw/stage1 \
    --save_path "${FOLD_DIR}/subject_features_all_op.npz" \
    --test_acquisition_save_path "$OUTPUT" \
    --skip_descriptor_save \
    --amp
done
