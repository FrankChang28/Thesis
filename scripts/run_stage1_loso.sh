#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${SUBJECT_NIRS_PYTHON:-python}"
DATA_DIR="./data/raw/stage1"
CONFIG="./configs/stage1_32.yaml"

export PYTHONPATH="./src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

for TEST_ID in {0..153}; do
  OUTPUT_DIR="./artifacts/stage1/${TEST_ID}/relat_cons_32"

  "$PYTHON" scripts/train_stage1.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --test_id "$TEST_ID" \
    --amp \
    --skip_tsne

  "$PYTHON" scripts/export_subject_features.py \
    --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --data_dir "$DATA_DIR" \
    --save_path "$OUTPUT_DIR/subject_features_all_op.npz" \
    --amp
done
