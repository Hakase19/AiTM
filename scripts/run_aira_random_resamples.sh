#!/usr/bin/env bash
# Re-sample only the random-victim baseline while keeping the task subset and
# topology seed fixed.  Outputs are separate from the earlier pilot files.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hhy/miniconda3/envs/aitm/bin/python}"
SAMPLES="${1:-20}"
SEED="${2:-42}"
OUT_DIR="${3:-results}"
OFFSETS=(101 202 303)

for index in "${!OFFSETS[@]}"; do
  repetition=$((index + 1))
  "$PYTHON_BIN" -u scripts/run_experiments.py \
    --method random_victim \
    --random-victim-seed-offset "${OFFSETS[$index]}" \
    --attack target \
    --datasets mmlu_bio \
    --structures asymmetric_tree \
    --samples "$SAMPLES" \
    --sample-seed "$SEED" \
    --output "$OUT_DIR/aira_asymmetric_tree_aitm_random_resample_${SAMPLES}_rep${repetition}.json"
done
