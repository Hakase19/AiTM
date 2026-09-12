#!/usr/bin/env bash
# Opt-in pilot: compares identical AiTM injection against target-selection
# policies on the controlled asymmetric Tree.  It does not alter Table-1 runs.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hhy/miniconda3/envs/aitm/bin/python}"
SAMPLES="${1:-20}"
SEED="${2:-42}"
OUT_DIR="${3:-results}"
COMMON=(
  -u scripts/run_experiments.py
  --attack target
  --datasets mmlu_bio
  --structures asymmetric_tree
  --samples "$SAMPLES"
  --sample-seed "$SEED"
)

"$PYTHON_BIN" "${COMMON[@]}" --method no_attack \
  --output "$OUT_DIR/aira_asymmetric_tree_no_attack_${SAMPLES}.json"
"$PYTHON_BIN" "${COMMON[@]}" --method aitm \
  --output "$OUT_DIR/aira_asymmetric_tree_aitm_fixed_${SAMPLES}.json"
"$PYTHON_BIN" "${COMMON[@]}" --method random_victim \
  --output "$OUT_DIR/aira_asymmetric_tree_aitm_random_${SAMPLES}.json"
"$PYTHON_BIN" "${COMMON[@]}" --method aira \
  --observation-events 9 \
  --role-inference llm \
  --output "$OUT_DIR/aira_asymmetric_tree_aira_${SAMPLES}.json"
