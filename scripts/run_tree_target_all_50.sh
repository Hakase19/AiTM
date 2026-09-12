#!/usr/bin/env bash
# Preliminary Table-1-style AutoGen/AiTM target attacks:
# Tree topology, four datasets, 50 fixed-seed samples per dataset.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"

for DATASET in mmlu_bio mmlu_phy humaneval mbpp; do
  echo
  echo "============================================================"
  echo "Running Tree target attack: ${DATASET} (50 samples, seed 42)"
  echo "============================================================"

  "$PYTHON_BIN" -u scripts/run_experiments.py \
    --attack target \
    --datasets "$DATASET" \
    --structures tree \
    --samples 50 \
    --sample-seed 42 \
    --output "results/autogen_tree_${DATASET}_target_50.json"
done
