#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

CASE="${1:-}"
LOAD_FP32_PATH="${2:-${LOAD_FP32_PATH:-}}"

if [[ -z "$CASE" || -z "$LOAD_FP32_PATH" ]]; then
  echo "Usage: $0 <case> <final_model.pt>"
  echo "Example: $0 ladder_12288 /path/to/final_model.pt"
  exit 1
fi

export EVAL_ONLY=1
unset LOAD_INT6_PATH || true
export LOAD_FP32_PATH
export SUITE_ROOT="${SUITE_ROOT:-$REPO_ROOT/records/scylla_2_claude/checkpoint_reexport_eval_runs}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export SEEDS="${SEEDS:-2027}"
export USE_COMPILE="${USE_COMPILE:-0}"

bash "$REPO_ROOT/launch_scylla_frontier_plan.sh" "$CASE"
