#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
LOOP_PY="$ROOT_DIR/jepa_target_embedding_autoresearch_simple.py"

if [ -d "$ROOT_DIR/.venv/bin" ]; then
  export PATH="$ROOT_DIR/.venv/bin:$PATH"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODE_RAW="${1:-${AUTORESEARCH_MUTATION_MODE:-constrained}}"
MODE="$(printf '%s' "$MODE_RAW" | tr '[:upper:]' '[:lower:]')"
case "$MODE" in
  constrained|strict)
    MUTATION_MODE="constrained"
    DEFAULT_NAMESPACE="jepa_target_embedding_discovery"
    DEFAULT_PROGRAM_PATH="$ROOT_DIR/records/byte-level-jepa/target-embedding-autoresearch/program.md"
    ;;
  freecode|free_code|open)
    MUTATION_MODE="freecode"
    DEFAULT_NAMESPACE="jepa_target_embedding_freecode_discovery"
    DEFAULT_PROGRAM_PATH="$ROOT_DIR/records/byte-level-jepa/target-embedding-autoresearch/program_freecode.md"
    ;;
  *)
    echo "Usage: $0 [constrained|freecode]" >&2
    echo "  constrained: narrow mutation surface (TargetPatchEncoder +/- Predictor)" >&2
    echo "  freecode: broader probe-level coding surface with guardrails" >&2
    exit 1
    ;;
esac

NAMESPACE="${AUTORESEARCH_NAMESPACE:-$DEFAULT_NAMESPACE}"
WORK_DIR="$ROOT_DIR/autoresearch/$NAMESPACE/work"
WORK_TRAIN_SCRIPT="$WORK_DIR/target_embedding_probe.py"
PROGRAM_PATH="${AUTORESEARCH_PROGRAM_FILE:-$DEFAULT_PROGRAM_PATH}"
SOURCE_TRAIN_SCRIPT="$ROOT_DIR/records/byte-level-jepa/target-embedding-autoresearch/target_embedding_probe.py"
BACKGROUND="${BACKGROUND:-0}"
RESET_WORKTREE="${RESET_WORKTREE:-0}"

mkdir -p "$WORK_DIR"

if [ "$RESET_WORKTREE" = "1" ] || [ ! -f "$WORK_TRAIN_SCRIPT" ]; then
  cp "$SOURCE_TRAIN_SCRIPT" "$WORK_TRAIN_SCRIPT"
fi

if [ ! -f "$PROGRAM_PATH" ]; then
  echo "Missing program file: $PROGRAM_PATH" >&2
  exit 1
fi

COMMON_ENV=(
  "AUTORESEARCH_TRAIN_SCRIPT=$WORK_TRAIN_SCRIPT"
  "AUTORESEARCH_SEED_SCRIPT=$SOURCE_TRAIN_SCRIPT"
  "AUTORESEARCH_PROGRAM_FILE=$PROGRAM_PATH"
  "AUTORESEARCH_LANE=${AUTORESEARCH_LANE:-representation}"
  "AUTORESEARCH_STAGE=${AUTORESEARCH_STAGE:-discovery}"
  "AUTORESEARCH_NAMESPACE=$NAMESPACE"
  "REPRESENTATION_VERIFIED=${REPRESENTATION_VERIFIED:-1}"
  "GPUS=${GPUS:-1}"
  "EXPERIMENT_SECONDS=${EXPERIMENT_SECONDS:-180}"
  "MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-20}"
  "AUTORESEARCH_MODEL=${AUTORESEARCH_MODEL:-claude-sonnet-4-6}"
  "CLAUDE_EFFORT=${CLAUDE_EFFORT:-medium}"
  "AUTORESEARCH_ALLOWED_TOOLS=${AUTORESEARCH_ALLOWED_TOOLS:-Bash,Read,Edit}"
  "AUTORESEARCH_PERMISSION_MODE=${AUTORESEARCH_PERMISSION_MODE:-bypassPermissions}"
  "PROPOSAL_TIMEOUT_SECONDS=${PROPOSAL_TIMEOUT_SECONDS:-240}"
  "TRAIN_TIMEOUT_PADDING_SECONDS=${TRAIN_TIMEOUT_PADDING_SECONDS:-300}"
  "VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}"
  "DATA_PATH=${DATA_PATH:-./data/datasets/fineweb10B_byte260}"
  "TOKENIZER_PATH=${TOKENIZER_PATH:-./data/tokenizers/fineweb_pure_byte_260.json}"
  "PROBE_USE_SYNTHETIC_DATA=${PROBE_USE_SYNTHETIC_DATA:-0}"
)

echo "Launching JEPA target-embedding autoresearch"
echo "  Mutation mode:  $MUTATION_MODE"
echo "  Namespace:      $NAMESPACE"
echo "  Work script:    $WORK_TRAIN_SCRIPT"
echo "  Program:        $PROGRAM_PATH"
echo "  GPUS:           ${GPUS:-1}"
echo "  Seconds/run:    ${EXPERIMENT_SECONDS:-180}"

if [ "$BACKGROUND" = "1" ]; then
  LOG_FILE="$ROOT_DIR/autoresearch/$NAMESPACE/autoresearch.out"
  env "${COMMON_ENV[@]}" nohup "$PYTHON_BIN" -u "$LOOP_PY" >"$LOG_FILE" 2>&1 &
  echo "Started in background: $!"
  echo "Log: $LOG_FILE"
else
  exec env "${COMMON_ENV[@]}" "$PYTHON_BIN" -u "$LOOP_PY"
fi
