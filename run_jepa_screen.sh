#!/bin/bash
set -euo pipefail

VARIANT="${1:-control}"
SCREEN_SECONDS="${SCREEN_SECONDS:-180}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

case "$VARIANT" in
  control)
    DEFAULT_JEPA_LOSS_WEIGHT="0.0"
    ;;
  jepa01)
    DEFAULT_JEPA_LOSS_WEIGHT="0.1"
    ;;
  jepa02)
    DEFAULT_JEPA_LOSS_WEIGHT="0.2"
    ;;
  patch16)
    DEFAULT_JEPA_LOSS_WEIGHT="0.2"
    DEFAULT_PATCH_SIZE="16"
    ;;
  ema099)
    DEFAULT_JEPA_LOSS_WEIGHT="0.2"
    DEFAULT_TARGET_EMA="0.99"
    ;;
  *)
    echo "Usage: $0 {control|jepa01|jepa02|patch16|ema099}" >&2
    exit 1
    ;;
esac

if [ -z "${DATA_PATH:-}" ]; then
  export DATA_PATH="$ROOT_DIR/data/datasets/fineweb10B_byte260"
fi

if [ ! -d "$DATA_PATH" ]; then
  echo "missing DATA_PATH: $DATA_PATH" >&2
  echo "download byte-level data first: python3 data/cached_challenge_fineweb.py --variant byte260 --train-shards 1" >&2
  exit 1
fi

export TOKENIZER_KIND=byte
export TOKENIZER_PATH=""
export TOKENIZER_META_PATH=""
export VOCAB_SIZE="${VOCAB_SIZE:-260}"
export USE_COMPILE="${USE_COMPILE:-0}"
export MAX_WALLCLOCK_SECONDS="$SCREEN_SECONDS"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-25}"
export RUN_ID="${RUN_ID:-jepa_screen_${VARIANT}}"
export JEPA_LOSS_WEIGHT="${JEPA_LOSS_WEIGHT:-$DEFAULT_JEPA_LOSS_WEIGHT}"

if [ -n "${DEFAULT_PATCH_SIZE:-}" ]; then
  export PATCH_SIZE="${PATCH_SIZE:-$DEFAULT_PATCH_SIZE}"
fi
if [ -n "${DEFAULT_TARGET_EMA:-}" ]; then
  export TARGET_EMA="${TARGET_EMA:-$DEFAULT_TARGET_EMA}"
fi

echo "running $RUN_ID with $PYTHON_BIN"
echo "  DATA_PATH=$DATA_PATH"
echo "  SCREEN_SECONDS=$SCREEN_SECONDS"
echo "  VARIANT=$VARIANT"

exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" train_jepa.py
