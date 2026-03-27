#!/bin/bash
set -euo pipefail

VARIANT="${1:-base}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"

cd /workspace/parameter-golf

VERIFY_SCRIPT="/workspace/parameter-golf/verify_runpod_data_ready.sh"

SCRIPT_PATH="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/train_gpt.py"
RECORD_ROOT="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix"

case "$DATA_ROOT_MODE" in
  workspace)
    export DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
    export TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
    ;;
  tmp)
    export DATA_PATH=/tmp/parameter-golf-data/datasets/fineweb10B_sp1024
    export TOKENIZER_PATH=/tmp/parameter-golf-data/tokenizers/fineweb_1024_bpe.model
    ;;
  *)
    echo "DATA_ROOT_MODE must be one of: workspace, tmp"
    exit 1
    ;;
esac

"$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_PATH"

export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=200
export VOCAB_SIZE=1024
export INT8_KEEP_TOK_EMB_FP16=1
export USE_COMPILE="${USE_COMPILE:-1}"
export PAID_PREFIX_CODEC=raw

case "$VARIANT" in
  base)
    export RUN_ID=leadercore10l_paidprefix_base
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_base"
    ;;
  prefix690k)
    export RUN_ID=leadercore10l_paidprefix_690k
    export PAID_PREFIX_FILE="$RECORD_ROOT/prefix_690000.raw"
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_prefix690k"
    ;;
  prefix680k)
    export RUN_ID=leadercore10l_paidprefix_680k
    export PAID_PREFIX_FILE="$RECORD_ROOT/prefix_680000.raw"
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_prefix680k"
    ;;
  prefix700k)
    export RUN_ID=leadercore10l_paidprefix_700k
    export PAID_PREFIX_FILE="$RECORD_ROOT/prefix_700000.raw"
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_prefix700k"
    ;;
  prefix700k_tokembint8)
    export RUN_ID=leadercore10l_paidprefix_700k_tokembint8
    export PAID_PREFIX_FILE="$RECORD_ROOT/prefix_700000.raw"
    export INT8_KEEP_TOK_EMB_FP16=0
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_prefix700k_tokembint8"
    ;;
  prefix720k)
    export RUN_ID=leadercore10l_paidprefix_720k
    export PAID_PREFIX_FILE="$RECORD_ROOT/prefix_720000.raw"
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_prefix720k"
    ;;
  prefix768k)
    export RUN_ID=leadercore10l_paidprefix_768k
    export PAID_PREFIX_FILE="$RECORD_ROOT/prefix_768000.raw"
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_prefix768k"
    ;;
  *)
    echo "Usage: $0 {base|prefix680k|prefix690k|prefix700k|prefix700k_tokembint8|prefix720k|prefix768k}"
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

if [ -n "${PAID_PREFIX_FILE:-}" ] && [ ! -s "$PAID_PREFIX_FILE" ]; then
  echo "missing or empty paid prefix file: $PAID_PREFIX_FILE" >&2
  exit 1
fi

nohup python3 -m torch.distributed.run --standalone --nproc_per_node=8 "$SCRIPT_PATH" \
  > "$OUT_DIR/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo "started $RUN_ID data_root_mode=$DATA_ROOT_MODE pid=$PID log=$OUT_DIR/train.log"
