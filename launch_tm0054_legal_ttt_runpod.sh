#!/bin/bash
set -euo pipefail

DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd /workspace/parameter-golf

VERIFY_SCRIPT="/workspace/parameter-golf/verify_runpod_data_ready.sh"
SCRIPT_PATH="/workspace/parameter-golf/records/track_10min_16mb/2026-03-29_PPM_LOO_TokenizerBundle/train_gpt.py"
FALLBACK_SCRIPT_PATH="/workspace/parameter-golf/train_gpt.py"
RECORD_ROOT="/workspace/parameter-golf/records/track_10min_16mb/2026-03-30_tm0054_legal_ttt_runpod"
WORKSPACE_BUNDLE_ROOT="${WORKSPACE_BUNDLE_ROOT:-/workspace/parameter-golf/tm0054_bundle}"
TMP_BUNDLE_ROOT="/tmp/parameter-golf-tm0054"
EXPECTED_TRAIN_SHARDS="${EXPECTED_TRAIN_SHARDS:-10}"
EXPECTED_VAL_SHARDS="${EXPECTED_VAL_SHARDS:-1}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  if [[ -f "$FALLBACK_SCRIPT_PATH" ]]; then
    mkdir -p "$(dirname "$SCRIPT_PATH")"
    cp "$FALLBACK_SCRIPT_PATH" "$SCRIPT_PATH"
  else
    echo "missing train script: $SCRIPT_PATH (fallback $FALLBACK_SCRIPT_PATH also missing)"
    exit 1
  fi
fi

case "$DATA_ROOT_MODE" in
  workspace)
    export DATA_PATH="$WORKSPACE_BUNDLE_ROOT/datasets/fineweb10B_tm0054"
    export TOKENIZER_PATH="$WORKSPACE_BUNDLE_ROOT/tokenizers/candidate.vocab"
    export TOKENIZER_META_PATH="$WORKSPACE_BUNDLE_ROOT/tokenizers/candidate.meta.npz"
    ;;
  tmp)
    mkdir -p "$TMP_BUNDLE_ROOT/datasets" "$TMP_BUNDLE_ROOT/tokenizers"
    rsync -a --delete "$WORKSPACE_BUNDLE_ROOT/datasets/fineweb10B_tm0054/" "$TMP_BUNDLE_ROOT/datasets/fineweb10B_tm0054/"
    rsync -a --delete "$WORKSPACE_BUNDLE_ROOT/tokenizers/" "$TMP_BUNDLE_ROOT/tokenizers/"
    export DATA_PATH="$TMP_BUNDLE_ROOT/datasets/fineweb10B_tm0054"
    export TOKENIZER_PATH="$TMP_BUNDLE_ROOT/tokenizers/candidate.vocab"
    export TOKENIZER_META_PATH="$TMP_BUNDLE_ROOT/tokenizers/candidate.meta.npz"
    ;;
  *)
    echo "DATA_ROOT_MODE must be one of: workspace, tmp"
    exit 1
    ;;
esac

"$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_META_PATH" "$EXPECTED_TRAIN_SHARDS" "$EXPECTED_VAL_SHARDS"

export RUN_ID="${RUN_ID:-tm0054_legal_ttt_runpod}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export VOCAB_SIZE=998
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export NUM_LAYERS=11
export MLP_MULT=3
export BIGRAM_VOCAB_SIZE=1536
export XSA_LAST_N=4
export SWA_ENABLED=1
export SWA_EVERY=50
export ROPE_DIMS=16
export LN_SCALE=1
export LATE_QAT_THRESHOLD=0.15
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS=9,10
export MUON_WD=0.04
export ADAM_WD=0.04
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export EMBED_LR=0.6
export HEAD_LR=0.008
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3500
export ITERATIONS="${ITERATIONS:-9000}"
export EVAL_STRIDE=64
export TTT_ENABLED=1
export NGRAM_ENABLED=0
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_MOMENTUM=0.9
export TTT_GRAD_CLIP=1.0
export NCCL_IB_DISABLE=1

OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_seed_${SEED:-1337}"
mkdir -p "$OUT_DIR"

nohup python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_PATH" \
  > "$OUT_DIR/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo "started $RUN_ID data_root_mode=$DATA_ROOT_MODE pid=$PID log=$OUT_DIR/train.log"
