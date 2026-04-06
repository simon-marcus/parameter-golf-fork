#!/bin/bash
set -euo pipefail

DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
USE_COMPILE="${USE_COMPILE:-0}"

cd /workspace/parameter-golf

VERIFY_SCRIPT="/workspace/parameter-golf/verify_runpod_data_ready.sh"
SCRIPT_PATH="/workspace/parameter-golf/records/codex_scylla_2/train_gpt_legal_ttt.py"
SWEEP_ROOT="/workspace/parameter-golf/records/codex_scylla_2/runs"
WORKSPACE_BUNDLE_ROOT="${WORKSPACE_BUNDLE_ROOT:-/workspace/parameter-golf/scylla_v2_bundle}"
TMP_BUNDLE_ROOT="/tmp/parameter-golf-scylla-v2"
DATASET_NAME="${DATASET_NAME:-fineweb10B_scylla_v2_cap0_fullbyte}"
TOKENIZER_NAME="${TOKENIZER_NAME:-scylla_v2_cap0_fullbyte}"
EXPECTED_TRAIN_SHARDS="${EXPECTED_TRAIN_SHARDS:-11}"
EXPECTED_VAL_SHARDS="${EXPECTED_VAL_SHARDS:-1}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "FATAL: missing train script: $SCRIPT_PATH"
  exit 1
fi

case "$DATA_ROOT_MODE" in
  workspace)
    export DATA_PATH="$WORKSPACE_BUNDLE_ROOT/datasets/$DATASET_NAME"
    export TOKENIZER_PATH="$WORKSPACE_BUNDLE_ROOT/tokenizers/$TOKENIZER_NAME.yaml"
    export TOKENIZER_META_PATH="$WORKSPACE_BUNDLE_ROOT/tokenizers/$TOKENIZER_NAME.meta.npz"
    ;;
  tmp)
    mkdir -p "$TMP_BUNDLE_ROOT/datasets" "$TMP_BUNDLE_ROOT/tokenizers"
    rsync -a --delete "$WORKSPACE_BUNDLE_ROOT/datasets/$DATASET_NAME/" "$TMP_BUNDLE_ROOT/datasets/$DATASET_NAME/"
    rsync -a --delete "$WORKSPACE_BUNDLE_ROOT/tokenizers/" "$TMP_BUNDLE_ROOT/tokenizers/"
    export DATA_PATH="$TMP_BUNDLE_ROOT/datasets/$DATASET_NAME"
    export TOKENIZER_PATH="$TMP_BUNDLE_ROOT/tokenizers/$TOKENIZER_NAME.yaml"
    export TOKENIZER_META_PATH="$TMP_BUNDLE_ROOT/tokenizers/$TOKENIZER_NAME.meta.npz"
    ;;
  *)
    echo "DATA_ROOT_MODE must be one of: workspace, tmp"
    exit 1
    ;;
esac

"$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_META_PATH" "$EXPECTED_TRAIN_SHARDS" "$EXPECTED_VAL_SHARDS"

export VOCAB_SIZE=1254
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export NUM_LAYERS=11
export MLP_MULT=3
export BIGRAM_VOCAB_SIZE=6400
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
export TTT_MOMENTUM=0.9
export TTT_GRAD_CLIP=1.0
export NCCL_IB_DISABLE=1
export USE_COMPILE="$USE_COMPILE"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}"
export SEED="${SEED:-1337}"

# Format:
# NAME CHUNK_TOKENS LR EPOCHS FREEZE_BLOCKS FREEZE_EMBEDDINGS BATCH_SEQS PREQUANT_TTT XSA_LAST_N WARMDOWN_ITERS
declare -A CONFIGS
CONFIGS[C0]="34304 0.0015 3 2 0 32 0 4 3500"
CONFIGS[C1]="34304 0.0015 3 2 1 32 0 4 3500"
CONFIGS[C2]="65536 0.0015 3 2 1 32 0 4 3500"
CONFIGS[C3]="65536 0.0010 2 4 1 32 0 4 3500"
CONFIGS[C4]="98304 0.0010 2 4 1 32 0 4 3500"
CONFIGS[C5]="65536 0.0010 2 6 1 64 0 4 3500"
CONFIGS[P0]="34304 0.0015 3 2 0 32 1 4 3500"
CONFIGS[P1]="34304 0.0010 3 2 0 32 1 4 3500"
CONFIGS[P2]="34304 0.0007 3 2 0 32 1 4 3500"
CONFIGS[T0]="34304 0.0010 3 2 0 32 1 4 3500"
CONFIGS[T1]="34304 0.0010 3 2 0 32 1 11 4000"

run_config() {
  local NAME="$1"
  local CFG="${CONFIGS[$NAME]}"
  read -r CHUNK LR EPOCHS FREEZE_BLOCKS FREEZE_EMB BATCH_SEQS PREQUANT_TTT CFG_XSA_LAST_N CFG_WARMDOWN_ITERS <<< "$CFG"

  export TTT_CHUNK_TOKENS="$CHUNK"
  export TTT_LR="$LR"
  export TTT_EPOCHS="$EPOCHS"
  export TTT_FREEZE_BLOCKS="$FREEZE_BLOCKS"
  export TTT_FREEZE_EMBEDDINGS="$FREEZE_EMB"
  export TTT_BATCH_SEQS="$BATCH_SEQS"
  export TTT_USE_PREQUANT="$PREQUANT_TTT"
  export XSA_LAST_N="$CFG_XSA_LAST_N"
  export WARMDOWN_ITERS="$CFG_WARMDOWN_ITERS"
  export RUN_ID="codex_scylla_2_${NAME}"

  local OUT_DIR="$SWEEP_ROOT/$NAME"
  mkdir -p "$OUT_DIR"

  cat > "$OUT_DIR/config.json" <<EOCFG
{
  "name": "$NAME",
  "ttt_chunk_tokens": $CHUNK,
  "ttt_lr": $LR,
  "ttt_epochs": $EPOCHS,
  "ttt_freeze_blocks": $FREEZE_BLOCKS,
  "ttt_freeze_embeddings": $FREEZE_EMB,
  "ttt_batch_seqs": $BATCH_SEQS,
  "ttt_use_prequant": $PREQUANT_TTT,
  "xsa_last_n": $CFG_XSA_LAST_N,
  "warmdown_iters": $CFG_WARMDOWN_ITERS,
  "nproc_per_node": $NPROC_PER_NODE,
  "max_wallclock_seconds": $MAX_WALLCLOCK_SECONDS,
  "seed": $SEED
}
EOCFG

  echo "=== $NAME: chunk=$CHUNK lr=$LR epochs=$EPOCHS freeze_blocks=$FREEZE_BLOCKS freeze_emb=$FREEZE_EMB batch_seqs=$BATCH_SEQS prequant_ttt=$PREQUANT_TTT xsa_last_n=$CFG_XSA_LAST_N warmdown_iters=$CFG_WARMDOWN_ITERS ==="
  echo "  output: $OUT_DIR/train.log"

  python3 -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC_PER_NODE" \
    "$SCRIPT_PATH" \
    > "$OUT_DIR/train.log" 2>&1

  local EXIT_CODE=$?
  echo "  $NAME finished (exit=$EXIT_CODE)"
  return $EXIT_CODE
}

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "Usage: $0 <CONFIG_NAME|ALL>"
  echo "  Configs: ${!CONFIGS[*]}"
  exit 1
fi

if [[ "$TARGET" == "ALL" ]]; then
  for NAME in C0 C1 C2 C3 C4 C5 P0 P1 P2 T0 T1; do
    run_config "$NAME" || echo "WARNING: $NAME failed"
  done
  echo "=== All runs complete ==="
elif [[ -n "${CONFIGS[$TARGET]+x}" ]]; then
  run_config "$TARGET"
else
  echo "Unknown config: $TARGET"
  echo "Available: ${!CONFIGS[*]}"
  exit 1
fi
