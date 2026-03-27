#!/bin/bash
set -euo pipefail

VARIANT="${1:-chunk_full}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd /workspace/parameter-golf

VERIFY_SCRIPT="/workspace/parameter-golf/verify_runpod_data_ready.sh"
SCRIPT_PATH="/workspace/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py"
RECORD_ROOT="/workspace/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate"

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
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}"
export TRAIN_LOG_EVERY=500
export VOCAB_SIZE=1024
export USE_COMPILE="${USE_COMPILE:-1}"
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
export ITERATIONS=9000
export EVAL_STRIDE=64
export TTT_ENABLED=1
export TTT_MOMENTUM=0.9
export TTT_GRAD_CLIP=1.0

case "$VARIANT" in
  audit_parity|chunk_full|chunk_baseline)
    if [[ "$VARIANT" == "chunk_full" || "$VARIANT" == "audit_parity" ]]; then
      export RUN_ID=streamttt_chunk_full
      OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_chunk_full"
    else
      export RUN_ID=streamttt_chunk_baseline
      OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_chunk_baseline"
    fi
    export TTT_PARAM_MODE=full
    export TTT_BLOCK_START=0
    export TTT_BLOCK_END=-1
    export TTT_LR=0.002
    export TTT_EPOCHS=3
    export TTT_CHUNK_TOKENS=32768
    export TTT_LR_SCHEDULE=chunk_cosine
    ;;
  audit_seed42|chunk_full_seed42)
    export RUN_ID=streamttt_chunk_full_seed42
    export SEED=42
    export TTT_PARAM_MODE=full
    export TTT_BLOCK_START=0
    export TTT_BLOCK_END=-1
    export TTT_LR=0.002
    export TTT_EPOCHS=3
    export TTT_CHUNK_TOKENS=32768
    export TTT_LR_SCHEDULE=chunk_cosine
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_chunk_full_seed42"
    ;;
  audit_seed2025|chunk_full_seed2025)
    export RUN_ID=streamttt_chunk_full_seed2025
    export SEED=2025
    export TTT_PARAM_MODE=full
    export TTT_BLOCK_START=0
    export TTT_BLOCK_END=-1
    export TTT_LR=0.002
    export TTT_EPOCHS=3
    export TTT_CHUNK_TOKENS=32768
    export TTT_LR_SCHEDULE=chunk_cosine
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_chunk_full_seed2025"
    ;;
  audit_lr98|chunk_full_lr98)
    export RUN_ID=streamttt_chunk_full_lr98
    export TTT_PARAM_MODE=full
    export TTT_BLOCK_START=0
    export TTT_BLOCK_END=-1
    export TTT_LR=0.002
    export TTT_EPOCHS=3
    export TTT_CHUNK_TOKENS=32768
    export TTT_LR_SCHEDULE=chunk_cosine
    export EMBED_LR=0.588
    export HEAD_LR=0.00784
    export TIED_EMBED_LR=0.0343
    export MATRIX_LR=0.0245
    export SCALAR_LR=0.0245
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_chunk_full_lr98"
    ;;
  audit_lateqat_0145|chunk_full_lateqat_0145)
    export RUN_ID=streamttt_chunk_full_lateqat_0145
    export TTT_PARAM_MODE=full
    export TTT_BLOCK_START=0
    export TTT_BLOCK_END=-1
    export TTT_LR=0.002
    export TTT_EPOCHS=3
    export TTT_CHUNK_TOKENS=32768
    export TTT_LR_SCHEDULE=chunk_cosine
    export LATE_QAT_THRESHOLD=0.145
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_chunk_full_lateqat_0145"
    ;;
  chunk_2_11)
    export RUN_ID=streamttt_chunk_2_11
    export TTT_PARAM_MODE=block_range
    export TTT_BLOCK_START=2
    export TTT_BLOCK_END=11
    export TTT_LR=0.002
    export TTT_EPOCHS=3
    export TTT_CHUNK_TOKENS=32768
    export TTT_LR_SCHEDULE=chunk_cosine
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_chunk_2_11"
    ;;
  chunk_midlate)
    export RUN_ID=streamttt_chunk_midlate
    export TTT_PARAM_MODE=block_range
    export TTT_BLOCK_START=4
    export TTT_BLOCK_END=11
    export TTT_LR=0.002
    export TTT_EPOCHS=3
    export TTT_CHUNK_TOKENS=32768
    export TTT_LR_SCHEDULE=chunk_cosine
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_chunk_midlate"
    ;;
  *)
    echo "Usage: $0 {chunk_full|audit_parity|audit_seed42|audit_seed2025|audit_lr98|audit_lateqat_0145|chunk_2_11|chunk_midlate|chunk_baseline}"
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

nohup python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_PATH" \
  > "$OUT_DIR/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo "started $RUN_ID data_root_mode=$DATA_ROOT_MODE pid=$PID log=$OUT_DIR/train.log"
