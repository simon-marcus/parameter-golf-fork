#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
VERIFY_SCRIPT="$REPO_ROOT/verify_runpod_data_ready.sh"
SCRIPT_PATH="$REPO_ROOT/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py"
RECORD_ROOT="$REPO_ROOT/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate"
SUMMARY_SCRIPT="$REPO_ROOT/summarize_streaming_ttt_activation_suite.py"

ACTION="${1:-all}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"
SEEDS_CSV="${SEEDS:-1337,42,2025}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SUITE_ROOT="${SUITE_ROOT:-$RECORD_ROOT/bench_1gpu_circuit_scan}"

case "$DATA_ROOT_MODE" in
  workspace)
    export DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}"
    export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
    ;;
  tmp)
    export DATA_PATH="${DATA_PATH:-/tmp/parameter-golf-data/datasets/fineweb10B_sp1024}"
    export TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/parameter-golf-data/tokenizers/fineweb_1024_bpe.model}"
    ;;
  *)
    echo "DATA_ROOT_MODE must be one of: workspace, tmp"
    exit 1
    ;;
esac

ensure_data_ready() {
  "$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_PATH"
  mkdir -p "$SUITE_ROOT"
}

export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-120}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export USE_COMPILE="${USE_COMPILE:-0}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"
export VAL_TOKENS_LIMIT="${VAL_TOKENS_LIMIT:-1048576}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export MLP_MULT="${MLP_MULT:-3}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export SWA_ENABLED="${SWA_ENABLED:-1}"
export SWA_EVERY="${SWA_EVERY:-50}"
export ROPE_DIMS="${ROPE_DIMS:-16}"
export LN_SCALE="${LN_SCALE:-1}"
export LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.15}"
export VE_ENABLED="${VE_ENABLED:-1}"
export VE_DIM="${VE_DIM:-128}"
export VE_LAYERS="${VE_LAYERS:-9,10}"
export MUON_WD="${MUON_WD:-0.04}"
export ADAM_WD="${ADAM_WD:-0.04}"
export MATRIX_LR="${MATRIX_LR:-0.025}"
export SCALAR_LR="${SCALAR_LR:-0.025}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-400}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-600}"
export ITERATIONS="${ITERATIONS:-1200}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export TTT_ENABLED="${TTT_ENABLED:-1}"
export TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"
export TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"
export TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-16}"
export ACTIVATION_MODE="${ACTIVATION_MODE:-leaky_relu_sq}"
export ACTIVATION_NEG_SLOPE="${ACTIVATION_NEG_SLOPE:-0.5}"
export ASYMMETRIC_SQUARE_INIT="${ASYMMETRIC_SQUARE_INIT:-0.25}"
export GATED_SQUARE_BETA_INIT="${GATED_SQUARE_BETA_INIT:-1.0}"

CASE_NAMES=(
  chunk_full
  chunk_2_11
  chunk_4_11
  chunk_3_8
  chunk_4_9
  chunk_5_10
  chunk_6_11
  stream_3_8
  stream_4_9
  stream_5_10
  stream_6_11
)

configure_case() {
  local case_name="$1"

  unset RUN_ID SEED
  unset TTT_MODE TTT_PARAM_MODE TTT_FREEZE_BLOCKS TTT_BLOCK_START TTT_BLOCK_END
  unset TTT_LR TTT_EPOCHS TTT_CHUNK_TOKENS TTT_LR_SCHEDULE TTT_LAST_N_BLOCKS

  case "$case_name" in
    chunk_full)
      export TTT_MODE=chunk
      export TTT_PARAM_MODE=full
      export TTT_LR="${TTT_LR:-0.002}"
      export TTT_EPOCHS="${TTT_EPOCHS:-3}"
      export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
      export TTT_LR_SCHEDULE="${TTT_LR_SCHEDULE:-chunk_cosine}"
      ;;
    chunk_*)
      export TTT_MODE=chunk
      export TTT_PARAM_MODE=block_range
      export TTT_LR="${TTT_LR:-0.002}"
      export TTT_EPOCHS="${TTT_EPOCHS:-3}"
      export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
      export TTT_LR_SCHEDULE="${TTT_LR_SCHEDULE:-chunk_cosine}"
      ;;
    stream_*)
      export TTT_MODE=stream
      export TTT_PARAM_MODE=block_range
      export TTT_LR="${TTT_LR:-0.0005}"
      export TTT_EPOCHS="${TTT_EPOCHS:-1}"
      export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
      export TTT_LR_SCHEDULE="${TTT_LR_SCHEDULE:-constant}"
      ;;
    *)
      echo "Unknown case $case_name"
      exit 1
      ;;
  esac

  case "$case_name" in
    chunk_2_11) export TTT_BLOCK_START=2; export TTT_BLOCK_END=11 ;;
    chunk_4_11) export TTT_BLOCK_START=4; export TTT_BLOCK_END=11 ;;
    chunk_3_8) export TTT_BLOCK_START=3; export TTT_BLOCK_END=8 ;;
    chunk_4_9) export TTT_BLOCK_START=4; export TTT_BLOCK_END=9 ;;
    chunk_5_10) export TTT_BLOCK_START=5; export TTT_BLOCK_END=10 ;;
    chunk_6_11) export TTT_BLOCK_START=6; export TTT_BLOCK_END=11 ;;
    stream_3_8) export TTT_BLOCK_START=3; export TTT_BLOCK_END=8 ;;
    stream_4_9) export TTT_BLOCK_START=4; export TTT_BLOCK_END=9 ;;
    stream_5_10) export TTT_BLOCK_START=5; export TTT_BLOCK_END=10 ;;
    stream_6_11) export TTT_BLOCK_START=6; export TTT_BLOCK_END=11 ;;
  esac
}

run_case_seed() {
  local case_name="$1"
  local seed="$2"
  local case_dir="$SUITE_ROOT/$case_name/seed_$seed"
  local log_path="$case_dir/train.log"

  mkdir -p "$case_dir"
  configure_case "$case_name"
  export SEED="$seed"
  export RUN_ID="${case_name}_seed${seed}"

  {
    echo "=== case=$case_name seed=$seed started=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    env | grep -E '^(RUN_ID|SEED|TTT_|ACTIVATION_|MAX_WALLCLOCK_SECONDS|ITERATIONS|TRAIN_BATCH_TOKENS|VAL_BATCH_SIZE|VAL_TOKENS_LIMIT|TRAIN_SEQ_LEN|EVAL_SEQ_LEN|EVAL_STRIDE|NPROC_PER_NODE|DATA_PATH|TOKENIZER_PATH|USE_COMPILE)=' | sort
    echo "=== command ==="
    echo "python3 -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE $SCRIPT_PATH"
    echo "==============="
  } > "$log_path"

  (
    cd "$case_dir"
    python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_PATH"
  ) >> "$log_path" 2>&1
}

run_many() {
  local -a cases=("$@")
  IFS=',' read -r -a seeds <<< "$SEEDS_CSV"
  for case_name in "${cases[@]}"; do
    for seed in "${seeds[@]}"; do
      run_case_seed "$case_name" "$seed"
    done
  done
}

case "$ACTION" in
  all)
    ensure_data_ready
    run_many "${CASE_NAMES[@]}"
    python3 "$SUMMARY_SCRIPT" "$SUITE_ROOT"
    ;;
  summary)
    python3 "$SUMMARY_SCRIPT" "$SUITE_ROOT"
    ;;
  list)
    printf '%s\n' "${CASE_NAMES[@]}"
    ;;
  *)
    matched=0
    for case_name in "${CASE_NAMES[@]}"; do
      if [[ "$ACTION" == "$case_name" ]]; then
        matched=1
        ensure_data_ready
        run_many "$case_name"
        python3 "$SUMMARY_SCRIPT" "$SUITE_ROOT"
        break
      fi
    done
    if [[ "$matched" -eq 0 ]]; then
      echo "Usage: $0 {all|summary|list|${CASE_NAMES[*]}}"
      exit 1
    fi
    ;;
esac
