#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
VERIFY_SCRIPT="$REPO_ROOT/verify_runpod_data_ready.sh"
TRAIN_SCRIPT="$REPO_ROOT/records/scylla_2_claude/train_gpt_legal_ttt.py"

CASE="${1:-}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
SEEDS_CSV="${SEEDS:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SUITE_ROOT="${SUITE_ROOT:-$REPO_ROOT/records/scylla_2_claude/frontier_runs}"

if [[ -z "$CASE" ]]; then
  echo "Usage: $0 {phase1_all|phase2_all|phase3_all|control_1254_legacy|frontier_1254|frontier_1254_loop|ablate_1254_no_bigram|ablate_1254_no_ve|ablate_1254_no_bigram_no_ve|ladder_1254|ladder_1536|ladder_2048|ladder_3072}"
  exit 1
fi

resolve_scylla_paths() {
  local vocab="$1"
  local data_var="SCYLLA_V${vocab}_DATA_PATH"
  local tok_var="SCYLLA_V${vocab}_TOKENIZER_PATH"
  local meta_var="SCYLLA_V${vocab}_TOKENIZER_META_PATH"
  local shards_var="SCYLLA_V${vocab}_EXPECTED_TRAIN_SHARDS"
  local vals_var="SCYLLA_V${vocab}_EXPECTED_VAL_SHARDS"
  local data_override="${!data_var:-}"
  local tok_override="${!tok_var:-}"
  local meta_override="${!meta_var:-}"

  if [[ -n "$data_override" && -n "$tok_override" && -n "$meta_override" ]]; then
    export DATA_PATH="$data_override"
    export TOKENIZER_PATH="$tok_override"
    export TOKENIZER_META_PATH="$meta_override"
    export EXPECTED_TRAIN_SHARDS="${!shards_var:-11}"
    export EXPECTED_VAL_SHARDS="${!vals_var:-1}"
    return
  fi

  local dataset_name tokenizer_name
  if [[ "$vocab" == "1254" ]]; then
    dataset_name="${SCYLLA_1254_DATASET_NAME:-fineweb10B_scylla_v2_cap0_fullbyte}"
    tokenizer_name="${SCYLLA_1254_TOKENIZER_NAME:-scylla_v2_cap0_fullbyte}"
  else
    local dataset_template="${SCYLLA_DATASET_TEMPLATE:-fineweb10B_scylla_v2_v{vocab}}"
    local tokenizer_template="${SCYLLA_TOKENIZER_TEMPLATE:-scylla_v2_v{vocab}}"
    dataset_name="${dataset_template//\{vocab\}/$vocab}"
    tokenizer_name="${tokenizer_template//\{vocab\}/$vocab}"
  fi

  case "$DATA_ROOT_MODE" in
    workspace)
      local bundle_root="${SCYLLA_BUNDLE_ROOT:-$REPO_ROOT/scylla_v2_bundle}"
      export DATA_PATH="$bundle_root/datasets/$dataset_name"
      export TOKENIZER_PATH="$bundle_root/tokenizers/$tokenizer_name.yaml"
      export TOKENIZER_META_PATH="$bundle_root/tokenizers/$tokenizer_name.meta.npz"
      ;;
    tmp)
      local bundle_root="${SCYLLA_TMP_BUNDLE_ROOT:-/tmp/parameter-golf-data}"
      export DATA_PATH="$bundle_root/datasets/$dataset_name"
      export TOKENIZER_PATH="$bundle_root/tokenizers/$tokenizer_name.yaml"
      export TOKENIZER_META_PATH="$bundle_root/tokenizers/$tokenizer_name.meta.npz"
      ;;
    *)
      echo "DATA_ROOT_MODE must be workspace or tmp"
      exit 1
      ;;
  esac
  export EXPECTED_TRAIN_SHARDS="${!shards_var:-11}"
  export EXPECTED_VAL_SHARDS="${!vals_var:-1}"
}

ensure_data_ready() {
  local vocab="$1"
  resolve_scylla_paths "$vocab"
  "$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_META_PATH" "$EXPECTED_TRAIN_SHARDS" "$EXPECTED_VAL_SHARDS"
  mkdir -p "$SUITE_ROOT"
}

set_common_env() {
  export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}"
  export ITERATIONS="${ITERATIONS:-1200}"
  export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
  export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
  export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
  export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
  export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-262144}"
  export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"
  export VAL_TOKENS_LIMIT="${VAL_TOKENS_LIMIT:-1048576}"
  export NUM_LAYERS="${NUM_LAYERS:-11}"
  export MODEL_DIM="${MODEL_DIM:-512}"
  export NUM_HEADS="${NUM_HEADS:-8}"
  export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
  export ROPE_DIMS="${ROPE_DIMS:-16}"
  export LN_SCALE="${LN_SCALE:-1}"
  export SWA_ENABLED="${SWA_ENABLED:-1}"
  export SWA_EVERY="${SWA_EVERY:-50}"
  export LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.0}"
  export TTT_ENABLED=1
  export NGRAM_ENABLED=0
  export CAUSAL_CACHE_ENABLED=0
  export COMPLEMENT_ENABLED=0
  export USE_COMPILE="${USE_COMPILE:-0}"
  export EVAL_STRIDE="${EVAL_STRIDE:-64}"
  export TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-32}"
  export TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"
  export TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"
  export TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-2}"
  export TTT_FREEZE_EMBEDDINGS="${TTT_FREEZE_EMBEDDINGS:-0}"
  export TTT_EPOCHS="${TTT_EPOCHS:-3}"
}

apply_legacy_1254_defaults() {
  set_common_env
  export VOCAB_SIZE=1254
  export QK_GAIN_INIT=1.5
  export XSA_LAST_N=4
  export MLP_MULT=3
  export BIGRAM_VOCAB_SIZE=6400
  export VE_ENABLED=1
  export VE_DIM=128
  export VE_LAYERS=9,10
  export SMEAR_ENABLED=1
  export NUM_LOOPS=0
  export LOOP_START=4
  export LOOP_END=5
  export ENABLE_LOOPING_AT=0.5
  export MUON_WD=0.04
  export EMBED_WD=0.04
  export ADAM_WD=0.04
  export EMBED_LR=0.6
  export HEAD_LR=0.008
  export MATRIX_LR=0.025
  export SCALAR_LR=0.025
  export TIED_EMBED_LR=0.035
  export MUON_MOMENTUM=0.99
  export MUON_MOMENTUM_WARMUP_START=0.92
  export MUON_MOMENTUM_WARMUP_STEPS=400
  export WARMDOWN_ITERS=600
  export COMPRESSOR=lzma
  export BYTE_SHUFFLE=0
  export MATRIX_BITS=6
  export EMBED_BITS=8
  export MATRIX_CLIP_MODE=quantile
  export EMBED_CLIP_MODE=quantile
  export MATRIX_CLIP_SIGMAS=12.85
  export EMBED_CLIP_SIGMAS=20.0
  export TTT_LR=0.0015
  export TTT_CHUNK_TOKENS=34304
  export TTT_CHUNK_BYTES=0
}

apply_frontier_defaults() {
  set_common_env
  export VOCAB_SIZE=1254
  export QK_GAIN_INIT=5.0
  export XSA_LAST_N=11
  export MLP_MULT=4
  export BIGRAM_VOCAB_SIZE=6400
  export VE_ENABLED=1
  export VE_DIM=128
  export VE_LAYERS=9,10
  export SMEAR_ENABLED=0
  export NUM_LOOPS=0
  export LOOP_START=4
  export LOOP_END=5
  export ENABLE_LOOPING_AT=0.5
  export MUON_WD=0.085
  export EMBED_WD=0.085
  export ADAM_WD=0.02
  export EMBED_LR=0.6
  export HEAD_LR=0.008
  export MATRIX_LR=0.02
  export SCALAR_LR=0.02
  export TIED_EMBED_LR=0.03
  export MUON_MOMENTUM=0.99
  export MUON_MOMENTUM_WARMUP_START=0.92
  export MUON_MOMENTUM_WARMUP_STEPS=400
  export WARMDOWN_ITERS=900
  export COMPRESSOR=brotli
  export BYTE_SHUFFLE=1
  export MATRIX_BITS=6
  export EMBED_BITS=8
  export MATRIX_CLIP_MODE=std
  export EMBED_CLIP_MODE=std
  export MATRIX_CLIP_SIGMAS=12.85
  export EMBED_CLIP_SIGMAS=20.0
  export TTT_LR=0.0010
  export TTT_CHUNK_TOKENS=0
  export TTT_CHUNK_BYTES=131072
}

apply_case() {
  local case_name="$1"
  PHASE="unknown"
  DESCRIPTION=""
  case "$case_name" in
    control_1254_legacy)
      PHASE="phase1"
      DESCRIPTION="Legacy 1254 Scylla control"
      apply_legacy_1254_defaults
      ;;
    frontier_1254)
      PHASE="phase1"
      DESCRIPTION="1254 frontier-style port without recurrence"
      apply_frontier_defaults
      ;;
    frontier_1254_loop)
      PHASE="phase1"
      DESCRIPTION="1254 frontier-style port with depth recurrence"
      apply_frontier_defaults
      export NUM_LOOPS=2
      ;;
    ablate_1254_no_bigram)
      PHASE="phase2"
      DESCRIPTION="1254 frontier stack with bigram removed"
      apply_frontier_defaults
      export NUM_LOOPS=2
      export BIGRAM_VOCAB_SIZE=0
      ;;
    ablate_1254_no_ve)
      PHASE="phase2"
      DESCRIPTION="1254 frontier stack with value embeddings removed"
      apply_frontier_defaults
      export NUM_LOOPS=2
      export VE_ENABLED=0
      ;;
    ablate_1254_no_bigram_no_ve)
      PHASE="phase2"
      DESCRIPTION="1254 frontier stack with bigram and value embeddings removed"
      apply_frontier_defaults
      export NUM_LOOPS=2
      export BIGRAM_VOCAB_SIZE=0
      export VE_ENABLED=0
      ;;
    ladder_1254)
      PHASE="phase3"
      DESCRIPTION="Vocab ladder baseline at 1254"
      apply_frontier_defaults
      export NUM_LOOPS=2
      export BIGRAM_VOCAB_SIZE=0
      export VE_ENABLED=0
      export VOCAB_SIZE=1254
      ;;
    ladder_1536)
      PHASE="phase3"
      DESCRIPTION="Vocab ladder candidate at 1536"
      apply_frontier_defaults
      export NUM_LOOPS=2
      export BIGRAM_VOCAB_SIZE=0
      export VE_ENABLED=0
      export VOCAB_SIZE=1536
      ;;
    ladder_2048)
      PHASE="phase3"
      DESCRIPTION="Vocab ladder candidate at 2048"
      apply_frontier_defaults
      export NUM_LOOPS=2
      export BIGRAM_VOCAB_SIZE=0
      export VE_ENABLED=0
      export VOCAB_SIZE=2048
      ;;
    ladder_3072)
      PHASE="phase3"
      DESCRIPTION="Vocab ladder candidate at 3072"
      apply_frontier_defaults
      export NUM_LOOPS=2
      export BIGRAM_VOCAB_SIZE=0
      export VE_ENABLED=0
      export VOCAB_SIZE=3072
      ;;
    *)
      echo "Unknown case: $case_name"
      exit 1
      ;;
  esac
}

case_list() {
  case "$1" in
    phase1_all)
      echo "control_1254_legacy frontier_1254 frontier_1254_loop"
      ;;
    phase2_all)
      echo "ablate_1254_no_bigram ablate_1254_no_ve ablate_1254_no_bigram_no_ve"
      ;;
    phase3_all)
      echo "ladder_1254 ladder_1536 ladder_2048 ladder_3072"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

run_case_seed() {
  local case_name="$1"
  local seed="$2"
  apply_case "$case_name"
  ensure_data_ready "$VOCAB_SIZE"

  local case_dir="$SUITE_ROOT/$PHASE/$case_name/seed_$seed"
  local log_path="$case_dir/train.log"
  mkdir -p "$case_dir"

  export SEED="$seed"
  export RUN_ID="scylla_frontier_${case_name}_seed${seed}"

  cat > "$case_dir/config.json" <<EOF
{
  "phase": "$PHASE",
  "case_name": "$case_name",
  "description": "$DESCRIPTION",
  "seed": $seed,
  "data_path": "$DATA_PATH",
  "tokenizer_path": "$TOKENIZER_PATH",
  "tokenizer_meta_path": "$TOKENIZER_META_PATH",
  "vocab_size": $VOCAB_SIZE,
  "iterations": $ITERATIONS,
  "train_batch_tokens": $TRAIN_BATCH_TOKENS,
  "val_tokens_limit": $VAL_TOKENS_LIMIT,
  "qk_gain_init": $QK_GAIN_INIT,
  "xsa_last_n": $XSA_LAST_N,
  "mlp_mult": $MLP_MULT,
  "bigram_vocab_size": $BIGRAM_VOCAB_SIZE,
  "ve_enabled": $VE_ENABLED,
  "smear_enabled": $SMEAR_ENABLED,
  "num_loops": $NUM_LOOPS,
  "loop_start": $LOOP_START,
  "loop_end": $LOOP_END,
  "enable_looping_at": $ENABLE_LOOPING_AT,
  "muon_wd": $MUON_WD,
  "embed_wd": $EMBED_WD,
  "adam_wd": $ADAM_WD,
  "compressor": "$COMPRESSOR",
  "byte_shuffle": $BYTE_SHUFFLE,
  "matrix_bits": $MATRIX_BITS,
  "embed_bits": $EMBED_BITS,
  "matrix_clip_mode": "$MATRIX_CLIP_MODE",
  "embed_clip_mode": "$EMBED_CLIP_MODE",
  "matrix_clip_sigmas": $MATRIX_CLIP_SIGMAS,
  "embed_clip_sigmas": $EMBED_CLIP_SIGMAS,
  "ttt_lr": $TTT_LR,
  "ttt_epochs": $TTT_EPOCHS,
  "ttt_chunk_tokens": $TTT_CHUNK_TOKENS,
  "ttt_chunk_bytes": $TTT_CHUNK_BYTES
}
EOF

  {
    echo "=== phase=$PHASE case=$case_name seed=$seed started=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    env | grep -E '^(RUN_ID|SEED|MAX_WALLCLOCK_SECONDS|ITERATIONS|TRAIN_BATCH_TOKENS|VAL_BATCH_SIZE|VAL_TOKENS_LIMIT|TRAIN_SEQ_LEN|EVAL_SEQ_LEN|EVAL_STRIDE|NPROC_PER_NODE|DATA_PATH|TOKENIZER_PATH|TOKENIZER_META_PATH|VOCAB_SIZE|MLP_MULT|QK_GAIN_INIT|XSA_LAST_N|BIGRAM_VOCAB_SIZE|VE_ENABLED|SMEAR_ENABLED|NUM_LOOPS|LOOP_START|LOOP_END|ENABLE_LOOPING_AT|MUON_WD|EMBED_WD|ADAM_WD|COMPRESSOR|BYTE_SHUFFLE|MATRIX_BITS|EMBED_BITS|MATRIX_CLIP_MODE|EMBED_CLIP_MODE|MATRIX_CLIP_SIGMAS|EMBED_CLIP_SIGMAS|TTT_ENABLED|TTT_LR|TTT_EPOCHS|TTT_CHUNK_TOKENS|TTT_CHUNK_BYTES|TTT_FREEZE_BLOCKS|TTT_FREEZE_EMBEDDINGS)=' | sort
    echo "=== command ==="
    echo "python3 -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE $TRAIN_SCRIPT"
    echo "==============="
  } > "$log_path"

  (
    cd "$case_dir"
    python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$TRAIN_SCRIPT"
  ) >> "$log_path" 2>&1
}

main() {
  local case_name seed
  for case_name in $(case_list "$CASE"); do
    for seed in ${SEEDS_CSV//,/ }; do
      run_case_seed "$case_name" "$seed"
    done
  done
}

main "$@"
