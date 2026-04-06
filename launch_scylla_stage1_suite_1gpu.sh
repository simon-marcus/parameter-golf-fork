#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
VERIFY_SCRIPT="$REPO_ROOT/verify_runpod_data_ready.sh"
TRAIN_SCRIPT="$REPO_ROOT/records/scylla_2_claude/train_gpt_legal_ttt.py"

ROLE="${1:-}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
SEEDS_CSV="${SEEDS:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SUITE_ROOT="${SUITE_ROOT:-$REPO_ROOT/records/scylla_2_claude/stage1_1gpu_runs}"

if [[ -z "$ROLE" ]]; then
  echo "Usage: $0 {scylla_qk_xsa|scylla_capacity|scylla_combo|standard_control}"
  exit 1
fi

ensure_data_ready() {
  local family="$1"
  case "$family" in
    scylla)
      case "$DATA_ROOT_MODE" in
        workspace)
          export DATA_PATH="${DATA_PATH:-$REPO_ROOT/scylla_v2_bundle/datasets/fineweb10B_scylla_v2_cap0_fullbyte}"
          export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/scylla_v2_bundle/tokenizers/scylla_v2_cap0_fullbyte.yaml}"
          export TOKENIZER_META_PATH="${TOKENIZER_META_PATH:-$REPO_ROOT/scylla_v2_bundle/tokenizers/scylla_v2_cap0_fullbyte.meta.npz}"
          ;;
        tmp)
          export DATA_PATH="${DATA_PATH:-/tmp/parameter-golf-data/datasets/fineweb10B_scylla_v2_cap0_fullbyte}"
          export TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/parameter-golf-data/tokenizers/scylla_v2_cap0_fullbyte.yaml}"
          export TOKENIZER_META_PATH="${TOKENIZER_META_PATH:-/tmp/parameter-golf-data/tokenizers/scylla_v2_cap0_fullbyte.meta.npz}"
          ;;
        *)
          echo "DATA_ROOT_MODE must be workspace or tmp"
          exit 1
          ;;
      esac
      "$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_META_PATH" 11 1
      ;;
    sp1024)
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
          echo "DATA_ROOT_MODE must be workspace or tmp"
          exit 1
          ;;
      esac
      export TOKENIZER_META_PATH="${TOKENIZER_META_PATH:-}"
      "$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_PATH" 80 1
      ;;
    *)
      echo "Unknown dataset family: $family"
      exit 1
      ;;
  esac
  mkdir -p "$SUITE_ROOT"
}

set_common_env() {
  export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
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
  export VE_ENABLED="${VE_ENABLED:-1}"
  export VE_DIM="${VE_DIM:-128}"
  export VE_LAYERS="${VE_LAYERS:-9,10}"
  export MUON_WD="${MUON_WD:-0.04}"
  export ADAM_WD="${ADAM_WD:-0.04}"
  export EMBED_LR="${EMBED_LR:-0.6}"
  export HEAD_LR="${HEAD_LR:-0.008}"
  export MATRIX_LR="${MATRIX_LR:-0.025}"
  export SCALAR_LR="${SCALAR_LR:-0.025}"
  export TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
  export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
  export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
  export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-400}"
  export WARMDOWN_ITERS="${WARMDOWN_ITERS:-600}"
  export SWA_ENABLED="${SWA_ENABLED:-1}"
  export SWA_EVERY="${SWA_EVERY:-50}"
  export LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.15}"
  export TTT_ENABLED=0
  export NGRAM_ENABLED=0
  export USE_COMPILE="${USE_COMPILE:-0}"
}

configure_dataset_family() {
  local family="$1"
  set_common_env
  case "$family" in
    scylla)
      export VOCAB_SIZE="${VOCAB_SIZE:-1254}"
      export MLP_MULT="${MLP_MULT:-3}"
      export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-6400}"
      export XSA_LAST_N="${XSA_LAST_N:-4}"
      ;;
    sp1024)
      export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
      export MLP_MULT="${MLP_MULT:-3}"
      export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
      export XSA_LAST_N="${XSA_LAST_N:-4}"
      ;;
    *)
      echo "Unknown dataset family: $family"
      exit 1
      ;;
  esac
}

role_cases() {
  case "$ROLE" in
    scylla_qk_xsa)
      echo "scylla" "scylla_base scylla_qk45 scylla_qk50 scylla_xsa_all"
      ;;
    scylla_capacity)
      echo "scylla" "scylla_mlp35 scylla_mlp40 scylla_warmdown900 scylla_mlp35_warmdown900"
      ;;
    scylla_combo)
      echo "scylla" "scylla_qk45_xsa_all scylla_qk50_xsa_all scylla_qk45_mlp35 scylla_qk50_mlp35"
      ;;
    standard_control)
      echo "sp1024" "sp_base sp_qk45 sp_qk50 sp_xsa_all"
      ;;
    *)
      echo ""
      ;;
  esac
}

configure_case() {
  local dataset_family="$1"
  local case_name="$2"

  configure_dataset_family "$dataset_family"
  export QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}"

  case "$case_name" in
    scylla_base|sp_base)
      ;;
    scylla_qk45|sp_qk45)
      export QK_GAIN_INIT=4.5
      ;;
    scylla_qk50|sp_qk50)
      export QK_GAIN_INIT=5.0
      ;;
    scylla_xsa_all|sp_xsa_all)
      export XSA_LAST_N=11
      ;;
    scylla_mlp35)
      export MLP_MULT=3.5
      ;;
    scylla_mlp40)
      export MLP_MULT=4.0
      ;;
    scylla_warmdown900)
      export WARMDOWN_ITERS=900
      ;;
    scylla_mlp35_warmdown900)
      export MLP_MULT=3.5
      export WARMDOWN_ITERS=900
      ;;
    scylla_qk45_xsa_all)
      export QK_GAIN_INIT=4.5
      export XSA_LAST_N=11
      ;;
    scylla_qk50_xsa_all)
      export QK_GAIN_INIT=5.0
      export XSA_LAST_N=11
      ;;
    scylla_qk45_mlp35)
      export QK_GAIN_INIT=4.5
      export MLP_MULT=3.5
      ;;
    scylla_qk50_mlp35)
      export QK_GAIN_INIT=5.0
      export MLP_MULT=3.5
      ;;
    *)
      echo "Unknown case: $case_name"
      exit 1
      ;;
  esac
}

run_case_seed() {
  local dataset_family="$1"
  local case_name="$2"
  local seed="$3"
  local case_dir="$SUITE_ROOT/$ROLE/$case_name/seed_$seed"
  local log_path="$case_dir/train.log"

  mkdir -p "$case_dir"
  configure_case "$dataset_family" "$case_name"
  export SEED="$seed"
  export RUN_ID="${ROLE}_${case_name}_seed${seed}"

  cat > "$case_dir/config.json" <<EOF
{
  "role": "$ROLE",
  "dataset_family": "$dataset_family",
  "case_name": "$case_name",
  "seed": $seed,
  "data_path": "$DATA_PATH",
  "tokenizer_path": "$TOKENIZER_PATH",
  "tokenizer_meta_path": "${TOKENIZER_META_PATH:-}",
  "iterations": ${ITERATIONS},
  "train_batch_tokens": ${TRAIN_BATCH_TOKENS},
  "val_tokens_limit": ${VAL_TOKENS_LIMIT},
  "qk_gain_init": ${QK_GAIN_INIT},
  "xsa_last_n": ${XSA_LAST_N},
  "mlp_mult": ${MLP_MULT},
  "warmdown_iters": ${WARMDOWN_ITERS}
}
EOF

  {
    echo "=== role=$ROLE case=$case_name seed=$seed started=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    env | grep -E '^(RUN_ID|SEED|MAX_WALLCLOCK_SECONDS|ITERATIONS|TRAIN_BATCH_TOKENS|VAL_BATCH_SIZE|VAL_TOKENS_LIMIT|TRAIN_SEQ_LEN|EVAL_SEQ_LEN|EVAL_STRIDE|NPROC_PER_NODE|DATA_PATH|TOKENIZER_PATH|TOKENIZER_META_PATH|VOCAB_SIZE|MLP_MULT|QK_GAIN_INIT|XSA_LAST_N|WARMDOWN_ITERS|BIGRAM_VOCAB_SIZE|TTT_ENABLED|NGRAM_ENABLED|USE_COMPILE)=' | sort
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
  local family_and_cases
  family_and_cases="$(role_cases)"
  if [[ -z "$family_and_cases" ]]; then
    echo "Usage: $0 {scylla_qk_xsa|scylla_capacity|scylla_combo|standard_control}"
    exit 1
  fi

  local dataset_family cases_str
  dataset_family="$(echo "$family_and_cases" | awk '{print $1}')"
  cases_str="$(echo "$family_and_cases" | cut -d' ' -f2-)"
  ensure_data_ready "$dataset_family"

  IFS=' ' read -r -a cases <<< "$cases_str"
  IFS=',' read -r -a seeds <<< "$SEEDS_CSV"
  for case_name in "${cases[@]}"; do
    for seed in "${seeds[@]}"; do
      run_case_seed "$dataset_family" "$case_name" "$seed"
    done
  done
}

main "$@"
