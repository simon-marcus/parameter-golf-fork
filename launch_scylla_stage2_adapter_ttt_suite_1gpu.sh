#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
VERIFY_SCRIPT="$REPO_ROOT/verify_runpod_data_ready.sh"
TRAIN_SCRIPT="$REPO_ROOT/records/scylla_2_claude/train_gpt_legal_ttt.py"

BASE_CONFIG_PATH="${1:-}"
SEEDS_CSV="${SEEDS:-1337}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SUITE_ROOT="${SUITE_ROOT:-$REPO_ROOT/records/scylla_2_claude/stage2_1gpu_adapter_runs}"

if [[ -z "$BASE_CONFIG_PATH" || ! -f "$BASE_CONFIG_PATH" ]]; then
  echo "Usage: $0 /path/to/base/config.json"
  exit 1
fi

json_field() {
  python3 - "$BASE_CONFIG_PATH" "$1" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get(sys.argv[2], ""))
PY
}

BASE_ROLE="$(json_field role)"
DATASET_FAMILY="$(json_field dataset_family)"
BASE_CASE="$(json_field case_name)"
BASE_QK_GAIN="$(json_field qk_gain_init)"
BASE_XSA_LAST_N="$(json_field xsa_last_n)"
BASE_MLP_MULT="$(json_field mlp_mult)"
BASE_WARMDOWN_ITERS="$(json_field warmdown_iters)"

ensure_data_ready() {
  case "$DATASET_FAMILY" in
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
      esac
      "$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_META_PATH" 11 1
      export VOCAB_SIZE=1254
      export BIGRAM_VOCAB_SIZE=6400
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
      esac
      export TOKENIZER_META_PATH="${TOKENIZER_META_PATH:-}"
      "$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_PATH" 80 1
      export VOCAB_SIZE=1024
      export BIGRAM_VOCAB_SIZE=3072
      ;;
    *)
      echo "unknown dataset family: $DATASET_FAMILY"
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
  export SWA_ENABLED="${SWA_ENABLED:-1}"
  export SWA_EVERY="${SWA_EVERY:-50}"
  export LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.15}"
  export USE_COMPILE="${USE_COMPILE:-0}"
  export NGRAM_ENABLED=0
  export CAUSAL_CACHE_ENABLED=0
  export TTT_ENABLED=1
  export TTT_MODE=adapter
  export QK_GAIN_INIT="$BASE_QK_GAIN"
  export XSA_LAST_N="$BASE_XSA_LAST_N"
  export MLP_MULT="$BASE_MLP_MULT"
  export WARMDOWN_ITERS="$BASE_WARMDOWN_ITERS"
}

adapter_cases() {
  echo "top4_r32 top4_r64 all_r32 all_r64"
}

configure_case() {
  local case_name="$1"
  set_common_env
  export TTT_LR=0.1
  export TTT_EPOCHS=1
  export TTT_CHUNK_TOKENS=32768
  export TTT_MOMENTUM=0.9
  export TTT_BATCH_SEQS=32
  export TTT_GRAD_CLIP=1.0
  export TTT_ADAPTER_SCALE=1.0
  case "$case_name" in
    top4_r32)
      export TTT_ADAPTER_LAST_N=4
      export TTT_ADAPTER_RANK=32
      ;;
    top4_r64)
      export TTT_ADAPTER_LAST_N=4
      export TTT_ADAPTER_RANK=64
      ;;
    all_r32)
      export TTT_ADAPTER_LAST_N=11
      export TTT_ADAPTER_RANK=32
      ;;
    all_r64)
      export TTT_ADAPTER_LAST_N=11
      export TTT_ADAPTER_RANK=64
      ;;
    *)
      echo "unknown case: $case_name"
      exit 1
      ;;
  esac
}

run_case_seed() {
  local case_name="$1"
  local seed="$2"
  local family_dir="${DATASET_FAMILY}_${BASE_CASE}"
  local case_dir="$SUITE_ROOT/$family_dir/$case_name/seed_$seed"
  local log_path="$case_dir/train.log"
  mkdir -p "$case_dir"
  configure_case "$case_name"
  export SEED="$seed"
  export RUN_ID="adapter_ttt_${DATASET_FAMILY}_${BASE_CASE}_${case_name}_seed${seed}"
  cat > "$case_dir/config.json" <<EOF
{
  "base_role": "$BASE_ROLE",
  "base_case_name": "$BASE_CASE",
  "base_config_path": "$BASE_CONFIG_PATH",
  "dataset_family": "$DATASET_FAMILY",
  "case_name": "$case_name",
  "seed": $seed,
  "ttt_mode": "adapter",
  "ttt_lr": ${TTT_LR},
  "ttt_epochs": ${TTT_EPOCHS},
  "ttt_chunk_tokens": ${TTT_CHUNK_TOKENS},
  "ttt_adapter_last_n": ${TTT_ADAPTER_LAST_N},
  "ttt_adapter_rank": ${TTT_ADAPTER_RANK}
}
EOF
  {
    echo "=== base_case=$BASE_CASE dataset_family=$DATASET_FAMILY case=$case_name seed=$seed started=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    env | grep -E '^(RUN_ID|SEED|ITERATIONS|TRAIN_BATCH_TOKENS|VAL_TOKENS_LIMIT|DATA_PATH|TOKENIZER_PATH|TOKENIZER_META_PATH|VOCAB_SIZE|QK_GAIN_INIT|XSA_LAST_N|MLP_MULT|WARMDOWN_ITERS|TTT_|CAUSAL_CACHE_)=' | sort
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
  ensure_data_ready
  IFS=' ' read -r -a cases <<< "$(adapter_cases)"
  IFS=',' read -r -a seeds <<< "$SEEDS_CSV"
  for case_name in "${cases[@]}"; do
    for seed in "${seeds[@]}"; do
      run_case_seed "$case_name" "$seed"
    done
  done
}

main "$@"
