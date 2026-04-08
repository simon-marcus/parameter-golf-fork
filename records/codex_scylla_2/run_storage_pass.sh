#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <variant>"
  exit 2
fi

variant="$1"
shift || true

export CHECKPOINT_PATH="${CHECKPOINT_PATH:-final_model.pt}"
export RUN_LEGAL_TTT="${RUN_LEGAL_TTT:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Match the old Scylla P-lane architecture unless the caller overrides it.
export VOCAB_SIZE="${VOCAB_SIZE:-1254}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export MLP_MULT="${MLP_MULT:-3}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-6400}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export ROPE_DIMS="${ROPE_DIMS:-16}"
export LN_SCALE="${LN_SCALE:-1}"
export VE_ENABLED="${VE_ENABLED:-1}"
export VE_DIM="${VE_DIM:-128}"
export VE_LAYERS="${VE_LAYERS:-9,10}"

case "$variant" in
  S0)
    # Broadest low-risk storage-only pass: all major categories at int6.
    export QUANT_CATS="${QUANT_CATS:-mlp,attn,embed,other}"
    export LZMA_PRESET="${LZMA_PRESET:-6}"
    export DROP_EXPORT_KEYS="${DROP_EXPORT_KEYS:-}"
    export DROP_EXPORT_PREFIXES="${DROP_EXPORT_PREFIXES:-}"
    export ZERO_BIGRAM="${ZERO_BIGRAM:-0}"
    ;;
  S1)
    # Same as S0, but stronger compression.
    export QUANT_CATS="${QUANT_CATS:-mlp,attn,embed,other}"
    export LZMA_PRESET="${LZMA_PRESET:-9}"
    export DROP_EXPORT_KEYS="${DROP_EXPORT_KEYS:-}"
    export DROP_EXPORT_PREFIXES="${DROP_EXPORT_PREFIXES:-}"
    export ZERO_BIGRAM="${ZERO_BIGRAM:-0}"
    ;;
  S2)
    # Fallback if still over cap: export-time bigram ablation plus strongest compression.
    export QUANT_CATS="${QUANT_CATS:-mlp,attn,embed,other}"
    export LZMA_PRESET="${LZMA_PRESET:-9}"
    export DROP_EXPORT_KEYS="${DROP_EXPORT_KEYS:-}"
    export DROP_EXPORT_PREFIXES="${DROP_EXPORT_PREFIXES:-}"
    export ZERO_BIGRAM="${ZERO_BIGRAM:-1}"
    ;;
  *)
    echo "unknown variant: $variant"
    exit 2
    ;;
esac

echo "storage-pass variant=$variant"
echo "checkpoint=$CHECKPOINT_PATH"
echo "quant_cats=$QUANT_CATS"
echo "lzma_preset=$LZMA_PRESET"
echo "drop_export_keys=${DROP_EXPORT_KEYS:-}"
echo "drop_export_prefixes=${DROP_EXPORT_PREFIXES:-}"
echo "zero_bigram=$ZERO_BIGRAM"
echo "run_legal_ttt=$RUN_LEGAL_TTT"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  records/codex_scylla_2/scylla_storage_pass.py "$@"
