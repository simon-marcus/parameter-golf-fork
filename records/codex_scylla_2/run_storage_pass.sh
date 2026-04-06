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
