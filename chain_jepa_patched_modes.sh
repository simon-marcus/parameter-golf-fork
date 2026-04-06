#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_MODES="${PATCH_MODES:-mixer attn}"
CHAIN_VARIANTS="${CHAIN_VARIANTS:-control jepa01}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SCREEN_SECONDS="${SCREEN_SECONDS:-180}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"
USE_COMPILE="${USE_COMPILE:-0}"
PATCH_SIZE="${PATCH_SIZE:-2}"
RECORD_ROOT="${RECORD_ROOT:-$ROOT_DIR/records/nonrecord_jepa_patched}"
SUMMARY_PATH="$RECORD_ROOT/mode_chain_summary.tsv"
DRIVER_PATH="$RECORD_ROOT/mode_chain_driver.log"

mkdir -p "$RECORD_ROOT"
printf "mode_variant\trun_id\tlog_path\tfinal_exact_bpb\tfinal_exact_loss\n" >"$SUMMARY_PATH"
: >"$DRIVER_PATH"

cd "$ROOT_DIR"

for mode in $PATCH_MODES; do
  echo "=== mode=$mode ===" | tee -a "$DRIVER_PATH"
  export PATCH_EMBED_MODE="$mode"
  export DATA_ROOT_MODE
  export NPROC_PER_NODE
  export SCREEN_SECONDS
  export TRAIN_BATCH_TOKENS
  export TRAIN_SEQ_LEN
  export VAL_BATCH_SIZE
  export USE_COMPILE
  export PATCH_SIZE
  export CHAIN_VARIANTS

  bash ./chain_jepa_patched_screen.sh >>"$DRIVER_PATH" 2>&1

  mode_suffix="_${mode}"
  for variant in $CHAIN_VARIANTS; do
    case "$variant" in
      control) out_dir="$RECORD_ROOT/${DATA_ROOT_MODE}${mode_suffix}_control" ;;
      jepa01) out_dir="$RECORD_ROOT/${DATA_ROOT_MODE}${mode_suffix}_jepa01" ;;
      jepa02) out_dir="$RECORD_ROOT/${DATA_ROOT_MODE}${mode_suffix}_jepa02" ;;
      weight:*)
        weight_value="${variant#weight:}"
        weight_tag="${weight_value//./p}"
        out_dir="$RECORD_ROOT/${DATA_ROOT_MODE}${mode_suffix}_w${weight_tag}"
        ;;
      *)
        echo "unrecognized variant: $variant" >&2
        exit 1
        ;;
    esac
    log_path="$out_dir/train.log"
    exact_line="$(grep 'final_int8_zlib_roundtrip_exact val_loss:' "$log_path" | tail -1 || true)"
    exact_loss=""
    exact_bpb=""
    run_id=""
    if [ -n "$exact_line" ]; then
      exact_loss="$(sed -E 's/.*val_loss:([0-9.]+) val_bpb:.*/\1/' <<<"$exact_line")"
      exact_bpb="$(sed -E 's/.*val_bpb:([0-9.]+).*/\1/' <<<"$exact_line")"
    fi
    if [ -f "$log_path" ]; then
      run_id="$(grep '^RUN_ID=' "$log_path" | head -1 | cut -d= -f2- || true)"
    fi
    printf "%s:%s\t%s\t%s\t%s\t%s\n" "$mode" "$variant" "$run_id" "$log_path" "$exact_bpb" "$exact_loss" >>"$SUMMARY_PATH"
  done
done

echo "Mode chain complete: $SUMMARY_PATH" | tee -a "$DRIVER_PATH"
