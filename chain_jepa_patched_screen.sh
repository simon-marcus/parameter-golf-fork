#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHAIN_VARIANTS="${CHAIN_VARIANTS:-control jepa01}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SCREEN_SECONDS="${SCREEN_SECONDS:-180}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"
USE_COMPILE="${USE_COMPILE:-0}"
PATCH_SIZE="${PATCH_SIZE:-4}"
PATCH_EMBED_MODE="${PATCH_EMBED_MODE:-flat}"
RECORD_ROOT="${RECORD_ROOT:-$ROOT_DIR/records/nonrecord_jepa_patched}"
SUMMARY_PATH="$RECORD_ROOT/summary.tsv"

mkdir -p "$RECORD_ROOT"
printf "variant\trun_id\tlog_path\tfinal_exact_bpb\tfinal_exact_loss\n" >"$SUMMARY_PATH"

cd "$ROOT_DIR"

for variant in $CHAIN_VARIANTS; do
  echo "=== Running variant=$variant ===" | tee -a "$RECORD_ROOT/chain_driver.log"

  export DATA_ROOT_MODE
  export NPROC_PER_NODE
  export MAX_WALLCLOCK_SECONDS="$SCREEN_SECONDS"
  export TRAIN_BATCH_TOKENS
  export TRAIN_SEQ_LEN
  export VAL_BATCH_SIZE
  export USE_COMPILE
  export PATCH_SIZE
  export PATCH_EMBED_MODE
  export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
  export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"

  "$ROOT_DIR/launch_jepa_patched_runpod.sh" "$variant"

  mode_suffix=""
  if [ "$PATCH_EMBED_MODE" != "flat" ]; then
    mode_suffix="_${PATCH_EMBED_MODE}"
  fi
  case "$variant" in
    control) out_dir="$ROOT_DIR/records/nonrecord_jepa_patched/${DATA_ROOT_MODE}${mode_suffix}_control" ;;
    jepa01) out_dir="$ROOT_DIR/records/nonrecord_jepa_patched/${DATA_ROOT_MODE}${mode_suffix}_jepa01" ;;
    jepa02) out_dir="$ROOT_DIR/records/nonrecord_jepa_patched/${DATA_ROOT_MODE}${mode_suffix}_jepa02" ;;
    weight:*)
      weight_value="${variant#weight:}"
      weight_tag="${weight_value//./p}"
      out_dir="$ROOT_DIR/records/nonrecord_jepa_patched/${DATA_ROOT_MODE}${mode_suffix}_w${weight_tag}"
      ;;
    *)
      echo "unrecognized variant for summary path: $variant" >&2
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
  printf "%s\t%s\t%s\t%s\t%s\n" "$variant" "$run_id" "$log_path" "$exact_bpb" "$exact_loss" >>"$SUMMARY_PATH"
done

echo "Patched screen complete: $SUMMARY_PATH"
