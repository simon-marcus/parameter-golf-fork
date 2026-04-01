#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SCREEN_SECONDS="${SCREEN_SECONDS:-180}"
WEIGHTS_CSV="${WEIGHTS_CSV:-0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00}"
RECORD_ROOT="${RECORD_ROOT:-$ROOT_DIR/records/nonrecord_jepa_baseline_weight_sweep}"

cd "$ROOT_DIR"

case "$DATA_ROOT_MODE" in
  workspace)
    export DATA_PATH="$ROOT_DIR/data/datasets/fineweb10B_byte260"
    export TOKENIZER_PATH="$ROOT_DIR/data/tokenizers/fineweb_pure_byte_260.json"
    ;;
  tmp)
    export DATA_PATH="/tmp/parameter-golf-data/datasets/fineweb10B_byte260"
    export TOKENIZER_PATH="/tmp/parameter-golf-data/tokenizers/fineweb_pure_byte_260.json"
    ;;
  *)
    echo "DATA_ROOT_MODE must be one of: workspace, tmp" >&2
    exit 1
    ;;
esac

if [ ! -d "$DATA_PATH" ]; then
  echo "missing data path: $DATA_PATH" >&2
  exit 1
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
  echo "missing tokenizer path: $TOKENIZER_PATH" >&2
  exit 1
fi

mkdir -p "$RECORD_ROOT"
SUMMARY_PATH="$RECORD_ROOT/summary.tsv"
printf "weight\trun_id\tlog_path\tfinal_exact_bpb\tfinal_exact_loss\n" >"$SUMMARY_PATH"

OLDIFS="$IFS"
IFS=','
read -r -a WEIGHTS <<< "$WEIGHTS_CSV"
IFS="$OLDIFS"

for weight in "${WEIGHTS[@]}"; do
  weight_tag="${weight/./p}"
  run_id="jepa_iso_w${weight_tag}"
  out_dir="$RECORD_ROOT/$run_id"
  log_path="$out_dir/train.log"
  mkdir -p "$out_dir"

  echo "=== Running JEPA_LOSS_WEIGHT=$weight (run_id=$run_id) ==="
  export RUN_ID="$run_id"
  export TOKENIZER_KIND=byte
  export VOCAB_SIZE=260
  export JEPA_LOSS_WEIGHT="$weight"
  export MAX_WALLCLOCK_SECONDS="$SCREEN_SECONDS"
  export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
  export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
  export USE_COMPILE="${USE_COMPILE:-0}"
  export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
  export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
  export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"

  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" train_jepa_baseline.py \
    >"$log_path" 2>&1

  exact_line="$(grep 'final_int8_zlib_roundtrip_exact val_loss:' "$log_path" | tail -1 || true)"
  exact_loss=""
  exact_bpb=""
  if [ -n "$exact_line" ]; then
    exact_loss="$(sed -E 's/.*val_loss:([0-9.]+) val_bpb:.*/\1/' <<<"$exact_line")"
    exact_bpb="$(sed -E 's/.*val_bpb:([0-9.]+).*/\1/' <<<"$exact_line")"
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" "$weight" "$run_id" "$log_path" "$exact_bpb" "$exact_loss" >>"$SUMMARY_PATH"
done

echo "Sweep complete: $SUMMARY_PATH"
