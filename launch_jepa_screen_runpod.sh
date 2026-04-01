#!/bin/bash
set -euo pipefail

VARIANT="${1:-control}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
SCREEN_SECONDS="${SCREEN_SECONDS:-180}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

cd /workspace/parameter-golf

RECORD_ROOT="/workspace/parameter-golf/records/nonrecord_jepa"
SCRIPT_PATH="/workspace/parameter-golf/train_jepa.py"

case "$DATA_ROOT_MODE" in
  workspace)
    export DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_byte260
    ;;
  tmp)
    export DATA_PATH=/tmp/parameter-golf-data/datasets/fineweb10B_byte260
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

python3 - "$DATA_PATH" <<'PY'
import sys
from pathlib import Path
import numpy as np

data_path = Path(sys.argv[1])
files = sorted(data_path.glob("fineweb_train_*.bin")) + sorted(data_path.glob("fineweb_val_*.bin"))
if not files:
    raise SystemExit(f"no shard files found in {data_path}")
header_bytes = 256 * np.dtype("<i4").itemsize
token_bytes = np.dtype("<u2").itemsize
for path in files:
    if path.stat().st_size <= header_bytes:
        raise SystemExit(f"shard too small or empty: {path}")
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise SystemExit(f"invalid shard header: {path}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if path.stat().st_size != expected_size:
        raise SystemExit(f"shard size mismatch: {path} expected {expected_size} got {path.stat().st_size}")
print(f"verified byte260 shards in {data_path}")
PY

export TOKENIZER_KIND=byte
export TOKENIZER_PATH=""
export TOKENIZER_META_PATH=""
export VOCAB_SIZE=260
export MAX_WALLCLOCK_SECONDS="$SCREEN_SECONDS"
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=25
export USE_COMPILE="${USE_COMPILE:-0}"

case "$VARIANT" in
  control)
    export RUN_ID=jepa_screen_control
    export JEPA_LOSS_WEIGHT="${JEPA_LOSS_WEIGHT:-0.0}"
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_control"
    ;;
  jepa01)
    export RUN_ID=jepa_screen_jepa01
    export JEPA_LOSS_WEIGHT="${JEPA_LOSS_WEIGHT:-0.1}"
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_jepa01"
    ;;
  jepa02)
    export RUN_ID=jepa_screen_jepa02
    export JEPA_LOSS_WEIGHT="${JEPA_LOSS_WEIGHT:-0.2}"
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_jepa02"
    ;;
  patch16)
    export RUN_ID=jepa_screen_patch16
    export JEPA_LOSS_WEIGHT="${JEPA_LOSS_WEIGHT:-0.2}"
    export PATCH_SIZE="${PATCH_SIZE:-16}"
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_patch16"
    ;;
  ema099)
    export RUN_ID=jepa_screen_ema099
    export JEPA_LOSS_WEIGHT="${JEPA_LOSS_WEIGHT:-0.2}"
    export TARGET_EMA="${TARGET_EMA:-0.99}"
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_ema099"
    ;;
  *)
    echo "Usage: $0 {control|jepa01|jepa02|patch16|ema099}" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

nohup python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_PATH" \
  > "$OUT_DIR/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo "started $RUN_ID data_root_mode=$DATA_ROOT_MODE nproc_per_node=$NPROC_PER_NODE screen_seconds=$SCREEN_SECONDS pid=$PID log=$OUT_DIR/train.log"
