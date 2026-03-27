#!/bin/bash
set -euo pipefail

DATA_PATH="${1:?usage: verify_runpod_data_ready.sh <data_path> <tokenizer_path> [expected_train_shards] [expected_val_shards] }"
TOKENIZER_PATH="${2:?usage: verify_runpod_data_ready.sh <data_path> <tokenizer_path> [expected_train_shards] [expected_val_shards] }"
EXPECTED_TRAIN_SHARDS="${3:-80}"
EXPECTED_VAL_SHARDS="${4:-1}"

if [ ! -d "$DATA_PATH" ]; then
  echo "missing data path: $DATA_PATH" >&2
  exit 1
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
  echo "missing tokenizer model: $TOKENIZER_PATH" >&2
  exit 1
fi

if [ ! -s "$TOKENIZER_PATH" ]; then
  echo "tokenizer model is empty: $TOKENIZER_PATH" >&2
  exit 1
fi

TRAIN_COUNT=$(find "$DATA_PATH" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l | awk '{print $1}')
VAL_COUNT=$(find "$DATA_PATH" -maxdepth 1 -name 'fineweb_val_*.bin' | wc -l | awk '{print $1}')

if [ "$TRAIN_COUNT" -lt "$EXPECTED_TRAIN_SHARDS" ]; then
  echo "insufficient train shards: found $TRAIN_COUNT expected at least $EXPECTED_TRAIN_SHARDS" >&2
  exit 1
fi

if [ "$VAL_COUNT" -lt "$EXPECTED_VAL_SHARDS" ]; then
  echo "insufficient val shards: found $VAL_COUNT expected at least $EXPECTED_VAL_SHARDS" >&2
  exit 1
fi

python3 - "$DATA_PATH" "$TOKENIZER_PATH" <<'PY'
import sys
from pathlib import Path
import numpy as np

data_path = Path(sys.argv[1])
tokenizer_path = Path(sys.argv[2])

files = sorted(data_path.glob("fineweb_train_*.bin")) + sorted(data_path.glob("fineweb_val_*.bin"))
if not files:
    raise SystemExit("no shard files found")

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

if tokenizer_path.stat().st_size <= 0:
    raise SystemExit(f"tokenizer file is empty: {tokenizer_path}")

print(f"verified train_shards={len(sorted(data_path.glob('fineweb_train_*.bin')))} val_shards={len(sorted(data_path.glob('fineweb_val_*.bin')))}")
print(f"verified tokenizer={tokenizer_path}")
PY
