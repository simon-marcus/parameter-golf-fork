#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/parameter-golf}"
VARIANT="${2:-sp1024}"
TRAIN_SHARDS="${3:-80}"

cd "$ROOT_DIR"

if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.local
  set +a
fi

DATASET_DIR="$ROOT_DIR/data/datasets/fineweb10B_${VARIANT}"
TOKENIZER_PATH="$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model"

python3 - "$DATASET_DIR" <<'PY'
import os
import sys
from pathlib import Path

import numpy as np

data_path = Path(sys.argv[1])
header_bytes = 256 * np.dtype("<i4").itemsize
token_bytes = np.dtype("<u2").itemsize

removed = []
for path in sorted(data_path.glob("fineweb_*.bin")):
    try:
        size = path.stat().st_size
        if size <= header_bytes:
            raise ValueError("too small")
        header = np.fromfile(path, dtype="<i4", count=256)
        if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
            raise ValueError("bad header")
        num_tokens = int(header[2])
        expected_size = header_bytes + num_tokens * token_bytes
        if size != expected_size:
            raise ValueError(f"size mismatch expected {expected_size} got {size}")
    except Exception as exc:
        removed.append((path.name, str(exc)))
        path.unlink(missing_ok=True)

print(f"removed_bad_shards={len(removed)}")
for name, reason in removed:
    print(f"removed {name}: {reason}")
PY

RUNPOD_DOWNLOAD_DATA=1 python3 data/cached_challenge_fineweb.py --variant "$VARIANT" --train-shards "$TRAIN_SHARDS"
bash "$ROOT_DIR/verify_runpod_data_ready.sh" "$DATASET_DIR" "$TOKENIZER_PATH" "$TRAIN_SHARDS" 1
