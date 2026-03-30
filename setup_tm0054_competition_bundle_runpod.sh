#!/bin/bash
set -euo pipefail

VOLUME_ROOT="${1:-/workspace/pg-data/tm0054_competition_export}"
DEST_ROOT="${2:-/tmp/parameter-golf-tm0054-competition}"
VARIANT="${3:-fineweb10B_tm0054}"
EXPECTED_TRAIN_SHARDS="${EXPECTED_TRAIN_SHARDS:-79}"
EXPECTED_VAL_SHARDS="${EXPECTED_VAL_SHARDS:-1}"

SRC_DATASET_DIR="$VOLUME_ROOT/datasets/$VARIANT"
SRC_TOKENIZER_DIR="$VOLUME_ROOT/tokenizers"
DEST_DATASET_DIR="$DEST_ROOT/datasets/$VARIANT"
DEST_TOKENIZER_DIR="$DEST_ROOT/tokenizers"

if [ ! -d "$SRC_DATASET_DIR" ]; then
  echo "missing source dataset dir: $SRC_DATASET_DIR" >&2
  exit 1
fi

if [ ! -f "$SRC_TOKENIZER_DIR/candidate.vocab" ]; then
  echo "missing source tokenizer vocab: $SRC_TOKENIZER_DIR/candidate.vocab" >&2
  exit 1
fi

if [ ! -f "$SRC_TOKENIZER_DIR/candidate.meta.npz" ]; then
  echo "missing source tokenizer meta: $SRC_TOKENIZER_DIR/candidate.meta.npz" >&2
  exit 1
fi

rm -rf "$DEST_DATASET_DIR" "$DEST_TOKENIZER_DIR"
mkdir -p "$DEST_DATASET_DIR" "$DEST_TOKENIZER_DIR"

echo "staging dataset from $SRC_DATASET_DIR to $DEST_DATASET_DIR"
rsync -a --delete "$SRC_DATASET_DIR"/ "$DEST_DATASET_DIR"/

echo "staging tokenizer from $SRC_TOKENIZER_DIR to $DEST_TOKENIZER_DIR"
rsync -a --delete "$SRC_TOKENIZER_DIR"/ "$DEST_TOKENIZER_DIR"/

echo "verifying staged competition bundle"
bash /workspace/parameter-golf/verify_runpod_data_ready.sh \
  "$DEST_DATASET_DIR" \
  "$DEST_TOKENIZER_DIR/candidate.meta.npz" \
  "$EXPECTED_TRAIN_SHARDS" \
  "$EXPECTED_VAL_SHARDS"

echo "competition tm0054 bundle ready"
echo "DATA_PATH=$DEST_DATASET_DIR"
echo "TOKENIZER_PATH=$DEST_TOKENIZER_DIR/candidate.vocab"
echo "TOKENIZER_META_PATH=$DEST_TOKENIZER_DIR/candidate.meta.npz"
