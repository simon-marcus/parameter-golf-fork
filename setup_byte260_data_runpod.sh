#!/bin/bash
set -euo pipefail

SRC_ROOT="${1:-/workspace/parameter-golf/data}"
DEST_ROOT="${2:-/tmp/parameter-golf-data}"

SRC_DATASET_DIR="$SRC_ROOT/datasets/fineweb10B_byte260"
SRC_TOKENIZER_PATH="$SRC_ROOT/tokenizers/fineweb_pure_byte_260.json"
DEST_DATASET_DIR="$DEST_ROOT/datasets/fineweb10B_byte260"
DEST_TOKENIZER_DIR="$DEST_ROOT/tokenizers"
DEST_TOKENIZER_PATH="$DEST_TOKENIZER_DIR/fineweb_pure_byte_260.json"

rm -rf "$DEST_DATASET_DIR" "$DEST_TOKENIZER_DIR"
mkdir -p "$DEST_DATASET_DIR" "$DEST_TOKENIZER_DIR"

echo "staging byte260 dataset from $SRC_DATASET_DIR to $DEST_DATASET_DIR"
cp -a "$SRC_DATASET_DIR"/. "$DEST_DATASET_DIR"/

echo "staging byte260 tokenizer from $SRC_TOKENIZER_PATH to $DEST_TOKENIZER_PATH"
cp -a "$SRC_TOKENIZER_PATH" "$DEST_TOKENIZER_PATH"

echo "byte260 local parity data ready"
echo "DATA_PATH=$DEST_DATASET_DIR"
echo "TOKENIZER_PATH=$DEST_TOKENIZER_PATH"
echo "dataset_files=$(find "$DEST_DATASET_DIR" -maxdepth 1 -type f | wc -l)"
echo "dataset_bytes=$(du -sh "$DEST_DATASET_DIR" | cut -f1)"
