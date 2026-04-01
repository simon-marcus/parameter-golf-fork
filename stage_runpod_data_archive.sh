#!/bin/bash
set -euo pipefail

ARCHIVE_PATH="${1:?usage: stage_runpod_data_archive.sh <archive_path> [dest_root] [variant] [tokenizer_basename]}"
DEST_ROOT="${2:-/tmp/parameter-golf-data}"
VARIANT="${3:-fineweb10B_sp1024}"
TOKENIZER_BASENAME="${4:-fineweb_1024_bpe}"

DEST_DATASET_DIR="$DEST_ROOT/datasets/$VARIANT"
DEST_TOKENIZER_DIR="$DEST_ROOT/tokenizers"

if [[ ! -f "$ARCHIVE_PATH" ]]; then
  echo "missing archive path: $ARCHIVE_PATH" >&2
  exit 1
fi

rm -rf "$DEST_DATASET_DIR" "$DEST_TOKENIZER_DIR"
mkdir -p "$DEST_ROOT"

echo "staging archive from $ARCHIVE_PATH to $DEST_ROOT"
tar --zstd --no-same-owner -xf "$ARCHIVE_PATH" -C "$DEST_ROOT"

echo "archive stage ready"
echo "DATA_PATH=$DEST_DATASET_DIR"
echo "TOKENIZER_PATH=$DEST_TOKENIZER_DIR/$TOKENIZER_BASENAME.model"
echo "dataset_files=$(find "$DEST_DATASET_DIR" -maxdepth 1 -type f | wc -l)"
echo "dataset_bytes=$(du -sh "$DEST_DATASET_DIR" | cut -f1)"
