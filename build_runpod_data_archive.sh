#!/bin/bash
set -euo pipefail

SRC_ROOT="${1:-./data}"
ARCHIVE_ROOT="${2:-$SRC_ROOT/archives}"
VARIANT="${3:-fineweb10B_sp1024}"
TOKENIZER_BASENAME="${4:-fineweb_1024_bpe}"
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"

SRC_DATASET_DIR="$SRC_ROOT/datasets/$VARIANT"
SRC_TOKENIZER_DIR="$SRC_ROOT/tokenizers"
ARCHIVE_BASENAME="${VARIANT}__${TOKENIZER_BASENAME}"
ARCHIVE_PATH="$ARCHIVE_ROOT/${ARCHIVE_BASENAME}.tar.zst"
MANIFEST_PATH="$ARCHIVE_ROOT/${ARCHIVE_BASENAME}.manifest.tsv"

mkdir -p "$ARCHIVE_ROOT"

if [[ ! -d "$SRC_DATASET_DIR" ]]; then
  echo "missing dataset dir: $SRC_DATASET_DIR" >&2
  exit 1
fi
if [[ ! -f "$SRC_TOKENIZER_DIR/$TOKENIZER_BASENAME.model" ]]; then
  echo "missing tokenizer model: $SRC_TOKENIZER_DIR/$TOKENIZER_BASENAME.model" >&2
  exit 1
fi

TMP_ARCHIVE="${ARCHIVE_PATH}.tmp"
rm -f "$TMP_ARCHIVE"

TAR_INPUTS=(
  "datasets/$VARIANT"
  "tokenizers/$TOKENIZER_BASENAME.model"
)
if [[ -f "$SRC_TOKENIZER_DIR/$TOKENIZER_BASENAME.vocab" ]]; then
  TAR_INPUTS+=("tokenizers/$TOKENIZER_BASENAME.vocab")
fi

(
  cd "$SRC_ROOT"
  tar -I "zstd -T0 -${ZSTD_LEVEL}" -cf "$TMP_ARCHIVE" "${TAR_INPUTS[@]}"
)

mv "$TMP_ARCHIVE" "$ARCHIVE_PATH"

{
  printf "key\tvalue\n"
  printf "archive_path\t%s\n" "$ARCHIVE_PATH"
  printf "archive_bytes\t%s\n" "$(stat -c '%s' "$ARCHIVE_PATH" 2>/dev/null || stat -f '%z' "$ARCHIVE_PATH")"
  printf "variant\t%s\n" "$VARIANT"
  printf "tokenizer_basename\t%s\n" "$TOKENIZER_BASENAME"
  printf "train_shards\t%s\n" "$(find "$SRC_DATASET_DIR" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l | awk '{print $1}')"
  printf "val_shards\t%s\n" "$(find "$SRC_DATASET_DIR" -maxdepth 1 -name 'fineweb_val_*.bin' | wc -l | awk '{print $1}')"
  printf "dataset_bytes\t%s\n" "$(du -sh "$SRC_DATASET_DIR" | awk '{print $1}')"
  printf "archive_sha256\t%s\n" "$(shasum -a 256 "$ARCHIVE_PATH" | awk '{print $1}')"
} >"$MANIFEST_PATH"

echo "archive_ready=$ARCHIVE_PATH"
echo "manifest_ready=$MANIFEST_PATH"
