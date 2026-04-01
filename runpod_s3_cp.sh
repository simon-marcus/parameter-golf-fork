#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <local_or_s3_src> <local_or_s3_dst> [extra aws s3 cp args]" >&2
  exit 1
fi

SRC="$1"
DST="$2"
EXTRA_ARGS="${3:-}"

VOLUME_ID="${RUNPOD_S3_VOLUME_ID:?set RUNPOD_S3_VOLUME_ID}"
DATACENTER="${RUNPOD_S3_DATACENTER:?set RUNPOD_S3_DATACENTER}"
ENDPOINT="${RUNPOD_S3_ENDPOINT:?set RUNPOD_S3_ENDPOINT}"

normalize_path() {
  local value="$1"
  if [[ "$value" == s3://* ]]; then
    printf '%s' "$value"
  elif [[ "$value" == /* || "$value" == ./* || "$value" == ../* ]]; then
    printf '%s' "$value"
  else
    printf 's3://%s/%s' "$VOLUME_ID" "$value"
  fi
}

SRC_NORM="$(normalize_path "$SRC")"
DST_NORM="$(normalize_path "$DST")"

set -x
aws s3 cp \
  --region "$DATACENTER" \
  --endpoint-url "$ENDPOINT" \
  ${EXTRA_ARGS} \
  "$SRC_NORM" \
  "$DST_NORM"
