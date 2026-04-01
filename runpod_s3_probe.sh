#!/bin/bash
set -euo pipefail

VOLUME_ID="${RUNPOD_S3_VOLUME_ID:?set RUNPOD_S3_VOLUME_ID}"
DATACENTER="${RUNPOD_S3_DATACENTER:?set RUNPOD_S3_DATACENTER}"
ENDPOINT="${RUNPOD_S3_ENDPOINT:?set RUNPOD_S3_ENDPOINT}"
PROBE_KEY="${RUNPOD_S3_PROBE_KEY:-parameter-golf/s3-probe.txt}"
TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

printf 'runpod-s3-probe %s\n' "$(date -u +%FT%TZ)" >"$TMP_FILE"

set -x
aws s3 cp \
  --region "$DATACENTER" \
  --endpoint-url "$ENDPOINT" \
  "$TMP_FILE" \
  "s3://$VOLUME_ID/$PROBE_KEY"

aws s3 ls \
  --region "$DATACENTER" \
  --endpoint-url "$ENDPOINT" \
  "s3://$VOLUME_ID/$(dirname "$PROBE_KEY")/"

aws s3 cp \
  --region "$DATACENTER" \
  --endpoint-url "$ENDPOINT" \
  "s3://$VOLUME_ID/$PROBE_KEY" \
  -
