#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
VOLUME_PREFIX="${2:-parameter-golf}"

VOLUME_ID="${RUNPOD_S3_VOLUME_ID:?set RUNPOD_S3_VOLUME_ID}"
DATACENTER="${RUNPOD_S3_DATACENTER:?set RUNPOD_S3_DATACENTER}"
ENDPOINT="${RUNPOD_S3_ENDPOINT:?set RUNPOD_S3_ENDPOINT}"

cp_one() {
  local src="$1"
  local dst="$2"
  aws s3 cp \
    --region "$DATACENTER" \
    --endpoint-url "$ENDPOINT" \
    "$src" \
    "s3://$VOLUME_ID/$dst"
}

cp_one "$ROOT_DIR/runpod_artifacts/leader_stack_jepa_bundle/parameter-golf-leader-stack-jepa-bundle.tar" \
  "$VOLUME_PREFIX/staging_bundles/parameter-golf-leader-stack-jepa-bundle.tar"

cp_one "$ROOT_DIR/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst" \
  "$VOLUME_PREFIX/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst"

cp_one "$ROOT_DIR/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.manifest.tsv" \
  "$VOLUME_PREFIX/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.manifest.tsv"
