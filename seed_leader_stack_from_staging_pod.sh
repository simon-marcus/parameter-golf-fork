#!/bin/bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <stage_host> <stage_port> <target_host> <target_port>" >&2
  exit 1
fi

STAGE_HOST="$1"
STAGE_PORT="$2"
TARGET_HOST="$3"
TARGET_PORT="$4"

REPO_ROOT="/workspace/parameter-golf"
STAGE_REPO_ROOT="/workspace/parameter-golf"
BUNDLE_PATH="$STAGE_REPO_ROOT/parameter-golf-leader-stack-jepa-bundle.tar"
ARCHIVE_PATH="$STAGE_REPO_ROOT/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst"
TARGET_ARCHIVE_DIR="$REPO_ROOT/data/archives"
TARGET_RECORD_DIR="$REPO_ROOT/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon"

ssh -o StrictHostKeyChecking=no "root@${TARGET_HOST}" -p "$TARGET_PORT" \
  "mkdir -p '$REPO_ROOT' '$TARGET_ARCHIVE_DIR' '$TARGET_RECORD_DIR' '$REPO_ROOT/logs'"

ssh -o StrictHostKeyChecking=no "root@${STAGE_HOST}" -p "$STAGE_PORT" "cat '$BUNDLE_PATH'" \
  | ssh -o StrictHostKeyChecking=no "root@${TARGET_HOST}" -p "$TARGET_PORT" \
      "cat > '$REPO_ROOT/parameter-golf-leader-stack-jepa-bundle.tar'"

ssh -o StrictHostKeyChecking=no "root@${TARGET_HOST}" -p "$TARGET_PORT" \
  "tar xf '$REPO_ROOT/parameter-golf-leader-stack-jepa-bundle.tar' -C '$REPO_ROOT' && rm -f '$REPO_ROOT/parameter-golf-leader-stack-jepa-bundle.tar'"

"$(cd "$(dirname "$0")" && pwd)/relay_runpod_archive.sh" \
  "$STAGE_HOST" "$STAGE_PORT" "$ARCHIVE_PATH" \
  "$TARGET_HOST" "$TARGET_PORT" "$TARGET_ARCHIVE_DIR/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst"

ssh -o StrictHostKeyChecking=no "root@${TARGET_HOST}" -p "$TARGET_PORT" \
  "ls -lh '$TARGET_ARCHIVE_DIR/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst' '$REPO_ROOT/train_gpt_leader_stack_jepa.py' '$TARGET_RECORD_DIR/train_gpt.py'"
