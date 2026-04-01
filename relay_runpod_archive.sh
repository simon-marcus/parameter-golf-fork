#!/bin/bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <src_host> <src_port> <src_path> <dst_host> <dst_port> <dst_path>" >&2
  exit 1
fi

SRC_HOST="$1"
SRC_PORT="$2"
SRC_PATH="$3"
DST_HOST="$4"
DST_PORT="$5"
DST_PATH="$6"

DST_DIR="$(dirname "$DST_PATH")"

ssh -o StrictHostKeyChecking=no "root@${SRC_HOST}" -p "$SRC_PORT" "cat '$SRC_PATH'" \
  | dd bs=16m status=progress \
  | ssh -o StrictHostKeyChecking=no "root@${DST_HOST}" -p "$DST_PORT" "mkdir -p '$DST_DIR' && cat > '$DST_PATH'"
