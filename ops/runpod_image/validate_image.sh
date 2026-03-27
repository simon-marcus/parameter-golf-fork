#!/usr/bin/env bash
set -euo pipefail

echo "== parameter-golf image validation =="

for bin in python3 git rsync tmux jq pigz; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "missing required binary: $bin" >&2
    exit 1
  fi
done

python3 /opt/parameter-golf/preflight_report.py

mkdir -p /workspace/.image-check /tmp/.image-check
touch /workspace/.image-check/ok /tmp/.image-check/ok

echo "workspace_write_ok=1"
echo "tmp_write_ok=1"
echo "image_validation=passed"
