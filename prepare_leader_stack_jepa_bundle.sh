#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
OUT_DIR="${2:-$ROOT_DIR/runpod_artifacts/leader_stack_jepa_bundle}"

mkdir -p "$OUT_DIR"

MANIFEST="$OUT_DIR/MANIFEST.txt"
ARCHIVE="$OUT_DIR/parameter-golf-leader-stack-jepa-bundle.tar"

cat >"$MANIFEST" <<'EOF'
train_gpt_leader_stack_jepa.py
launch_leader_stack_jepa_screen_runpod.sh
chain_leader_stack_jepa_runpod.sh
setup_local_parity_data_runpod.sh
verify_runpod_data_ready.sh
build_runpod_data_archive.sh
stage_runpod_data_archive.sh
records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py
EOF

tar -cf "$ARCHIVE" -C "$ROOT_DIR" $(cat "$MANIFEST")

echo "bundle_ready=$ARCHIVE"
echo "bundle_bytes=$(stat -f %z "$ARCHIVE" 2>/dev/null || stat -c %s "$ARCHIVE")"
