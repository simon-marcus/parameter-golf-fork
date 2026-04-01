#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
OUT_DIR="${2:-$ROOT_DIR/runpod_artifacts/jepa_iso_bundle}"

mkdir -p "$OUT_DIR"

MANIFEST="$OUT_DIR/MANIFEST.txt"
ARCHIVE="$OUT_DIR/parameter-golf-jepa-iso-bundle.tar"

cat >"$MANIFEST" <<'EOF'
PLAN_AND_PROGRESS.md
train_jepa_baseline.py
launch_jepa_baseline_runpod.sh
setup_byte260_data_runpod.sh
verify_runpod_data_ready.sh
records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py
data/manifest.json
data/tokenizers/fineweb_pure_byte_260.json
data/datasets/fineweb10B_byte260
EOF

tar -cf "$ARCHIVE" -C "$ROOT_DIR" $(cat "$MANIFEST")

echo "bundle_ready=$ARCHIVE"
echo "bundle_bytes=$(stat -f %z "$ARCHIVE" 2>/dev/null || stat -c %s "$ARCHIVE")"
