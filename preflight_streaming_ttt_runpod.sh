#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/parameter-golf}"
SMOKE_VARIANT="${2:-}"

cd "$ROOT_DIR"

echo "=== Streaming TTT RunPod Preflight ==="

if [ "${RUNPOD_BOOTSTRAP:-1}" = "1" ]; then
  bash "$ROOT_DIR/setup_runpod.sh"
else
  echo "Skipping setup_runpod.sh because RUNPOD_BOOTSTRAP=0"
fi

bash "$ROOT_DIR/repair_runpod_data.sh" "$ROOT_DIR" sp1024 80
bash "$ROOT_DIR/setup_local_parity_data_runpod.sh" "$ROOT_DIR/data" /tmp/parameter-golf-data fineweb10B_sp1024 fineweb_1024_bpe
bash "$ROOT_DIR/verify_runpod_data_ready.sh" /tmp/parameter-golf-data/datasets/fineweb10B_sp1024 /tmp/parameter-golf-data/tokenizers/fineweb_1024_bpe.model

echo "workspace verification: ok"
echo "tmp verification: ok"

if [ -n "$SMOKE_VARIANT" ]; then
  echo "launching smoke variant: $SMOKE_VARIANT"
  USE_COMPILE="${USE_COMPILE:-0}" \
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-90}" \
  ITERATIONS="${ITERATIONS:-400}" \
  DATA_ROOT_MODE=tmp \
  bash "$ROOT_DIR/launch_streaming_ttt_candidate_runpod.sh" "$SMOKE_VARIANT"
else
  echo "No smoke variant requested."
fi

echo "=== Preflight complete ==="
