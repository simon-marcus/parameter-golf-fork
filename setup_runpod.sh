#!/bin/bash
# Setup script for RunPod environment (using the parameter-golf template)
# The template pre-installs PyTorch + CUDA. We only install missing Python deps here.
# Data download is opt-in because pulling shards on an 8xH100 pod is expensive.
# Run from /workspace/parameter-golf/

set -euo pipefail

echo "=== Parameter Golf — RunPod Setup ==="

if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.local
  set +a
fi

if [ "${RUNPOD_SKIP_PIP:-0}" = "1" ]; then
  echo "Skipping pip bootstrap because RUNPOD_SKIP_PIP=1"
else
  # Keep the pod bootstrap narrow by default: the competition runtime plus data access.
  pip install --break-system-packages -r requirements.txt
  if [ "${RUNPOD_INSTALL_RESEARCH_EXTRAS:-0}" = "1" ]; then
    pip install --break-system-packages -r requirements-research.txt
  fi
fi

if [ "${RUNPOD_DOWNLOAD_DATA:-0}" = "1" ]; then
  echo "Downloading dataset..."
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
  ./verify_runpod_data_ready.sh \
    /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
    /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
else
  echo "Skipping dataset download."
  echo "Set RUNPOD_DOWNLOAD_DATA=1 to download shards on this pod."
  echo "Preferred flow: prepare data on a cheap pod or volume, then stage to /tmp on the expensive pod."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run autoresearch:"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "  # Single GPU, 3-min experiments (for fast iteration):"
echo "  python3 autoresearch.py"
echo ""
echo "  # Single GPU, custom time:"
echo "  EXPERIMENT_SECONDS=120 python3 autoresearch.py"
echo ""
echo "  # 8×GPU, full 10-min runs (for final validation):"
echo "  GPUS=8 EXPERIMENT_SECONDS=600 python3 autoresearch.py"
