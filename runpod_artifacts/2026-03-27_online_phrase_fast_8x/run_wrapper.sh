#!/bin/bash
set -euo pipefail
cd /workspace/parameter-golf
export PYTHONUNBUFFERED=1
echo "[$(date)] fast-eval launch start"
DATA_ROOT_MODE=tmp SEED=${SEED:-1337} RUN_ID=${RUN_ID:-online_phrase_fast_seed1337} \
  bash /workspace/parameter-golf/records/track_10min_16mb/2026-03-27_OnlinePhraseCacheFastEval/launch.sh base
echo "[$(date)] fast-eval launch done"
