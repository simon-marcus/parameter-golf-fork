#!/bin/bash
set -euo pipefail
cd /workspace/parameter-golf
export PYTHONUNBUFFERED=1
echo "[$(date)] setup_runpod start"
RUNPOD_DOWNLOAD_DATA=1 bash ./setup_runpod.sh
echo "[$(date)] staging tmp data"
bash ./setup_local_parity_data_runpod.sh
echo "[$(date)] launch start"
DATA_ROOT_MODE=tmp SEED=${SEED:-1337} RUN_ID=${RUN_ID:-online_phrase_cal_seed1337} \
  bash /workspace/parameter-golf/records/track_10min_16mb/2026-03-27_OnlinePhraseCacheCalibrated/launch.sh base
echo "[$(date)] launch done"
