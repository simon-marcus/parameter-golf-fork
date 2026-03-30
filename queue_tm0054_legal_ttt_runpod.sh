#!/bin/bash
set -euo pipefail

cd /workspace/parameter-golf

DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RECORD_ROOT="/workspace/parameter-golf/records/track_10min_16mb/2026-03-30_tm0054_legal_ttt_runpod"
CURRENT_SEED="${CURRENT_SEED:-42}"
FOLLOW_ON_SEEDS="${FOLLOW_ON_SEEDS:-1337 2026}"
POLL_SECONDS="${POLL_SECONDS:-30}"
QUEUE_LOG="${QUEUE_LOG:-$RECORD_ROOT/follow_on_queue.log}"
LAUNCH_SCRIPT="/workspace/parameter-golf/launch_tm0054_legal_ttt_runpod.sh"

mkdir -p "$RECORD_ROOT"
touch "$QUEUE_LOG"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$QUEUE_LOG"
}

wait_for_seed() {
  local seed="$1"
  local out_dir="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_seed_${seed}"
  local pid_file="$out_dir/train.pid"

  while [[ ! -f "$pid_file" ]]; do
    log "waiting for pid file seed=$seed path=$pid_file"
    sleep "$POLL_SECONDS"
  done

  local pid
  pid="$(cat "$pid_file")"
  log "monitoring seed=$seed pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  log "seed=$seed pid=$pid exited"
}

launch_seed() {
  local seed="$1"
  log "launching seed=$seed"
  SEED="$seed" \
  RUN_ID="tm0054_legal_ttt_runpod_seed${seed}" \
  DATA_ROOT_MODE="$DATA_ROOT_MODE" \
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  bash "$LAUNCH_SCRIPT" | tee -a "$QUEUE_LOG"
}

log "queue start current_seed=$CURRENT_SEED follow_on_seeds=$FOLLOW_ON_SEEDS"
wait_for_seed "$CURRENT_SEED"

for seed in $FOLLOW_ON_SEEDS; do
  launch_seed "$seed"
  wait_for_seed "$seed"
done

log "queue complete"
