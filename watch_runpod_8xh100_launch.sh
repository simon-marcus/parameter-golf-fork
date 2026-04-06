#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-30}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/runpod_artifacts/launch_watch}"
RUNPOD_API_URL="${RUNPOD_API_URL:-https://rest.runpod.io/v1/pods}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
POD_NAME="${POD_NAME:-pg-leader-jepa-8x}"
IMAGE_NAME="${IMAGE_NAME:-runpod/parameter-golf:latest}"
GPU_COUNT="${GPU_COUNT:-8}"
GPU_TYPE="${GPU_TYPE:-NVIDIA H100 80GB HBM3}"

DATACENTER_IDS='[
  "AP-JP-1",
  "CA-MTL-1",
  "CA-MTL-2",
  "CA-MTL-3",
  "CA-MTL-4",
  "EU-CZ-1",
  "EU-DK-1",
  "EU-FR-1",
  "EU-NL-1",
  "EU-RO-1",
  "EU-SE-1",
  "EU-SE-2",
  "EUR-IS-1",
  "EUR-IS-2",
  "EUR-IS-3",
  "EUR-IS-4",
  "EUR-NO-1",
  "EUR-NO-2",
  "OC-AU-1",
  "SEA-SG-1",
  "US-CA-1",
  "US-CA-2",
  "US-DE-1",
  "US-GA-1",
  "US-GA-2",
  "US-IL-1",
  "US-KS-1",
  "US-KS-2",
  "US-KS-3",
  "US-MD-1",
  "US-MO-1",
  "US-MO-2",
  "US-NC-1",
  "US-NC-2",
  "US-NE-1",
  "US-OR-1",
  "US-OR-2",
  "US-PA-1",
  "US-TX-1",
  "US-TX-2",
  "US-TX-3",
  "US-TX-4",
  "US-TX-5",
  "US-TX-6",
  "US-WA-1"
]'

DCS_JSON="${DCS_JSON:-$ALLOWLIST_DEFAULT}"

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/watch_${STAMP}.log"
SUCCESS_JSON="$LOG_DIR/success_${STAMP}.json"

if [[ -z "$RUNPOD_API_KEY" ]]; then
  if [[ -f "$ROOT_DIR/.env.local" ]]; then
    RUNPOD_API_KEY="$(grep '^RUNPOD_API_KEY=' "$ROOT_DIR/.env.local" | head -n 1 | cut -d= -f2-)"
  fi
fi

if [[ -z "$RUNPOD_API_KEY" ]]; then
  echo "RUNPOD_API_KEY is not set and could not be read from $ROOT_DIR/.env.local" >&2
  exit 1
fi

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_PATH"
}

build_body() {
  cat <<EOF
{
  "cloudType": "SECURE",
  "computeType": "GPU",
  "name": "${POD_NAME}",
  "imageName": "${IMAGE_NAME}",
  "gpuCount": ${GPU_COUNT},
  "gpuTypeIds": ["${GPU_TYPE}"],
  "gpuTypePriority": "custom",
  "dataCenterIds": ${DCS_JSON},
  "dataCenterPriority": "availability",
  "interruptible": false,
  "containerDiskInGb": 50,
  "volumeInGb": 50,
  "volumeMountPath": "/workspace"
}
EOF
}

log "starting 8xH100 REST watch"
log "interval_seconds=$INTERVAL_SECONDS"
log "pod_name=$POD_NAME"
log "gpu_type=$GPU_TYPE gpu_count=$GPU_COUNT"
log "datacenter_priority=availability"

BODY="$(build_body)"

while true; do
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  log "trying create ts=$ts"

  resp="$(
    curl -sS -X POST "$RUNPOD_API_URL" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -H "Content-Type: application/json" \
      --data "$BODY" || true
  )"

  if printf '%s' "$resp" | grep -Eq '"success"[[:space:]]*:[[:space:]]*true'; then
    printf '%s\n' "$resp" >"$SUCCESS_JSON"
    log "success json=$SUCCESS_JSON"
    printf '%s\n' "$resp" | jq -r '.id // .podId // .data.id // empty' | sed 's/^/pod_id=/' | tee -a "$LOG_PATH"
    exit 0
  fi

  msg="$(
    printf '%s' "$resp" \
      | sed -n 's/.*"message"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
      | head -n 1
  )"
  if [[ -n "${msg:-}" ]]; then
    log "not_available message=$msg"
  else
    log "response=$(printf '%s' "$resp" | tr '\n' ' ' | cut -c1-500)"
  fi

  sleep "$INTERVAL_SECONDS"
done
