#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <pod_id> <host> <port> [artifact_dir]" >&2
  exit 1
fi

POD_ID="$1"
HOST="$2"
PORT="$3"
ARTIFACT_DIR="${4:-/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-31_leader_stack_jepa_1xh100}"

REMOTE_ROOT="/workspace/parameter-golf"
REMOTE_RECORD_ROOT="${REMOTE_ROOT}/records/nonrecord_leader_stack_jepa"
REMOTE_LOG_ROOT="${REMOTE_ROOT}/logs"
SSH_OPTS=(-o StrictHostKeyChecking=no -p "$PORT")
CHAIN_VARIANTS="${CHAIN_VARIANTS:-control weight:0.05 jepa01 weight:0.15}"

mkdir -p "${ARTIFACT_DIR}/records" "${ARTIFACT_DIR}/logs"

variant_to_log_name() {
  local variant="$1"
  case "$variant" in
    control)
      echo "leader_stack_jepa_control.txt"
      ;;
    jepa01)
      echo "leader_stack_jepa_jepa01.txt"
      ;;
    weight:*)
      local weight_value="${variant#weight:}"
      local weight_tag="${weight_value//./p}"
      echo "leader_stack_jepa_w${weight_tag}.txt"
      ;;
    *)
      echo "unknown variant: $variant" >&2
      exit 1
      ;;
  esac
}

expected_variant_count() {
  awk '{print NF}' <<<"${CHAIN_VARIANTS}"
}

wait_for_chain_completion() {
  local expected_count
  expected_count="$(expected_variant_count)"
  while true; do
    local summary
    summary="$(ssh "${SSH_OPTS[@]}" "root@${HOST}" \
      "test -f '${REMOTE_RECORD_ROOT}/chain_summary.tsv' && cat '${REMOTE_RECORD_ROOT}/chain_summary.tsv' || true")"
    local completed_count
    completed_count="$(printf '%s\n' "$summary" | awk 'NR > 1 && $1 != \"\" {count += 1} END {print count + 0}')"
    if [[ "${completed_count}" -ge "${expected_count}" ]]; then
      break
    fi
    sleep 15
  done
}

pull_logs() {
  rsync -avz -e "ssh -p ${PORT} -o StrictHostKeyChecking=no" \
    "root@${HOST}:${REMOTE_RECORD_ROOT}/" "${ARTIFACT_DIR}/records/"
  for variant in ${CHAIN_VARIANTS}; do
    local log_name
    log_name="$(variant_to_log_name "${variant}")"
    rsync -avz -e "ssh -p ${PORT} -o StrictHostKeyChecking=no" \
      "root@${HOST}:${REMOTE_LOG_ROOT}/${log_name}" "${ARTIFACT_DIR}/logs/"
  done
}

stop_pod_with_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if runpodctl pod stop "${POD_ID}" -o json >/dev/null; then
      return 0
    fi
    sleep 10
  done
  echo "failed to stop pod ${POD_ID} after retries" >&2
  exit 1
}

wait_for_chain_completion
pull_logs
stop_pod_with_retry
