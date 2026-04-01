#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/parameter-golf}"
LAUNCH_SCRIPT="${REPO_ROOT}/launch_leader_stack_jepa_screen_runpod.sh"
RECORD_ROOT="${REPO_ROOT}/records/nonrecord_leader_stack_jepa"
SUMMARY_PATH="${RECORD_ROOT}/chain_summary.tsv"
POLL_SECONDS="${POLL_SECONDS:-10}"
CHAIN_VARIANTS="${CHAIN_VARIANTS:-control weight:0.05 jepa01 weight:0.15}"

mkdir -p "${RECORD_ROOT}"

variant_to_run_id() {
  local variant="$1"
  case "$variant" in
    control)
      echo "leader_stack_jepa_control"
      ;;
    jepa01)
      echo "leader_stack_jepa_jepa01"
      ;;
    weight:*)
      local weight_value="${variant#weight:}"
      local weight_tag="${weight_value//./p}"
      echo "leader_stack_jepa_w${weight_tag}"
      ;;
    *)
      echo "unknown variant: $variant" >&2
      exit 1
      ;;
  esac
}

variant_to_record_dir() {
  local variant="$1"
  case "$variant" in
    control)
      echo "${RECORD_ROOT}/tmp_control"
      ;;
    jepa01)
      echo "${RECORD_ROOT}/tmp_jepa01"
      ;;
    weight:*)
      local weight_value="${variant#weight:}"
      local weight_tag="${weight_value//./p}"
      echo "${RECORD_ROOT}/tmp_w${weight_tag}"
      ;;
    *)
      echo "unknown variant: $variant" >&2
      exit 1
      ;;
  esac
}

extract_final_bpb() {
  local log_path="$1"
  grep 'final_int8_zlib_roundtrip_exact' "$log_path" | tail -n 1 | sed -E 's/.*val_bpb:([0-9.]+)/\1/'
}

ensure_header() {
  if [[ ! -f "${SUMMARY_PATH}" ]]; then
    printf "variant\trun_id\ttrain_log\tfinal_int8_zlib_roundtrip_exact_bpb\n" >"${SUMMARY_PATH}"
  fi
}

summary_has_variant() {
  local variant="$1"
  grep -q "^${variant}[[:space:]]" "${SUMMARY_PATH}" 2>/dev/null
}

append_summary() {
  local variant="$1"
  local run_id="$2"
  local log_path="$3"
  local final_bpb
  final_bpb="$(extract_final_bpb "$log_path")"
  printf "%s\t%s\t%s\t%s\n" "$variant" "$run_id" "$log_path" "${final_bpb:-}" >>"${SUMMARY_PATH}"
}

wait_for_process_exit() {
  local pattern="$1"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    sleep "${POLL_SECONDS}"
  done
}

run_variant() {
  local variant="$1"
  local run_id="$2"
  (
    cd "${REPO_ROOT}"
    env \
      RUN_ID="${run_id}" \
      USE_COMPILE="${USE_COMPILE:-0}" \
      DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}" \
      NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
      MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}" \
      VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}" \
      TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}" \
      bash "${LAUNCH_SCRIPT}" "${variant}"
  )
}

ensure_header

for variant in ${CHAIN_VARIANTS}; do
  run_id="$(variant_to_run_id "${variant}")"
  record_dir="$(variant_to_record_dir "${variant}")"
  train_log="${record_dir}/train.log"

  if summary_has_variant "${variant}"; then
    continue
  fi

  if grep -q 'final_int8_zlib_roundtrip_exact' "${train_log}" 2>/dev/null; then
    append_summary "${variant}" "${run_id}" "${train_log}"
    continue
  fi

  if pgrep -f "train_gpt_leader_stack_jepa.py" >/dev/null 2>&1; then
    wait_for_process_exit 'train_gpt_leader_stack_jepa.py'
    if grep -q 'final_int8_zlib_roundtrip_exact' "${train_log}" 2>/dev/null; then
      append_summary "${variant}" "${run_id}" "${train_log}"
      continue
    fi
  fi

  run_variant "${variant}" "${run_id}"
  append_summary "${variant}" "${run_id}" "${train_log}"
done
