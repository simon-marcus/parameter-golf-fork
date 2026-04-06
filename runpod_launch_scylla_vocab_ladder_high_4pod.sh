#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
CANONICAL_REPO_ROOT="${CANONICAL_REPO_ROOT:-$HOME/Code/parameter-golf}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/records/scylla_2_claude/runpod_vocab_ladder_manifest}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST_PATH="$MANIFEST_DIR/$TIMESTAMP.tsv"
REMOTE_ROOT="/workspace/parameter-golf"
S3_BUCKET="${S3_BUCKET:-parameter-golf-staging-094651608775}"
PRESIGN_TTL="${PRESIGN_TTL:-43200}"
SUITE_ROOT_REMOTE="${SUITE_ROOT_REMOTE:-$REMOTE_ROOT/records/scylla_2_claude/frontier_runs_vocab_ladder_high}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
ITERATIONS="${ITERATIONS:-1200}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
USE_COMPILE="${USE_COMPILE:-0}"
SEEDS="${SEEDS:-2026}"
COMPRESSOR="${COMPRESSOR:-lzma}"
BYTE_SHUFFLE="${BYTE_SHUFFLE:-0}"
MATRIX_CLIP_MODE="${MATRIX_CLIP_MODE:-std}"
EMBED_CLIP_MODE="${EMBED_CLIP_MODE:-std}"
MATRIX_CLIP_SIGMAS="${MATRIX_CLIP_SIGMAS:-12.85}"
EMBED_CLIP_SIGMAS="${EMBED_CLIP_SIGMAS:-20.0}"

VOCABS=(6144 10240 12288 14336)
CASES=(ladder_6144 ladder_10240 ladder_12288 ladder_14336)
TRAIN_SHARDS=(9 8 8 8)
VAL_SHARDS=(1 1 1 1)
POD_NAMES=(
  "pg-scylla-v6144"
  "pg-scylla-v10240"
  "pg-scylla-v12288"
  "pg-scylla-v14336"
)

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

load_runpod_env() {
  local env_file=""
  if [[ -f "$REPO_ROOT/.env.local" ]]; then
    env_file="$REPO_ROOT/.env.local"
  elif [[ -f "$CANONICAL_REPO_ROOT/.env.local" ]]; then
    env_file="$CANONICAL_REPO_ROOT/.env.local"
  fi

  if [[ -n "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi

  if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "RUNPOD_API_KEY is not set. Populate .env.local or environment." >&2
    exit 1
  fi
}

create_pod() {
  local pod_name="$1"
  runpodctl pod create \
    --name "$pod_name" \
    --template-id "y5cejece4j" \
    --gpu-id "NVIDIA H100 80GB HBM3" \
    --gpu-count 1 \
    --cloud-type SECURE \
    --ssh \
    --container-disk-in-gb 50 \
    --volume-in-gb 50 \
    -o json
}

pod_ssh_parts() {
  local pod_id="$1"
  runpodctl pod get "$pod_id" -o json | python3 -c '
import json, sys
pod = json.load(sys.stdin)
ssh = pod.get("ssh", {}) if isinstance(pod, dict) else {}
ip = ssh.get("ip")
port = ssh.get("port")
if ip and port:
    print(f"{ip} {port}")
'
}

wait_for_ssh() {
  local pod_id="$1"
  local attempts="${2:-90}"
  local sleep_seconds="${3:-10}"
  local parts=""
  for ((i = 1; i <= attempts; i++)); do
    parts="$(pod_ssh_parts "$pod_id" || true)"
    if [[ -n "$parts" ]]; then
      echo "$parts"
      return 0
    fi
    sleep "$sleep_seconds"
  done
  echo "Timed out waiting for SSH on pod $pod_id" >&2
  return 1
}

rsync_repo() {
  local host="$1"
  local port="$2"
  rsync -az \
    --no-owner \
    --no-group \
    -e "ssh -p $port -o StrictHostKeyChecking=no -i $SSH_KEY_PATH" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.DS_Store' \
    --exclude '.venv*' \
    --exclude '.claude' \
    --exclude '.gstack' \
    --exclude 'data/datasets' \
    --exclude 'data/tokenizers' \
    --exclude 'data/archives' \
    --exclude 'runpod_artifacts' \
    --exclude 'modal_logs' \
    --exclude 'logs' \
    --exclude 'autoresearch' \
    --exclude 'tmp' \
    --exclude 'records/track_10min_16mb' \
    --exclude 'records/track_non_record_16mb' \
    --exclude 'records/non_record' \
    --exclude 'records/byte-level-jepa' \
    --exclude 'records/codex_scylla_2' \
    --exclude 'records/scylla_2_claude/runs' \
    --exclude 'records/scylla_2_claude/frontier_runs' \
    --exclude 'records/scylla_2_claude/frontier_runs_vocab_ladder' \
    --exclude 'records/scylla_2_claude/frontier_runs_vocab_ladder_high' \
    --exclude 'records/scylla_2_claude/pulled_*' \
    "$REPO_ROOT/" \
    "root@$host:$REMOTE_ROOT/"
}

remote_stage_and_launch() {
  local host="$1"
  local port="$2"
  local vocab="$3"
  local case_name="$4"
  local train_shards="$5"
  local val_shards="$6"
  local archive_url="$7"
  local archive_name="$8"

  local dataset_name="fineweb10B_scylla_v2_v${vocab}"
  local tokenizer_name="scylla_v2_v${vocab}"

  ssh -p "$port" -o StrictHostKeyChecking=no -i "$SSH_KEY_PATH" "root@$host" "bash -s" <<EOF
set -euo pipefail
test -x /opt/parameter-golf/validate_image.sh && bash /opt/parameter-golf/validate_image.sh || true
mkdir -p "$REMOTE_ROOT/data/archives"
cd "$REMOTE_ROOT"
bash ./ops/runpod_image/validate_image.sh

curl -L --fail "$archive_url" -o "$REMOTE_ROOT/data/archives/$archive_name"
bash ./stage_runpod_data_archive.sh \
  "$REMOTE_ROOT/data/archives/$archive_name" \
  /tmp/parameter-golf-data \
  "$dataset_name" \
  "$tokenizer_name"
bash ./verify_runpod_data_ready.sh \
  "/tmp/parameter-golf-data/datasets/$dataset_name" \
  "/tmp/parameter-golf-data/tokenizers/$tokenizer_name.meta.npz" \
  "$train_shards" \
  "$val_shards"

mkdir -p "$SUITE_ROOT_REMOTE"
nohup env \
  DATA_ROOT_MODE=tmp \
  NPROC_PER_NODE=1 \
  SEEDS="$SEEDS" \
  USE_COMPILE="$USE_COMPILE" \
  MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
  ITERATIONS="$ITERATIONS" \
  TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
  VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
  COMPRESSOR="$COMPRESSOR" \
  BYTE_SHUFFLE="$BYTE_SHUFFLE" \
  MATRIX_CLIP_MODE="$MATRIX_CLIP_MODE" \
  EMBED_CLIP_MODE="$EMBED_CLIP_MODE" \
  MATRIX_CLIP_SIGMAS="$MATRIX_CLIP_SIGMAS" \
  EMBED_CLIP_SIGMAS="$EMBED_CLIP_SIGMAS" \
  SUITE_ROOT="$SUITE_ROOT_REMOTE" \
  SCYLLA_V${vocab}_EXPECTED_TRAIN_SHARDS="$train_shards" \
  SCYLLA_V${vocab}_EXPECTED_VAL_SHARDS="$val_shards" \
  bash ./launch_scylla_frontier_plan.sh "$case_name" \
  > "$SUITE_ROOT_REMOTE/${case_name}.launch.log" 2>&1 < /dev/null &
echo \$! > "$SUITE_ROOT_REMOTE/${case_name}.pid"
echo "started case=$case_name pid=\$(cat "$SUITE_ROOT_REMOTE/${case_name}.pid")"
EOF
}

main() {
  require_cmd runpodctl
  require_cmd aws
  require_cmd rsync
  require_cmd python3
  require_cmd curl
  load_runpod_env
  mkdir -p "$MANIFEST_DIR"
  printf "case\tvocab\tpod_name\tpod_id\thost\tport\ttrain_shards\tval_shards\tarchive_key\n" > "$MANIFEST_PATH"

  local -a hosts=()
  local -a ports=()
  local -a archive_urls=()
  local -a archive_keys=()
  local -a archive_names=()

  local vocab archive_key archive_name
  for vocab in "${VOCABS[@]}"; do
    archive_key="data/archives/scylla_v2_v${vocab}.tar.zst"
    archive_name="scylla_v2_v${vocab}.tar.zst"
    archive_keys+=("$archive_key")
    archive_names+=("$archive_name")
    archive_urls+=("$(aws s3 presign "s3://$S3_BUCKET/$archive_key" --expires-in "$PRESIGN_TTL")")
  done

  local idx pod_name pod_json pod_id ssh_parts host port
  for idx in "${!VOCABS[@]}"; do
    pod_name="${POD_NAMES[$idx]}-$TIMESTAMP"
    pod_json="$(create_pod "$pod_name")"
    pod_id="$(printf '%s' "$pod_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
    ssh_parts="$(wait_for_ssh "$pod_id")"
    host="$(echo "$ssh_parts" | awk '{print $1}')"
    port="$(echo "$ssh_parts" | awk '{print $2}')"

    hosts+=("$host")
    ports+=("$port")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${CASES[$idx]}" "${VOCABS[$idx]}" "$pod_name" "$pod_id" "$host" "$port" \
      "${TRAIN_SHARDS[$idx]}" "${VAL_SHARDS[$idx]}" "${archive_keys[$idx]}" >> "$MANIFEST_PATH"
  done

  for idx in "${!VOCABS[@]}"; do
    (
      rsync_repo "${hosts[$idx]}" "${ports[$idx]}"
      remote_stage_and_launch \
        "${hosts[$idx]}" \
        "${ports[$idx]}" \
        "${VOCABS[$idx]}" \
        "${CASES[$idx]}" \
        "${TRAIN_SHARDS[$idx]}" \
        "${VAL_SHARDS[$idx]}" \
        "${archive_urls[$idx]}" \
        "${archive_names[$idx]}"
    ) &
  done
  wait

  echo "Launched 4 Stage A Scylla ladder pods."
  echo "Manifest: $MANIFEST_PATH"
  cat "$MANIFEST_PATH"
}

main "$@"
