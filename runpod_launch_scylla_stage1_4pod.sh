#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/records/scylla_2_claude/runpod_stage1_manifest}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST_PATH="$MANIFEST_DIR/$TIMESTAMP.tsv"
REMOTE_ROOT="/workspace/parameter-golf"
S3_BUCKET="parameter-golf-staging-094651608775"
PRESIGN_TTL="${PRESIGN_TTL:-43200}"

ROLE_NAMES=(
  "scylla_qk_xsa"
  "scylla_capacity"
  "scylla_combo"
  "standard_control"
)
POD_NAMES=(
  "pg-s1-scylla-qk-xsa"
  "pg-s1-scylla-capacity"
  "pg-s1-scylla-combo"
  "pg-s1-standard-control"
)
ARCHIVE_KEYS=(
  "data/archives/scylla_v2_cap0_fullbyte.tar.zst"
  "data/archives/scylla_v2_cap0_fullbyte.tar.zst"
  "data/archives/scylla_v2_cap0_fullbyte.tar.zst"
  "data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst"
)
ARCHIVE_NAMES=(
  "scylla_v2_cap0_fullbyte.tar.zst"
  "scylla_v2_cap0_fullbyte.tar.zst"
  "scylla_v2_cap0_fullbyte.tar.zst"
  "fineweb10B_sp1024__fineweb_1024_bpe.tar.zst"
)

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

load_runpod_env() {
  if [[ -f "$REPO_ROOT/.env.local" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env.local"
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
    "$REPO_ROOT/" \
    "root@$host:$REMOTE_ROOT/"
}

remote_stage_and_launch() {
  local host="$1"
  local port="$2"
  local role="$3"
  local archive_url="$4"
  local archive_name="$5"

  ssh -p "$port" -o StrictHostKeyChecking=no -i "$SSH_KEY_PATH" "root@$host" "bash -s" <<EOF
set -euo pipefail
mkdir -p $REMOTE_ROOT/data/archives
cd $REMOTE_ROOT

curl -L --fail "$archive_url" -o "$REMOTE_ROOT/data/archives/$archive_name"

if [[ "$role" == "standard_control" ]]; then
  bash ./stage_runpod_data_archive.sh \
    "$REMOTE_ROOT/data/archives/$archive_name" \
    /tmp/parameter-golf-data \
    fineweb10B_sp1024 \
    fineweb_1024_bpe
  bash ./verify_runpod_data_ready.sh \
    /tmp/parameter-golf-data/datasets/fineweb10B_sp1024 \
    /tmp/parameter-golf-data/tokenizers/fineweb_1024_bpe.model \
    80 1
else
  bash ./stage_runpod_data_archive.sh \
    "$REMOTE_ROOT/data/archives/$archive_name" \
    /tmp/parameter-golf-data \
    fineweb10B_scylla_v2_cap0_fullbyte \
    scylla_v2_cap0_fullbyte
  bash ./verify_runpod_data_ready.sh \
    /tmp/parameter-golf-data/datasets/fineweb10B_scylla_v2_cap0_fullbyte \
    /tmp/parameter-golf-data/tokenizers/scylla_v2_cap0_fullbyte.meta.npz \
    11 1
fi

mkdir -p "$REMOTE_ROOT/records/scylla_2_claude/stage1_1gpu_runs"
nohup env \
  DATA_ROOT_MODE=tmp \
  NPROC_PER_NODE=1 \
  SEEDS=1337 \
  USE_COMPILE=0 \
  SUITE_ROOT="$REMOTE_ROOT/records/scylla_2_claude/stage1_1gpu_runs" \
  bash ./launch_scylla_stage1_suite_1gpu.sh "$role" \
  > "$REMOTE_ROOT/records/scylla_2_claude/stage1_1gpu_runs/${role}.launch.log" 2>&1 < /dev/null &
echo \$! > "$REMOTE_ROOT/records/scylla_2_claude/stage1_1gpu_runs/${role}.pid"
echo "started role=$role pid=\$(cat "$REMOTE_ROOT/records/scylla_2_claude/stage1_1gpu_runs/${role}.pid")"
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
  printf "role\tpod_name\tpod_id\thost\tport\tarchive_key\n" > "$MANIFEST_PATH"

  local -a pod_ids=()
  local -a hosts=()
  local -a ports=()
  local -a archive_urls=()

  for archive_key in "${ARCHIVE_KEYS[@]}"; do
    archive_urls+=("$(aws s3 presign "s3://$S3_BUCKET/$archive_key" --expires-in "$PRESIGN_TTL")")
  done

  for idx in "${!ROLE_NAMES[@]}"; do
    local role="${ROLE_NAMES[$idx]}"
    local pod_name="${POD_NAMES[$idx]}-$TIMESTAMP"
    local pod_json pod_id ssh_parts host port

    pod_json="$(create_pod "$pod_name")"
    pod_id="$(printf '%s' "$pod_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
    ssh_parts="$(wait_for_ssh "$pod_id")"
    host="$(echo "$ssh_parts" | awk '{print $1}')"
    port="$(echo "$ssh_parts" | awk '{print $2}')"

    pod_ids+=("$pod_id")
    hosts+=("$host")
    ports+=("$port")

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$role" "$pod_name" "$pod_id" "$host" "$port" "${ARCHIVE_KEYS[$idx]}" >> "$MANIFEST_PATH"
  done

  for idx in "${!ROLE_NAMES[@]}"; do
    (
      rsync_repo "${hosts[$idx]}" "${ports[$idx]}"
      remote_stage_and_launch \
        "${hosts[$idx]}" \
        "${ports[$idx]}" \
        "${ROLE_NAMES[$idx]}" \
        "${archive_urls[$idx]}" \
        "${ARCHIVE_NAMES[$idx]}"
    ) &
  done
  wait

  echo "Launched 4 Stage 1 pods."
  echo "Manifest: $MANIFEST_PATH"
  cat "$MANIFEST_PATH"
}

main "$@"
