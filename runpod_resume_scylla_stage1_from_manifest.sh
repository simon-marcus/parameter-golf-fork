#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="/workspace/parameter-golf"
S3_BUCKET="parameter-golf-staging-094651608775"
PRESIGN_TTL="${PRESIGN_TTL:-43200}"
MANIFEST_PATH="${1:?usage: runpod_resume_scylla_stage1_from_manifest.sh <manifest.tsv>}"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/.env.local"
set +a

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
  local archive_key="$4"
  local archive_name
  archive_name="$(basename "$archive_key")"
  local archive_url
  archive_url="$(aws s3 presign "s3://$S3_BUCKET/$archive_key" --expires-in "$PRESIGN_TTL")"

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

while IFS=$'\t' read -r role pod_name pod_id host port archive_key; do
  (
    echo "syncing role=$role host=$host port=$port"
    rsync_repo "$host" "$port"
    echo "staging+launching role=$role host=$host"
    remote_stage_and_launch "$host" "$port" "$role" "$archive_key"
  ) &
done < <(tail -n +2 "$MANIFEST_PATH")
wait

echo "Resumed Stage 1 setup for pods in $MANIFEST_PATH"
