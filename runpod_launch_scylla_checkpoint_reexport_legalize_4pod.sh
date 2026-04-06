#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
CANONICAL_REPO_ROOT="${CANONICAL_REPO_ROOT:-$HOME/Code/parameter-golf}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/records/scylla_2_claude/runpod_checkpoint_reexport_manifest}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MANIFEST_PATH="$MANIFEST_DIR/$TIMESTAMP.tsv"
REMOTE_ROOT="/workspace/parameter-golf"
S3_BUCKET="${S3_BUCKET:-parameter-golf-staging-094651608775}"
PRESIGN_TTL="${PRESIGN_TTL:-43200}"
SUITE_ROOT_REMOTE="${SUITE_ROOT_REMOTE:-$REMOTE_ROOT/records/scylla_2_claude/checkpoint_reexport_eval_runs}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
ITERATIONS="${ITERATIONS:-1}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
USE_COMPILE="${USE_COMPILE:-0}"
SEEDS="${SEEDS:-2027}"
CASE_NAME="${CASE_NAME:-ladder_12288}"
VOCAB="${VOCAB:-12288}"
TRAIN_SHARDS="${TRAIN_SHARDS:-8}"
VAL_SHARDS="${VAL_SHARDS:-1}"
VAL_TOKENS_LIMIT="${VAL_TOKENS_LIMIT:-1048576}"
LOCAL_FP32_PATH="${LOCAL_FP32_PATH:-$HOME/Code/parameter-golf/runpod_artifacts/20260406T203530Z_checkpoint_ttt_train_8x_full10m/pg-scylla-12288-8x/final_model.pt}"

VARIANT_NAMES=(control embed6 matrix5_embed6 embed6_quantile)
POD_NAMES=(
  "pg-scylla-legalize-control"
  "pg-scylla-legalize-embed6"
  "pg-scylla-legalize-m5e6"
  "pg-scylla-legalize-e6q"
)
MATRIX_BITS_VALUES=(6 6 5 6)
EMBED_BITS_VALUES=(8 6 6 6)
MATRIX_CLIP_MODE_VALUES=(std std std quantile)
EMBED_CLIP_MODE_VALUES=(std std std quantile)
COMPRESSOR_VALUES=(lzma lzma lzma lzma)
BYTE_SHUFFLE_VALUES=(0 0 0 0)

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
    --exclude 'records/scylla_2_claude/checkpoint_ttt_train_8x' \
    --exclude 'records/scylla_2_claude/checkpoint_ttt_train_8x_full10m' \
    --exclude 'records/scylla_2_claude/checkpoint_ttt_eval_runs' \
    --exclude 'records/scylla_2_claude/checkpoint_reexport_eval_runs' \
    --exclude 'records/scylla_2_claude/pulled_*' \
    "$REPO_ROOT/" \
    "root@$host:$REMOTE_ROOT/"
}

copy_fp32_artifact() {
  local host="$1"
  local port="$2"
  local remote_dir="$3"
  scp -P "$port" -o StrictHostKeyChecking=no -i "$SSH_KEY_PATH" \
    "$LOCAL_FP32_PATH" \
    "root@$host:$remote_dir/final_model.pt"
}

remote_stage_and_launch() {
  local host="$1"
  local port="$2"
  local variant="$3"
  local matrix_bits="$4"
  local embed_bits="$5"
  local matrix_clip_mode="$6"
  local embed_clip_mode="$7"
  local compressor="$8"
  local byte_shuffle="$9"
  local archive_url="${10}"
  local archive_name="${11}"

  local dataset_name="fineweb10B_scylla_v2_v${VOCAB}"
  local tokenizer_name="scylla_v2_v${VOCAB}"
  local remote_artifact_dir="$SUITE_ROOT_REMOTE/artifacts/$variant"

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
  "$TRAIN_SHARDS" \
  "$VAL_SHARDS"

mkdir -p "$remote_artifact_dir"
nohup env \
  DATA_ROOT_MODE=tmp \
  NPROC_PER_NODE=1 \
  SEEDS="$SEEDS" \
  USE_COMPILE="$USE_COMPILE" \
  MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
  ITERATIONS="$ITERATIONS" \
  TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
  VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
  VAL_TOKENS_LIMIT="$VAL_TOKENS_LIMIT" \
  SUITE_ROOT="$SUITE_ROOT_REMOTE" \
  COMPRESSOR="$compressor" \
  BYTE_SHUFFLE="$byte_shuffle" \
  MATRIX_BITS="$matrix_bits" \
  EMBED_BITS="$embed_bits" \
  MATRIX_CLIP_MODE="$matrix_clip_mode" \
  EMBED_CLIP_MODE="$embed_clip_mode" \
  SCYLLA_V${VOCAB}_EXPECTED_TRAIN_SHARDS="$TRAIN_SHARDS" \
  SCYLLA_V${VOCAB}_EXPECTED_VAL_SHARDS="$VAL_SHARDS" \
  bash ./launch_scylla_checkpoint_reexport_eval.sh "$CASE_NAME" "$remote_artifact_dir/final_model.pt" \
  > "$SUITE_ROOT_REMOTE/${variant}.launch.log" 2>&1 < /dev/null &
echo \$! > "$SUITE_ROOT_REMOTE/${variant}.pid"
echo "started variant=$variant pid=\$(cat "$SUITE_ROOT_REMOTE/${variant}.pid")"
EOF
}

main() {
  require_cmd runpodctl
  require_cmd aws
  require_cmd rsync
  require_cmd scp
  require_cmd python3
  require_cmd curl
  load_runpod_env

  if [[ ! -f "$LOCAL_FP32_PATH" ]]; then
    echo "Missing LOCAL_FP32_PATH: $LOCAL_FP32_PATH" >&2
    exit 1
  fi

  mkdir -p "$MANIFEST_DIR"
  printf "variant\tpod_name\tpod_id\thost\tport\tseed\tcase\tvocab\ttrain_shards\tval_shards\tval_tokens_limit\tmatrix_bits\tembed_bits\tmatrix_clip_mode\tembed_clip_mode\tcompressor\tbyte_shuffle\tarchive_key\tlocal_fp32_path\n" > "$MANIFEST_PATH"

  local archive_key="data/archives/scylla_v2_v${VOCAB}.tar.zst"
  local archive_name="scylla_v2_v${VOCAB}.tar.zst"
  local archive_url
  archive_url="$(aws s3 presign "s3://$S3_BUCKET/$archive_key" --expires-in "$PRESIGN_TTL")"

  local -a hosts=()
  local -a ports=()

  local idx pod_name pod_json pod_id ssh_parts host port
  for idx in "${!VARIANT_NAMES[@]}"; do
    pod_name="${POD_NAMES[$idx]}-$TIMESTAMP"
    pod_json="$(create_pod "$pod_name")"
    pod_id="$(printf '%s' "$pod_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
    ssh_parts="$(wait_for_ssh "$pod_id")"
    host="$(echo "$ssh_parts" | awk '{print $1}')"
    port="$(echo "$ssh_parts" | awk '{print $2}')"

    hosts+=("$host")
    ports+=("$port")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${VARIANT_NAMES[$idx]}" "$pod_name" "$pod_id" "$host" "$port" \
      "$SEEDS" "$CASE_NAME" "$VOCAB" "$TRAIN_SHARDS" "$VAL_SHARDS" "$VAL_TOKENS_LIMIT" \
      "${MATRIX_BITS_VALUES[$idx]}" "${EMBED_BITS_VALUES[$idx]}" \
      "${MATRIX_CLIP_MODE_VALUES[$idx]}" "${EMBED_CLIP_MODE_VALUES[$idx]}" \
      "${COMPRESSOR_VALUES[$idx]}" "${BYTE_SHUFFLE_VALUES[$idx]}" \
      "$archive_key" "$LOCAL_FP32_PATH" >> "$MANIFEST_PATH"
  done

  for idx in "${!VARIANT_NAMES[@]}"; do
    (
      rsync_repo "${hosts[$idx]}" "${ports[$idx]}"
      ssh -p "${ports[$idx]}" -o StrictHostKeyChecking=no -i "$SSH_KEY_PATH" "root@${hosts[$idx]}" \
        "mkdir -p '$SUITE_ROOT_REMOTE/artifacts/${VARIANT_NAMES[$idx]}'"
      copy_fp32_artifact "${hosts[$idx]}" "${ports[$idx]}" "$SUITE_ROOT_REMOTE/artifacts/${VARIANT_NAMES[$idx]}"
      remote_stage_and_launch \
        "${hosts[$idx]}" \
        "${ports[$idx]}" \
        "${VARIANT_NAMES[$idx]}" \
        "${MATRIX_BITS_VALUES[$idx]}" \
        "${EMBED_BITS_VALUES[$idx]}" \
        "${MATRIX_CLIP_MODE_VALUES[$idx]}" \
        "${EMBED_CLIP_MODE_VALUES[$idx]}" \
        "${COMPRESSOR_VALUES[$idx]}" \
        "${BYTE_SHUFFLE_VALUES[$idx]}" \
        "$archive_url" \
        "$archive_name"
    ) &
  done
  wait

  echo "Launched 4 checkpoint re-export legalization pods."
  echo "Manifest: $MANIFEST_PATH"
  cat "$MANIFEST_PATH"
}

main "$@"
