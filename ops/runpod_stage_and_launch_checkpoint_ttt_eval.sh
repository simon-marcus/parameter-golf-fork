#!/bin/bash
set -euo pipefail

HOST="${1:?usage: $0 <host> <port> <variant> <ttt_chunk_bytes> <ttt_lr> [seed] [local_int6_path]}"
PORT="${2:?usage: $0 <host> <port> <variant> <ttt_chunk_bytes> <ttt_lr> [seed] [local_int6_path]}"
VARIANT="${3:?usage: $0 <host> <port> <variant> <ttt_chunk_bytes> <ttt_lr> [seed] [local_int6_path]}"
TTT_CHUNK_BYTES="${4:?usage: $0 <host> <port> <variant> <ttt_chunk_bytes> <ttt_lr> [seed] [local_int6_path]}"
TTT_LR="${5:?usage: $0 <host> <port> <variant> <ttt_chunk_bytes> <ttt_lr> [seed] [local_int6_path]}"
SEED="${6:-2027}"
LOCAL_INT6_PATH="${7:-$HOME/Code/parameter-golf/runpod_artifacts/20260406T203530Z_checkpoint_ttt_train_8x_full10m/pg-scylla-12288-8x/final_model.int6.ptz}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="/workspace/parameter-golf"
SUITE_ROOT_REMOTE="$REMOTE_ROOT/records/scylla_2_claude/checkpoint_ttt_eval_runs"
ARCHIVE_NAME="scylla_v2_v12288.tar.zst"
ARCHIVE_KEY="data/archives/$ARCHIVE_NAME"
S3_BUCKET="${S3_BUCKET:-parameter-golf-staging-094651608775}"
PRESIGN_TTL="${PRESIGN_TTL:-43200}"
COMPRESSOR="${COMPRESSOR:-lzma}"
BYTE_SHUFFLE="${BYTE_SHUFFLE:-0}"
REMOTE_ARTIFACT_DIR="$SUITE_ROOT_REMOTE/artifacts/$VARIANT"

if [[ ! -f "$LOCAL_INT6_PATH" ]]; then
  echo "Missing local int6 artifact: $LOCAL_INT6_PATH" >&2
  exit 1
fi

ARCHIVE_URL="$(aws s3 presign "s3://$S3_BUCKET/$ARCHIVE_KEY" --expires-in "$PRESIGN_TTL")"

ssh_base=(ssh -p "$PORT" -o StrictHostKeyChecking=no -i "$SSH_KEY_PATH" "root@$HOST")
scp_base=(scp -P "$PORT" -o StrictHostKeyChecking=no -i "$SSH_KEY_PATH")

"${ssh_base[@]}" "mkdir -p '$REMOTE_ARTIFACT_DIR' '$REMOTE_ROOT/data/archives'"
"${scp_base[@]}" "$LOCAL_INT6_PATH" "root@$HOST:$REMOTE_ARTIFACT_DIR/final_model.int6.ptz" >/dev/null

"${ssh_base[@]}" "URL='$ARCHIVE_URL' VARIANT='$VARIANT' TTT_CHUNK_BYTES='$TTT_CHUNK_BYTES' TTT_LR='$TTT_LR' SEED='$SEED' REMOTE_ROOT='$REMOTE_ROOT' SUITE_ROOT_REMOTE='$SUITE_ROOT_REMOTE' REMOTE_ARTIFACT_DIR='$REMOTE_ARTIFACT_DIR' ARCHIVE_NAME='$ARCHIVE_NAME' COMPRESSOR='$COMPRESSOR' BYTE_SHUFFLE='$BYTE_SHUFFLE' bash -s" <<'EOF'
set -euo pipefail
cd "$REMOTE_ROOT"

if [[ ! -f "data/archives/$ARCHIVE_NAME" ]]; then
  curl -L --fail "$URL" -o "data/archives/$ARCHIVE_NAME"
fi

bash ./stage_runpod_data_archive.sh \
  "data/archives/$ARCHIVE_NAME" \
  /tmp/parameter-golf-data \
  fineweb10B_scylla_v2_v12288 \
  scylla_v2_v12288

bash ./verify_runpod_data_ready.sh \
  /tmp/parameter-golf-data/datasets/fineweb10B_scylla_v2_v12288 \
  /tmp/parameter-golf-data/tokenizers/scylla_v2_v12288.meta.npz \
  8 \
  1

rm -rf "$SUITE_ROOT_REMOTE/phase4/ladder_12288/seed_$SEED"
rm -f "$SUITE_ROOT_REMOTE/$VARIANT.launch.log" "$SUITE_ROOT_REMOTE/$VARIANT.pid"

nohup env \
  DATA_ROOT_MODE=tmp \
  NPROC_PER_NODE=1 \
  SEEDS="$SEED" \
  USE_COMPILE=0 \
  MAX_WALLCLOCK_SECONDS=600 \
  ITERATIONS=1 \
  TRAIN_LOG_EVERY=50 \
  VAL_LOSS_EVERY=0 \
  COMPRESSOR="$COMPRESSOR" \
  BYTE_SHUFFLE="$BYTE_SHUFFLE" \
  SUITE_ROOT="$SUITE_ROOT_REMOTE" \
  TTT_CHUNK_BYTES="$TTT_CHUNK_BYTES" \
  TTT_LR="$TTT_LR" \
  SCYLLA_V12288_EXPECTED_TRAIN_SHARDS=8 \
  SCYLLA_V12288_EXPECTED_VAL_SHARDS=1 \
  bash ./launch_scylla_checkpoint_ttt_eval.sh \
    ladder_12288 \
    "$REMOTE_ARTIFACT_DIR/final_model.int6.ptz" \
  > "$SUITE_ROOT_REMOTE/$VARIANT.launch.log" 2>&1 < /dev/null &

echo $! > "$SUITE_ROOT_REMOTE/$VARIANT.pid"
sleep 2
cat "$SUITE_ROOT_REMOTE/$VARIANT.pid"
echo "---"
tail -n 20 "$SUITE_ROOT_REMOTE/$VARIANT.launch.log" || true
EOF
