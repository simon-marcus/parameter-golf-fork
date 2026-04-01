#!/bin/bash
set -euo pipefail

VARIANT="${1:-control}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}"
VERIFY_SCRIPT="/workspace/parameter-golf/verify_runpod_data_ready.sh"
STAGE_SCRIPT="/workspace/parameter-golf/setup_local_parity_data_runpod.sh"
SCRIPT_PATH="/workspace/parameter-golf/train_gpt_leader_stack_jepa.py"
RECORD_ROOT="/workspace/parameter-golf/records/nonrecord_leader_stack_jepa"
ARCHIVE_PATH="${ARCHIVE_PATH:-/workspace/parameter-golf/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst}"
REQUIRE_LOCAL_ARCHIVE_FOR_MULTI_GPU="${REQUIRE_LOCAL_ARCHIVE_FOR_MULTI_GPU:-1}"

if [[ "$REQUIRE_LOCAL_ARCHIVE_FOR_MULTI_GPU" != "0" && "$DATA_ROOT_MODE" == "tmp" && "$NPROC_PER_NODE" -gt 1 ]]; then
  if [[ ! -f "$ARCHIVE_PATH" ]]; then
    echo "refusing multi-gpu launch without local archive: $ARCHIVE_PATH" >&2
    echo "stage the sp1024 archive onto this pod first, then rerun" >&2
    exit 1
  fi
fi

case "$DATA_ROOT_MODE" in
  workspace)
    DATA_PREFIX="/workspace/parameter-golf/data"
    ;;
  tmp)
    bash "$STAGE_SCRIPT" /workspace/parameter-golf/data /tmp/parameter-golf-data fineweb10B_sp1024 fineweb_1024_bpe
    DATA_PREFIX="/tmp/parameter-golf-data"
    ;;
  *)
    echo "DATA_ROOT_MODE must be one of: workspace, tmp" >&2
    exit 1
    ;;
esac

export DATA_PATH="$DATA_PREFIX/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$DATA_PREFIX/tokenizers/fineweb_1024_bpe.model"
bash "$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_PATH"

export USE_COMPILE="${USE_COMPILE:-0}"
export TTT_ENABLED=0
export MAX_WALLCLOCK_SECONDS
export NPROC_PER_NODE
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"

case "$VARIANT" in
  control)
    export RUN_ID="${RUN_ID:-leader_stack_jepa_control}"
    export JEPA_LOSS_WEIGHT=0.0
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_control"
    ;;
  jepa01)
    export RUN_ID="${RUN_ID:-leader_stack_jepa_jepa01}"
    export JEPA_LOSS_WEIGHT=0.1
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_jepa01"
    ;;
  weight:*)
    WEIGHT_VALUE="${VARIANT#weight:}"
    WEIGHT_TAG="${WEIGHT_VALUE//./p}"
    export RUN_ID="${RUN_ID:-leader_stack_jepa_w${WEIGHT_TAG}}"
    export JEPA_LOSS_WEIGHT="$WEIGHT_VALUE"
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_w${WEIGHT_TAG}"
    ;;
  *)
    echo "usage: $0 {control|jepa01|weight:<float>}" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"
LOG_PATH="$OUT_DIR/train.log"

{
  echo "RUN_ID=$RUN_ID"
  echo "DATA_ROOT_MODE=$DATA_ROOT_MODE"
  echo "DATA_PATH=$DATA_PATH"
  echo "TOKENIZER_PATH=$TOKENIZER_PATH"
  echo "JEPA_LOSS_WEIGHT=$JEPA_LOSS_WEIGHT"
  echo "NPROC_PER_NODE=$NPROC_PER_NODE"
  echo "MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
  echo "USE_COMPILE=$USE_COMPILE"
  echo "TTT_ENABLED=$TTT_ENABLED"
  echo "ARCHIVE_PATH=$ARCHIVE_PATH"
  echo "REQUIRE_LOCAL_ARCHIVE_FOR_MULTI_GPU=$REQUIRE_LOCAL_ARCHIVE_FOR_MULTI_GPU"
  echo "python3 -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE $SCRIPT_PATH"
} >"$LOG_PATH"

python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_PATH" >>"$LOG_PATH" 2>&1
