#!/bin/bash
set -euo pipefail

VARIANT="${1:-jepa01}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
VERIFY_SCRIPT="/workspace/parameter-golf/verify_runpod_data_ready.sh"
STAGE_SCRIPT="/workspace/parameter-golf/setup_byte260_data_runpod.sh"
SCRIPT_PATH="/workspace/parameter-golf/train_jepa_baseline.py"
RECORD_ROOT="/workspace/parameter-golf/records/track_10min_16mb/2026-03-30_JEPAIsolationRunpod"

case "$DATA_ROOT_MODE" in
  workspace)
    DATA_PREFIX="/workspace/parameter-golf/data"
    ;;
  tmp)
    bash "$STAGE_SCRIPT" /workspace/parameter-golf/data /tmp/parameter-golf-data
    DATA_PREFIX="/tmp/parameter-golf-data"
    ;;
  *)
    echo "DATA_ROOT_MODE must be one of: workspace, tmp" >&2
    exit 1
    ;;
esac

export DATA_PATH="$DATA_PREFIX/datasets/fineweb10B_byte260"
export TOKENIZER_PATH="$DATA_PREFIX/tokenizers/fineweb_pure_byte_260.json"
bash "$VERIFY_SCRIPT" "$DATA_PATH" "$TOKENIZER_PATH" 3 2

export TOKENIZER_KIND=byte
export USE_COMPILE="${USE_COMPILE:-0}"
export MAX_WALLCLOCK_SECONDS
export NPROC_PER_NODE
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export MODEL_EMA_ENABLED="${MODEL_EMA_ENABLED:-1}"
export MODEL_EMA_DECAY="${MODEL_EMA_DECAY:-0.997}"
export SWA_ENABLED="${SWA_ENABLED:-0}"
export SWA_EVERY="${SWA_EVERY:-50}"
export XSA_ALL="${XSA_ALL:-0}"

case "$VARIANT" in
  control)
    export RUN_ID="${RUN_ID:-jepa_iso_control_runpod}"
    export JEPA_LOSS_WEIGHT=0.0
    export NUM_LAYERS="${NUM_LAYERS:-9}"
    export MODEL_DIM="${MODEL_DIM:-512}"
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_control"
    ;;
  jepa01)
    export RUN_ID="${RUN_ID:-jepa_iso_jepa01_runpod}"
    export JEPA_LOSS_WEIGHT=0.1
    export NUM_LAYERS="${NUM_LAYERS:-9}"
    export MODEL_DIM="${MODEL_DIM:-512}"
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_jepa01"
    ;;
  jepa02)
    export RUN_ID="${RUN_ID:-jepa_iso_jepa02_runpod}"
    export JEPA_LOSS_WEIGHT=0.2
    export NUM_LAYERS="${NUM_LAYERS:-9}"
    export MODEL_DIM="${MODEL_DIM:-512}"
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_jepa02"
    ;;
  xsa_control)
    export RUN_ID="${RUN_ID:-jepa_iso_xsa_control_runpod}"
    export JEPA_LOSS_WEIGHT=0.0
    export NUM_LAYERS="${NUM_LAYERS:-9}"
    export MODEL_DIM="${MODEL_DIM:-512}"
    export XSA_ALL=1
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_xsa_control"
    ;;
  xsa_jepa01)
    export RUN_ID="${RUN_ID:-jepa_iso_xsa_jepa01_runpod}"
    export JEPA_LOSS_WEIGHT=0.1
    export NUM_LAYERS="${NUM_LAYERS:-9}"
    export MODEL_DIM="${MODEL_DIM:-512}"
    export XSA_ALL=1
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_xsa_jepa01"
    ;;
  depth12_control)
    export RUN_ID="${RUN_ID:-jepa_iso_depth12_control_runpod}"
    export JEPA_LOSS_WEIGHT=0.0
    export NUM_LAYERS="${NUM_LAYERS:-12}"
    export MODEL_DIM="${MODEL_DIM:-512}"
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_depth12_control"
    ;;
  depth12_jepa01)
    export RUN_ID="${RUN_ID:-jepa_iso_depth12_jepa01_runpod}"
    export JEPA_LOSS_WEIGHT=0.1
    export NUM_LAYERS="${NUM_LAYERS:-12}"
    export MODEL_DIM="${MODEL_DIM:-512}"
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_depth12_jepa01"
    ;;
  depth12wide_control)
    export RUN_ID="${RUN_ID:-jepa_iso_depth12wide_control_runpod}"
    export JEPA_LOSS_WEIGHT=0.0
    export NUM_LAYERS="${NUM_LAYERS:-12}"
    export MODEL_DIM="${MODEL_DIM:-576}"
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_depth12wide_control"
    ;;
  depth12wide_jepa01)
    export RUN_ID="${RUN_ID:-jepa_iso_depth12wide_jepa01_runpod}"
    export JEPA_LOSS_WEIGHT=0.1
    export NUM_LAYERS="${NUM_LAYERS:-12}"
    export MODEL_DIM="${MODEL_DIM:-576}"
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_depth12wide_jepa01"
    ;;
  weight:*)
    WEIGHT_VALUE="${VARIANT#weight:}"
    WEIGHT_TAG="${WEIGHT_VALUE//./p}"
    export RUN_ID="${RUN_ID:-jepa_iso_w${WEIGHT_TAG}_runpod}"
    export JEPA_LOSS_WEIGHT="$WEIGHT_VALUE"
    export NUM_LAYERS="${NUM_LAYERS:-9}"
    export MODEL_DIM="${MODEL_DIM:-512}"
    OUT_DIR="$RECORD_ROOT/${DATA_ROOT_MODE}_w${WEIGHT_TAG}"
    ;;
  *)
    echo "usage: $0 {control|jepa01|jepa02|xsa_control|xsa_jepa01|depth12_control|depth12_jepa01|depth12wide_control|depth12wide_jepa01|weight:<float>}" >&2
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
  echo "TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS"
  echo "TRAIN_SEQ_LEN=$TRAIN_SEQ_LEN"
  echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
  echo "NUM_LAYERS=$NUM_LAYERS"
  echo "MODEL_DIM=$MODEL_DIM"
  echo "XSA_ALL=$XSA_ALL"
  echo "USE_COMPILE=$USE_COMPILE"
  echo "MODEL_EMA_ENABLED=$MODEL_EMA_ENABLED"
  echo "MODEL_EMA_DECAY=$MODEL_EMA_DECAY"
  echo "SWA_ENABLED=$SWA_ENABLED"
  echo "SWA_EVERY=$SWA_EVERY"
  echo "python3 -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE $SCRIPT_PATH"
} >"$LOG_PATH"

python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_PATH" >>"$LOG_PATH" 2>&1
