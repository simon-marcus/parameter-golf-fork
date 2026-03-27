#!/usr/bin/env bash
# Launch leave-one-out PPM N-gram Rescore follow-up
# Usage: bash launch.sh [base|smoke]
set -euo pipefail

MODE="${1:-base}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"

# Shared defaults
export DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
export COMPLEMENT_ENABLED="${COMPLEMENT_ENABLED:-1}"
export COMPLEMENT_ALPHA="${COMPLEMENT_ALPHA:-0.5}"
export NGRAM_ENABLED="${NGRAM_ENABLED:-1}"
export NGRAM_MIN_ORDER="${NGRAM_MIN_ORDER:-2}"
export NGRAM_MAX_ORDER="${NGRAM_MAX_ORDER:-12}"
export NGRAM_NUM_BUCKETS="${NGRAM_NUM_BUCKETS:-4194304}"
export NGRAM_CHUNK_SIZE="${NGRAM_CHUNK_SIZE:-512}"
export NGRAM_ALPHA_MIN="${NGRAM_ALPHA_MIN:-0.05}"
export NGRAM_ALPHA_MAX="${NGRAM_ALPHA_MAX:-0.70}"
export NGRAM_ENTROPY_CENTER="${NGRAM_ENTROPY_CENTER:-3.0}"
export NGRAM_ENTROPY_SCALE="${NGRAM_ENTROPY_SCALE:-2.0}"
export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-2}"
export NGRAM_LEAVE_ONE_OUT="${NGRAM_LEAVE_ONE_OUT:-1}"
export TTT_ENABLED="${TTT_ENABLED:-0}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"

# Data paths
if [[ "${DATA_ROOT_MODE}" == "tmp" ]]; then
    DATA_BASE="/tmp/parameter-golf-data"
else
    DATA_BASE="/workspace/parameter-golf/data"
fi
export DATA_PATH="${DATA_PATH:-${DATA_BASE}/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${DATA_BASE}/tokenizers/fineweb_1024_bpe.model}"

case "$MODE" in
    smoke)
        echo "=== SMOKE TEST (1xGPU, 180s, USE_COMPILE=0) ==="
        export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
        export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}"
        export USE_COMPILE="${USE_COMPILE:-0}"
        export NGRAM_MAX_ORDER="${NGRAM_MAX_ORDER:-9}"
        export NGRAM_NUM_BUCKETS="${NGRAM_NUM_BUCKETS:-4194304}"
        ;;
    base)
        echo "=== FULL RUN (8xGPU, 600s) ==="
        export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
        export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
        export USE_COMPILE="${USE_COMPILE:-1}"
        ;;
    *)
        echo "Unknown mode: $MODE (use 'base' or 'smoke')"
        exit 1
        ;;
esac

# Verify data
if [[ -f "/workspace/parameter-golf/verify_runpod_data_ready.sh" ]]; then
    bash /workspace/parameter-golf/verify_runpod_data_ready.sh "$DATA_PATH" "$TOKENIZER_PATH"
fi

echo "Train script: $TRAIN_SCRIPT"
echo "Data path: $DATA_PATH"
echo "NGRAM: orders=${NGRAM_MIN_ORDER}-${NGRAM_MAX_ORDER} buckets=${NGRAM_NUM_BUCKETS} alpha=[${NGRAM_ALPHA_MIN},${NGRAM_ALPHA_MAX}] leave_one_out=${NGRAM_LEAVE_ONE_OUT}"
echo "COMPLEMENT: enabled=${COMPLEMENT_ENABLED} alpha=${COMPLEMENT_ALPHA}"

NPROC="${NPROC_PER_NODE:-8}"
if [[ "$NPROC" -eq 1 ]]; then
    python3 "$TRAIN_SCRIPT"
else
    torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN_SCRIPT"
fi
