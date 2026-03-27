#!/usr/bin/env bash
# Launch BeatCacheMoney: Two-pass LOO + Phrase Cache + Calibration + T=0.85
# Target: val_bpb < 0.080 (beat PR #933 CacheMoney)
# Usage: bash launch.sh [base|smoke]
set -euo pipefail

MODE="${1:-base}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"

# Use default 11L/512d model (strong neural base at 1.12 BPB)
# N-gram + phrase cache defaults are in train_gpt.py Hyperparameters
export DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
export NGRAM_ENABLED="${NGRAM_ENABLED:-1}"
export PHRASE_ENABLED="${PHRASE_ENABLED:-0}"  # disabled: 4M buckets + 62M tokens = noise
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
        # Reduce cache sizes for fast iteration
        export NGRAM_MAX_ORDER="${NGRAM_MAX_ORDER:-12}"
        export NGRAM_NUM_BUCKETS="${NGRAM_NUM_BUCKETS:-8388608}"
        ;;
    base)
        echo "=== FULL RUN (8xGPU, 600s) ==="
        export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
        export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
        export USE_COMPILE="${USE_COMPILE:-1}"
        # Full cache settings (defaults from train_gpt.py):
        # NGRAM_MAX_ORDER=20, NGRAM_NUM_BUCKETS=32M
        # PHRASE_PROBE_LENGTHS=64,56,48,36,28,20,16, PHRASE_NUM_BUCKETS=4M
        # NGRAM_TEMPERATURE=0.85, calibration enabled
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
echo "NGRAM: orders=${NGRAM_MIN_ORDER:-2}-${NGRAM_MAX_ORDER:-20} buckets=${NGRAM_NUM_BUCKETS:-33554432} temp=${NGRAM_TEMPERATURE:-0.85} loo=${NGRAM_LEAVE_ONE_OUT:-1}"
echo "PHRASE: enabled=${PHRASE_ENABLED:-0}"
echo "CALIB: enabled=${NGRAM_CALIBRATION_ENABLED:-1} frac=${NGRAM_CALIBRATION_FRAC:-0.05} alpha_grid=${NGRAM_CALIBRATION_ALPHA_MAX_GRID:-0.70,0.80,0.90,0.95,0.99}"

NPROC="${NPROC_PER_NODE:-8}"
if [[ "$NPROC" -eq 1 ]]; then
    python3 "$TRAIN_SCRIPT"
else
    torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN_SCRIPT"
fi
