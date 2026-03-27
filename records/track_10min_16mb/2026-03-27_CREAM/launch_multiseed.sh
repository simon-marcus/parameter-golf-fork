#!/usr/bin/env bash
# Launch the leave-one-out PPM candidate across the standard 3-seed package.
# Usage:
#   bash launch_multiseed.sh            # seeds 1337,42,2025
#   SEEDS=1337,42 bash launch_multiseed.sh
#   MODE=smoke bash launch_multiseed.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-base}"
SEEDS_CSV="${SEEDS:-1337,42,2025}"

IFS=',' read -r -a SEEDS_ARR <<< "$SEEDS_CSV"

echo "mode=$MODE"
echo "seeds=${SEEDS_CSV}"
echo "leave_one_out=${NGRAM_LEAVE_ONE_OUT:-1}"

for seed in "${SEEDS_ARR[@]}"; do
    seed="$(echo "$seed" | xargs)"
    if [[ -z "$seed" ]]; then
        continue
    fi
    export SEED="$seed"
    export RUN_ID="cream_seed${seed}"
    echo
    echo "=== seed ${seed} ==="
    bash "$SCRIPT_DIR/launch.sh" "$MODE"
done
