#!/bin/bash
set -euo pipefail

SEEDS="${SEEDS:-1338 1339}"
ROOT="/workspace/parameter-golf"
RECORD_ROOT="$ROOT/records/nonrecord_leader_stack_jepa"
TRAIN_VARIANT="${TRAIN_VARIANT:-jepa01}"

mkdir -p "$RECORD_ROOT"

run_one() {
  local seed="$1"
  local run_tag="seed${seed}"
  local train_driver="$RECORD_ROOT/${run_tag}_train_driver.log"
  local storage_log="$RECORD_ROOT/${run_tag}_storage_pass.log"

  cd "$ROOT"
  rm -f final_model.pt final_model.int6.ptz

  echo "[$(date)] train start seed=$seed variant=$TRAIN_VARIANT" | tee -a "$train_driver"
  env \
    RUN_ID="leader_stack_jepa_${run_tag}" \
    SEED="$seed" \
    DATA_ROOT_MODE=tmp \
    NPROC_PER_NODE=8 \
    MAX_WALLCLOCK_SECONDS=600 \
    VAL_LOSS_EVERY=200 \
    TRAIN_LOG_EVERY=50 \
    USE_COMPILE=0 \
    bash ./launch_leader_stack_jepa_screen_runpod.sh "$TRAIN_VARIANT" >>"$train_driver" 2>&1

  cp final_model.pt "$RECORD_ROOT/final_model_${run_tag}.pt"
  cp final_model.int6.ptz "$RECORD_ROOT/final_model_${run_tag}.int6.ptz"

  echo "[$(date)] storage pass start seed=$seed" | tee -a "$storage_log"
  env \
    DATA_PATH=/tmp/parameter-golf-data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=/tmp/parameter-golf-data/tokenizers/fineweb_1024_bpe.model \
    NPROC_PER_NODE=8 \
    QUANT_CATS=mlp,attn,embed,other \
    DROP_JEPA_ALIAS_EXPORT=1 \
    CHECKPOINT_PATH="$RECORD_ROOT/final_model_${run_tag}.pt" \
    python3 -m torch.distributed.run --standalone --nproc_per_node=8 "$ROOT/leader_stack_storage_pass.py" >>"$storage_log" 2>&1
}

for seed in $SEEDS; do
  run_one "$seed"
done

