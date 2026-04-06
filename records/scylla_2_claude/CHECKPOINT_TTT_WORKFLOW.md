# Checkpoint TTT Workflow

This lane now supports splitting Scylla work into two phases:

1. Train once on expensive hardware, typically `8xH100`, and save artifacts.
2. Re-run the quantized eval stack, including legal score-first TTT, on cheaper hardware, typically `1xH100`.

## Why

TTT tuning is an eval-time optimization problem, not a training problem.

The expensive part is producing the trained `12288` model under the 10-minute cap.
Once that run finishes, most of the follow-up cost is in reloading the same quantized submission artifact and sweeping:

- `TTT_LR`
- `TTT_EPOCHS`
- `TTT_CHUNK_BYTES`
- `TTT_FREEZE_BLOCKS`
- `TTT_FREEZE_EMBEDDINGS`
- `TTT_ADAPTER_RANK`
- `TTT_ADAPTER_LAST_N`
- `TTT_ADAPTER_SCALE`

## Exact Artifact Boundary

The training script saves:

- `final_model.pt`
- `final_model.int6.ptz`

The score-path eval reloads `final_model.int6.ptz`, dequantizes it, then runs:

- `final_int6_roundtrip`
- `final_int6_sliding_window`
- `legal_ttt`

That means TTT sweeps should use `final_model.int6.ptz`, not `final_model.pt`, because the competition score depends on the quantized artifact path.

## New Eval-Only Mode

`records/scylla_2_claude/train_gpt_legal_ttt.py` now supports:

- `EVAL_ONLY=1`
- `LOAD_INT6_PATH=/abs/path/to/final_model.int6.ptz`

In this mode the script:

- skips training
- loads the saved quantized artifact
- reconstructs the eval model
- runs the normal quantized eval stack and legal TTT

## Recommended Usage

### Phase 1: 8x training run

Use the normal case launcher with `NPROC_PER_NODE=8`.

Example:

```bash
NPROC_PER_NODE=8 \
SEEDS=2026 \
USE_COMPILE=1 \
MAX_WALLCLOCK_SECONDS=600 \
SUITE_ROOT=/workspace/parameter-golf/records/scylla_2_claude/phase6_train_8x \
bash ./launch_scylla_frontier_plan.sh ladder_12288
```

The saved artifact of interest will be:

```text
.../phase4/ladder_12288/seed_2026/final_model.int6.ptz
```

or under whatever `SUITE_ROOT` was used.

### Phase 2: 1x eval-only TTT sweep

Use the eval-only wrapper:

```bash
SEEDS=2026 \
USE_COMPILE=0 \
TTT_LR=0.0008 \
TTT_CHUNK_BYTES=196608 \
TTT_FREEZE_BLOCKS=2 \
bash ./launch_scylla_checkpoint_ttt_eval.sh \
  ladder_12288 \
  /abs/path/to/final_model.int6.ptz
```

The wrapper writes results under:

```text
records/scylla_2_claude/checkpoint_ttt_eval_runs/
```

unless `SUITE_ROOT` is overridden.

## Notes

- Keep the tokenizer bundle and dataset bundle identical between the 8x train run and the 1x eval-only sweep.
- Use the same case name when replaying the artifact so architecture and export settings stay aligned.
- `LOAD_FP32_PATH` exists only as a diagnostic breadcrumb right now; the intended tuning path is the quantized artifact.
- After choosing the best TTT settings on `1xH100`, do one clean end-to-end confirmation run with those settings on the real training setup.
