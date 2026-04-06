# Competitive Execution Plan

This is the current best path to advance the Scylla entry without drifting into proxy work that does not move the real leaderboard score.

## Current Read

- `12288` is the best current Scylla vocab point from the `1xH100` ladder.
- The win is end-to-end, not purely pre-TTT.
- The strongest current `8xH100` `12288` checkpoint is over cap.
- Small-val checkpoint TTT sweeps are useful as smoke tests only. They are not the decision metric.

That means the next objective is not another vocab search. It is to turn `12288` into a legal competitive candidate.

## Priority Order

1. Legalize `12288`
2. Confirm `12288` on the real `8xH100` training regime
3. Tune full legal TTT on a legal `12288` artifact

## Phase 1: Legalize `12288` Cheaply

Use the saved EMA-applied fp32 checkpoint and re-export it with different artifact settings.

Why:

- This is the cheapest accurate way to study artifact legality.
- It gives exact submission size for each export setting.
- It gives the exact post-quantization score path for that checkpoint.
- It avoids spending another `8xH100` run before we know which export settings are viable.

New support:

- `records/scylla_2_claude/train_gpt_legal_ttt.py` now supports:
  - `EVAL_ONLY=1`
  - `LOAD_FP32_PATH=/abs/path/to/final_model.pt`
- Wrapper:
  - `launch_scylla_checkpoint_reexport_eval.sh`

Recommended first 4-way legalization sweep on `1xH100`:

1. control
   - `COMPRESSOR=lzma`
   - `BYTE_SHUFFLE=0`
   - `MATRIX_BITS=6`
   - `EMBED_BITS=8`
   - `MATRIX_CLIP_MODE=std`
   - `EMBED_CLIP_MODE=std`
2. embed6
   - same as control
   - `EMBED_BITS=6`
3. matrix5_embed6
   - `MATRIX_BITS=5`
   - `EMBED_BITS=6`
4. embed6_quantile
   - `MATRIX_BITS=6`
   - `EMBED_BITS=6`
   - `MATRIX_CLIP_MODE=quantile`
   - `EMBED_CLIP_MODE=quantile`

Checkpoint to use:

- `/Users/simon/Code/parameter-golf/runpod_artifacts/20260406T203530Z_checkpoint_ttt_train_8x_full10m/pg-scylla-12288-8x/final_model.pt`

Screening rule:

- Keep the top 1-2 settings that get under or near the cap with the smallest regression.
- Artifact size matters exactly here.
- Small-val score is acceptable for triage, but promote finalists to full-val eval before making a final export choice.

## Phase 2: Full-Val Re-Export Confirmation

For the best 1-2 export settings from Phase 1:

- rerun re-export eval with `VAL_TOKENS_LIMIT=0`
- keep `EVAL_ONLY=1`
- keep `LOAD_FP32_PATH` pointing to the same saved `12288` fp32 checkpoint

Why:

- This is still cheaper than retraining.
- It gives the real legal score path for the export setting on the existing checkpoint.
- It avoids promoting a weak export setting to `8xH100`.

## Phase 3: Reuse `qq0b5j0fu726ee` For `8xH100` Confirmation

Pod:

- `qq0b5j0fu726ee`

Use it once we have a legal or near-legal `12288` export recipe worth promoting.

Recommended `8xH100` confirm run:

- `ladder_12288`
- selected export settings from Phase 1/2
- raise `ITERATIONS` enough that the run is wallclock-limited, not artificially step-limited
- keep the run otherwise as close as possible to the current best `12288` stack

Goal:

- confirm that `12288` remains strong on the real `8xH100`, 10-minute regime with the legal export settings

Fallback:

- if `12288` cannot be made legal without too much score loss, fall back to `6144`

## Phase 4: Full Legal TTT Tuning

Only do this on a legal candidate.

Do not use the small-val proxy as the decision metric.

Recommended first full-val TTT sweep:

1. control
   - `VAL_TOKENS_LIMIT=0`
   - `TTT_FREEZE_BLOCKS=2`
   - `TTT_LR=0.0010`
2. freeze0_lr20
   - `VAL_TOKENS_LIMIT=0`
   - `TTT_FREEZE_BLOCKS=0`
   - `TTT_LR=0.0020`
3. freeze1_lr20
   - `VAL_TOKENS_LIMIT=0`
   - `TTT_FREEZE_BLOCKS=1`
   - `TTT_LR=0.0020`
4. freeze0_lr15
   - `VAL_TOKENS_LIMIT=0`
   - `TTT_FREEZE_BLOCKS=0`
   - `TTT_LR=0.0015`

Why this order:

- the strongest historical legal-TTT runs were more aggressive than the current Scylla defaults
- freeze and learning-rate are higher-value axes than chunk geometry
- chunk geometry should be revisited only after the best freeze/LR regime is clearer

## What Not To Do

- Do not spend more runs on broader vocab exploration right now.
- Do not treat small-val TTT sweeps as the deciding metric.
- Do not spend another expensive `8xH100` run on an over-cap candidate if export settings have not been screened first.

## Bottom Line

The Scylla tokenizer discovery already happened.

The best next move is:

1. cheap exact legalization via fp32 re-export
2. real `8xH100` promotion of the best legal `12288`
3. full-val legal-TTT tuning on that legal candidate
