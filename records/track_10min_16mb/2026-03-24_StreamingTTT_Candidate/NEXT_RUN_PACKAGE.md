# Next Run Package

This package operationalizes the parity/blame audit in [PARITY_BLAME_AUDIT.md](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/PARITY_BLAME_AUDIT.md).

Baseline assumption:
- `chunk_full + leaky_relu_sq` is the current best full 8x variant
- broad TTT exploration is paused
- the next cycle should target `post_ema` first
- the leader's own seed spread is larger than the tiny audit deltas we have measured

## Run 1

Name:
- `audit_parity`

Purpose:
- exact `chunk_full` recipe after the March 25 parity fixes
- restore leader-style in-training validation cadence and leader-style `train()/eval()` switching inside chunked legal TTT

Launcher:

```bash
DATA_ROOT_MODE=tmp bash /workspace/parameter-golf/launch_streaming_ttt_candidate_runpod.sh audit_parity
```

Effective settings versus prior `chunk_full`:
- same high-level recipe
- `VAL_LOSS_EVERY` falls back to the leader default (`4000`) instead of forced `0`
- chunked legal TTT now re-enters `train()` during adaptation and `eval()` during scoring
- same `TTT_PARAM_MODE=full`
- same `TTT_LR=0.002`
- same `TTT_EPOCHS=3`
- same `TTT_CHUNK_TOKENS=32768`

Expected success criterion:
- `post_ema` and `legal_ttt_exact` both move toward the March 23 leader

## Run 2

Name:
- `audit_seed42`

Purpose:
- exact corrected `chunk_full` recipe, different seed
- measure whether seed variance is the higher-EV frontier after the parity fixes

Launcher:

```bash
DATA_ROOT_MODE=tmp bash /workspace/parameter-golf/launch_streaming_ttt_candidate_runpod.sh audit_seed42
```

Diff versus Run 1:
- `SEED=42`

Everything else stays fixed.

Expected success criterion:
- final `legal_ttt_exact` beats the current best `chunk_full`

## Run 3

Name:
- `audit_seed2025`

Purpose:
- exact corrected `chunk_full` recipe, different seed
- complete the high-value seed sweep before spending on micro-tweaks

Launcher:

```bash
DATA_ROOT_MODE=tmp bash /workspace/parameter-golf/launch_streaming_ttt_candidate_runpod.sh audit_seed2025
```

Diff versus Run 2:
- `SEED=2025`

Everything else stays fixed.

Expected success criterion:
- final `legal_ttt_exact` beats the current best `chunk_full`

## Run 4

Name:
- `audit_lateqat_0145`

Purpose:
- first tiny train-quality nudge only if corrected parity plus the seed sweep still does not produce a better winner
- shifts late QAT slightly earlier without changing the export codec

Launcher:

```bash
DATA_ROOT_MODE=tmp bash /workspace/parameter-golf/launch_streaming_ttt_candidate_runpod.sh audit_lateqat_0145
```

Diff versus the corrected parity baseline:
- `LATE_QAT_THRESHOLD=0.145` instead of `0.15`

Everything else stays fixed.

Expected success criterion:
- `post_ema` improves first

## Stop Condition

If corrected parity plus the two seed runs still do not beat the current best `chunk_full`, and `audit_lateqat_0145` still does not improve `post_ema`, do not spend the next 8x cycle on:
- broader quantization pivots
- new TTT modes
- more block-window exploration

At that point, the next justified step is implementing matched export-aware QAT for the exact current int6 export path.
