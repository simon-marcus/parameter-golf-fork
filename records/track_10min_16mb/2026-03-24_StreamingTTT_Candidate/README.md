# Streaming Legal TTT Candidate

This directory starts from the March 23, 2026 leader submission and adds a stronger legal TTT path aligned with the OpenAI clarification that type-2 token-stream TTT is valid.

## What Changed

- added `TTT_MODE=stream` to evaluate with online suffix-only adaptation
- kept `TTT_MODE=chunk` for direct baseline comparison against the March 23 score-first chunk protocol
- added `TTT_PARAM_MODE`:
  - `full`
  - `freeze_first_n`
  - `late_blocks`
  - `block_range`
  - `control_plus`
- added bank-gradient masking so block-selection actually affects banked weights
- added explicit legality/audit logs:
  - mode
  - parameter subset
  - scored token count
  - adapted token count

## Recommended Next Full Run

Use:

- `TTT_MODE=chunk`
- `TTT_PARAM_MODE=full`
- `TTT_LR=0.002`
- `TTT_EPOCHS=3`
- `TTT_LR_SCHEDULE=chunk_cosine`
- `TTT_MOMENTUM=0.9`
- `TTT_GRAD_CLIP=1.0`

Rationale:

- full 8x confirmations favored `chunk_full` over `chunk_midlate` and `chunk_2_11`
- the reduced 1x circuit-scan ranking did not transfer cleanly enough to justify mid-late block freezing
- the remaining gap to the March 23 leader appears before export, so `chunk_full` is the right anchor while we investigate small train-quality deltas

## Notes

- this candidate intentionally leaves architecture and export unchanged from the March 23 leader starting point
- the current best follow-up hypothesis is "preserve chunk_full and recover a small pre-export quality gap"
- see [PARITY_BLAME_AUDIT.md](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/PARITY_BLAME_AUDIT.md) and [NEXT_RUN_PACKAGE.md](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/NEXT_RUN_PACKAGE.md)

## 1-GPU Activation Suite

A reduced benchmark harness is available at `launch_streaming_ttt_activation_suite_1gpu.sh`.

It compares 8 cases on one GPU across the default seeds `1337,42,2025`:

- exact March 23 leader code + `leaky_relu_sq`
- exact March 23 leader code + `asymmetric_square`
- exact March 23 leader code + `gated_square`
- exact March 23 leader code + `sign_preserving_square`
- Codex streaming TTT (`stream_late4` logic) + `leaky_relu_sq`
- Codex streaming TTT (`stream_late4` logic) + `asymmetric_square`
- Codex streaming TTT (`stream_late4` logic) + `gated_square`
- Codex streaming TTT (`stream_late4` logic) + `sign_preserving_square`

Cases 1-4 run against the fetched `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` codepath. Cases 5-8 run against the streaming TTT candidate codepath.

## Circuit Scan

A reduced contiguous-window scan is available at `launch_streaming_ttt_circuit_scan_1gpu.sh`.

This is the recommended next-step harness for the "coherent circuit" hypothesis:

- keep the March 23 architecture and `leaky_relu_sq`
- vary only which contiguous block window is adaptable during legal TTT
- compare chunked TTT against streaming TTT on the same reduced 1-GPU proxy

Default cases:

- `chunk_full`
- `chunk_2_11`
- `chunk_4_11`
- `chunk_3_8`
- `chunk_4_9`
- `chunk_5_10`
- `chunk_6_11`
- `stream_3_8`
- `stream_4_9`
- `stream_5_10`
- `stream_6_11`

Run:

- `DATA_ROOT_MODE=tmp CUDA_VISIBLE_DEVICES=0 bash /workspace/parameter-golf/launch_streaming_ttt_circuit_scan_1gpu.sh all`

Useful reduced-proxy defaults are already baked in:

- `USE_COMPILE=0`
- `VAL_TOKENS_LIMIT=1048576`
- `MAX_WALLCLOCK_SECONDS=120`
- `ITERATIONS=1200`

Interpretation:

- if `chunk_full` still wins, the March 23 all-block chunked TTT remains the right anchor
- if one of the contiguous chunk windows wins, that supports the "TTT acts on a circuit" hypothesis
- if one of the contiguous streaming windows wins, the streaming idea may still be viable, but the last-4 choice was too narrow or too late
