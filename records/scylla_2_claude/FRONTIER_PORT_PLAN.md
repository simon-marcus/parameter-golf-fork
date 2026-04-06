# Scylla Frontier Port Plan

## Goal

Move Scylla-v2 onto a closer-to-frontier stack before spending more budget on bespoke TTT tuning.

The requested sequence is:

1. Port frontier base-stack ideas to Scylla at the current `VOCAB_SIZE=1254`.
2. Ablate `bigram` and `ve` on top of that modernized stack.
3. Run a Scylla vocab ladder `1254 -> 1536 -> 2048 -> 3072`, while keeping TTT chunking in source-byte units and keeping the export budget under control with compression and clip-width tuning.

## What This Port Includes

The runnable port path uses `records/scylla_2_claude/train_gpt_legal_ttt.py` with new support for:

- byte-aware TTT chunking via `TTT_CHUNK_BYTES`
- frontier-style regularization defaults:
  - `QK_GAIN_INIT=5.0`
  - `XSA_LAST_N=11`
  - `MLP_MULT=4`
  - `MUON_WD=0.085`
  - `EMBED_WD=0.085`
  - `ADAM_WD=0.02`
- optional smear removal via `SMEAR_ENABLED=0`
- optional depth recurrence via:
  - `NUM_LOOPS`
  - `LOOP_START`
  - `LOOP_END`
  - `ENABLE_LOOPING_AT`
- more frontier-like export controls:
  - `COMPRESSOR=brotli`
  - `BYTE_SHUFFLE=1`
  - separate matrix and embedding bitwidth / clip controls
  - standard-deviation clip mode for export

## What This Port Does Not Yet Include

This path still does **not** implement the full April 5, 2026 / April 6, 2026 stack:

- no full-Hessian GPTQ
- no row-normalized MuonEq-R
- no dedicated `ShuffledSequenceLoader` replacement

Those are still worth considering, but the current patch is the highest-value compatible port that fits cleanly into the existing Scylla record workflow.

## Runner

Use:

```bash
bash ./launch_scylla_frontier_plan.sh <case>
```

The runner writes outputs under:

```bash
records/scylla_2_claude/frontier_runs/
```

For non-`1254` ladder runs, the runner expects corresponding Scylla bundle assets to exist. It resolves them either from:

- per-vocab explicit env vars:
  - `SCYLLA_V1536_DATA_PATH`
  - `SCYLLA_V1536_TOKENIZER_PATH`
  - `SCYLLA_V1536_TOKENIZER_META_PATH`
  - same pattern for `2048` and `3072`
- or from template names:
  - `SCYLLA_DATASET_TEMPLATE`
  - `SCYLLA_TOKENIZER_TEMPLATE`

Default templates are:

```bash
fineweb10B_scylla_v2_v{vocab}
scylla_v2_v{vocab}
```

## Phase 1: Frontier Port At 1254

Run:

```bash
bash ./launch_scylla_frontier_plan.sh phase1_all
```

Cases:

- `control_1254_legacy`
  - current 1254-style control
  - token-based TTT chunks: `TTT_CHUNK_TOKENS=34304`
  - old export path: `lzma`, no byte shuffle, quantile clip
- `frontier_1254`
  - frontier-style port without depth recurrence
  - byte-based TTT chunks: `TTT_CHUNK_BYTES=131072`
  - `QK_GAIN_INIT=5.0`, `XSA_LAST_N=11`, `MLP_MULT=4`, `SMEAR_ENABLED=0`
  - `brotli` + byte shuffle + SD clip export
- `frontier_1254_loop`
  - same as `frontier_1254`
  - adds recurrence on layers `4-5`, enabled at `50%` training

Promotion rule:

- Compare `legal_ttt_exact` first.
- Use artifact size as the second gate.
- If `frontier_1254_loop` is not clearly better than `frontier_1254`, keep the non-recurrent version as the base for phase 2.

## Phase 2: Bigram / VE Ablations

Run:

```bash
bash ./launch_scylla_frontier_plan.sh phase2_all
```

Cases:

- `ablate_1254_no_bigram`
- `ablate_1254_no_ve`
- `ablate_1254_no_bigram_no_ve`

These cases assume the frontier-ported stack with recurrence enabled, because the public frontier lineage removed auxiliary token-local structure as vocab increased.

Decision rule:

- if `no_bigram_no_ve` improves or stays neutral on `legal_ttt_exact` and reduces bytes, carry that stack into phase 3
- if only one ablation helps, carry the single winning ablation forward

## Phase 3: Vocab Ladder

Run:

```bash
bash ./launch_scylla_frontier_plan.sh phase3_all
```

Cases:

- `ladder_1254`
- `ladder_1536`
- `ladder_2048`
- `ladder_3072`

Shared assumptions:

- same modernized base stack
- `bigram=off`
- `ve=off`
- byte-aware TTT with `TTT_CHUNK_BYTES=131072`

### Byte-Budget Control Policy

Hold the architecture fixed first. When a larger vocab pushes the export up:

1. widen `EMBED_CLIP_SIGMAS`
2. then consider reducing `EMBED_BITS`
3. only after that change model capacity

The point of the ladder is to test whether Scylla benefits from larger vocab the same way the SP4096/SP8192 frontier did, not to confound vocab with model-size changes.

## Suggested Run Order

On `1xH100` smoke:

```bash
MAX_WALLCLOCK_SECONDS=180 NPROC_PER_NODE=1 bash ./launch_scylla_frontier_plan.sh phase1_all
MAX_WALLCLOCK_SECONDS=180 NPROC_PER_NODE=1 bash ./launch_scylla_frontier_plan.sh phase2_all
MAX_WALLCLOCK_SECONDS=180 NPROC_PER_NODE=1 bash ./launch_scylla_frontier_plan.sh phase3_all
```

On `8xH100` promotion:

- promote the best phase-1 stack first
- then the best phase-2 ablation
- only then spend `8x` budget on the vocab ladder winner candidates

## Hypothesis Summary

- If Scylla has been underperforming mainly because it was attached to an older SP1024-era stack, phase 1 should improve both roundtrip and legal-TTT BPB without needing a new tokenizer.
- If the larger frontier stack makes local token aids redundant, phase 2 should reduce bytes and may improve BPB by removing interference.
- If Scylla really shares the same “more bytes per token, more context per sequence” advantage seen in the SentencePiece frontier, phase 3 should show a non-trivial gain as vocab rises, provided the byte budget is controlled at export time rather than by shrinking the model immediately.
