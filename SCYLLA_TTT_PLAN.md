# Scylla TTT Plan

## Current Read

Scylla-v2 is no longer blocked on legality or byte accounting.

What is true now:
- the exact bundle is valid
- Scylla trains well
- Scylla quantizes well
- Scylla remains weaker than the best legal SP path specifically after the current score-first TTT pass

That points to an eval-time adaptation mismatch, not a dead tokenizer path.

## Main Thesis

The current TTT recipe is implicitly tuned to SP1024 token geometry.

Scylla has a different token-to-byte relationship, so fixed-token adaptation chunks do not represent the same amount of source text or the same amount of semantic regularity. Reusing SP1024's TTT settings therefore changes:
- bytes covered per update
- number of updates per byte
- overlap structure across updates
- optimizer pressure on the same model subset

The result is that Scylla comes into TTT strong, but the current adaptation regime fails to convert enough of that advantage into final BPB.

## What We Already Learned

From the first Scylla-specific smoke sweep:
- `TTT_CHUNK_TOKENS=34304`, `TTT_LR=0.0015` was best
- `TTT_CHUNK_TOKENS=49152`, `TTT_LR=0.0020` was second
- both beat the SP1024 smoke baseline

From the promoted `8xH100` run:
- the same Scylla setting did not beat the strong legal SP path
- Scylla still improved only modestly under TTT at scale

So:
- mild retuning helps
- but simple chunk/LR tuning alone is not enough yet

## High-Value Next Axes

### 1. Byte-Aware Chunking

Hand-designed sweeps.

Goal:
- compare TTT by comparable source-byte span, not just token count

Recommended cases:
- `TTT_CHUNK_TOKENS=34304`
- `TTT_CHUNK_TOKENS=49152`
- `TTT_CHUNK_TOKENS=65536`

Rationale:
- `34304` is the current best byte-matched setting
- `49152` tests fewer, richer updates
- `65536` tests whether Scylla benefits from significantly more context per adaptation step

### 2. Update Strength

Hand-designed sweeps.

Recommended cases:
- `TTT_LR=0.001`
- `TTT_LR=0.0015`
- `TTT_LR=0.002`

Rationale:
- the best current setting already moved lower than the old default
- Scylla may need gentler adaptation per chunk, not more aggression

### 3. Freeze Depth

Hand-designed sweeps first.

Current code exposes `TTT_FREEZE_BLOCKS`.

Recommended cases:
- `TTT_FREEZE_BLOCKS=2`
- `TTT_FREEZE_BLOCKS=4`
- `TTT_FREEZE_BLOCKS=6`

Rationale:
- byte-native tokenization may benefit from preserving lower-level representations and adapting mostly later layers
- this is the closest available proxy to "top-layer-only" adaptation in the current implementation

### 4. TTT Batch Size

Secondary hand sweep.

Current code exposes `TTT_BATCH_SEQS`.

Recommended cases:
- `TTT_BATCH_SEQS=16`
- `TTT_BATCH_SEQS=32`
- `TTT_BATCH_SEQS=64`

Rationale:
- this affects optimization noise and memory behavior inside each chunk
- lower batch size may improve local adaptation quality; higher may stabilize updates

### 5. TTT Epochs

Lower priority.

Recommended cases:
- `TTT_EPOCHS=2`
- `TTT_EPOCHS=3`
- `TTT_EPOCHS=4`

Rationale:
- useful only after chunk size / LR / freeze depth are better understood

## What Is Not Worth Autoresearch Yet

Do not hand broad free-form TTT design to an autoresearch loop yet.

Bad fit for autoresearch right now:
- picking among many continuous TTT hyperparameters without strong priors
- inventing brand-new adaptation algorithms
- mixing chunk size, LR, epochs, batch size, and freeze depth all at once

Why:
- the search space is still small enough for disciplined hand sweeps
- comparisons are expensive
- interactions are easy to misread without a clean ladder

## What *Is* Suitable For Autoresearch

After the next hand-sweep phase, a narrow autoresearch loop could be useful for:
- local refinement around a known-good chunk size / LR region
- choosing among a very small discrete set of freeze-depth / batch-size combinations
- ranking proposals using deterministic extracted metrics from completed logs

Good autoresearch objective:
- maximize `legal_ttt_exact`
- subject to no regression beyond a threshold in roundtrip BPB
- with intermediate progress measured from `ttt_chunk` trajectory

Good autoresearch mutation space:
- one discrete change at a time from a bounded menu:
  - chunk size from a whitelist
  - LR from a whitelist
  - freeze depth from a whitelist
  - batch size from a whitelist

Notably:
- this should be a *configuration-search* autoresearch loop
- not a code-editing free-for-all

## Immediate Path Forward

1. Run a small Scylla TTT ladder on `1xH100`.
   - chunk size
   - LR
   - freeze depth

2. Compare results using:
   - `final_int8_zlib_roundtrip_exact`
   - `legal_ttt_exact`
   - `ttt_chunk` progress over elapsed time

3. Promote only the best Scylla regime to `8xH100`.

4. If a stable region emerges, then build a narrow autoresearch loop over that region.

## Claude Sidecar

Claude has been asked to provide an independent eval-time adaptation diagnosis focused on:
- bytes vs tokens
- fewer richer updates vs more frequent small updates
- embeddings and output-path adaptation
- what belongs in hand sweeps vs autoresearch

That response should be treated as sidecar guidance, not ground truth.
