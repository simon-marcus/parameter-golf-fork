# Codex Scylla 2 — Plan And Progress

## Current State

Scylla-v2 is now legally exact:
- tokenizer path is byte-exact
- full bundle audit passes
- `val_bpb` accounting is correct

What remains unresolved is competitive eval-time adaptation.

## What We Learned So Far

### 1xH100 smoke results

The strongest Scylla smoke setting so far is still the conservative TTT regime:
- `TTT_CHUNK_TOKENS=34304`
- `TTT_LR=0.0015`
- `TTT_EPOCHS=3`
- `TTT_FREEZE_BLOCKS=2`
- `TTT_FREEZE_EMBEDDINGS=0`

Negative findings:
- larger chunks hurt
- fewer epochs hurt
- deeper freezing hurt
- freezing embeddings hurt the most

### 8xH100 result

The promoted exact Scylla-v2 run on `8xH100` did not beat the best SP legal-TTT path.

That shifts the working thesis:
- Scylla is not blocked by legality or tokenizer correctness
- Scylla is not obviously blocked by basic 1x smoke TTT tuning
- the likely issue is scale/training-state mismatch at 8x

## Updated Thesis

The 1x->8x transfer gap may be less about chunk geometry than about how much adaptation a stronger model actually wants.

A much more trained 8x model likely enters TTT in a sharper basin:
- the same TTT learning rate may be too blunt
- cumulative updates may overshoot more easily
- quantized TTT may destroy some of the adaptation signal

## Legality Update

`LEGALITY.md` now records the current read that pre-quant, score-first, backward-looking TTT is legal.

This makes pre-quant TTT the highest-value next experiment.

## Next Experiment

### P0 — Pre-Quant TTT Control

Goal:
- keep the current best Scylla smoke TTT regime
- switch only the adaptation target from post-quant eval weights to pre-quant trained weights

Config:
- `TTT_CHUNK_TOKENS=34304`
- `TTT_LR=0.0015`
- `TTT_EPOCHS=3`
- `TTT_FREEZE_BLOCKS=2`
- `TTT_FREEZE_EMBEDDINGS=0`
- `TTT_USE_PREQUANT=1`

What this tests:
- whether quantized-TTT is blunting Scylla’s adaptation signal
- whether Scylla’s 8x underperformance is really a quantized-adaptation bottleneck

## Immediate Follow-Ups If P0 Helps

1. Pre-quant TTT with lower LR on 8x
   - `P1`: same as `P0`, but `TTT_LR=0.0010`
   - `P2`: same as `P0`, but `TTT_LR=0.0007`
2. Pre-quant TTT with slightly lower total adaptation pressure
3. Only then consider more structural reset/EMA ideas

## April 2 Results

### Exact Scylla 8x control vs pre-quant TTT

Baseline exact Scylla 8x control:
- post-EMA BPB: `1.1496`
- roundtrip BPB: `1.13431108`
- legal TTT BPB: `1.13150122`
- total submission size: `16,541,448`

`P0` pre-quant TTT:
- post-EMA BPB: `1.1494`
- roundtrip BPB: `1.13361567`
- legal TTT BPB: `1.12536335`
- total submission size: `16,499,087`

`P1` pre-quant TTT with lower LR:
- post-EMA BPB: `1.1487`
- roundtrip BPB: `1.13231074`
- legal TTT BPB: `1.12466737`
- total submission size: `16,550,151`

### What Helped

The gains are not primarily from a different training curve.

Training moved only slightly:
- `P1` is a bit better than the old exact Scylla control at post-EMA
- step time stayed essentially unchanged

The material gain is in evaluation:
- pre-quant TTT improved Scylla substantially over post-quant TTT
- lowering TTT LR from `0.0015` to `0.0010` improved again

Chunk-trajectory comparison confirms that the advantage appears early and holds:
- control around chunk `681`: about `1.131452`
- `P0` around chunk `681`: about `1.124673`
- `P1` around chunk `681`: about `1.124154`

So the most helpful work so far is:
1. score-first TTT on pre-quant weights
2. gentler TTT LR at 8x

### What Did Not Help

Earlier smoke sweeps already ruled out:
- larger chunk sizes
- fewer TTT epochs
- deeper freezing
- freezing embeddings

So the current Scylla direction is not "invent a very different TTT."
It is:
- keep the successful Scylla pre-quant path
- refine it cautiously

## Current Blockers

### 1. Still not yet beating the strongest SP path

`P1` is the best Scylla result so far, but it still trails the strongest SP legal path.

### 2. Still over the `16,000,000` byte cap

Current totals:
- control: `16,541,448`
- `P0`: `16,499,087`
- `P1`: `16,550,151`

So even the best BPB run is not submission-legal yet.

## Updated Priority

The next branch should be storage-only export work first, not retraining.

Why:
- the current Scylla eval path is finally producing real gains
- the model is only modestly over cap
- we already have a successful repo precedent for a checkpoint-only storage pass in the JEPA line

## Storage-Pass Thesis

The JEPA storage pass showed that a modestly over-cap run can be rescued without retraining by:
- starting from the saved checkpoint
- tightening export representation only
- widening the int6 quantization set selectively
- removing any duplicated exported aliases if they exist

Reusable lesson for Scylla:
- first try a checkpoint-only export pass
- measure size immediately
- only then evaluate BPB on under-cap candidates

## Next Concrete Work

1. Run a Scylla storage pass from the saved `P1` checkpoint before any retraining.
2. First candidate:
   - broaden int6 quantization from `mlp,attn` to `mlp,attn,embed,other`
   - keep the pre-quant legal-TTT eval path fixed
3. Second candidate:
   - same broader int6 mix
   - raise `LZMA_PRESET` from `6` to `9`
4. Only if still over cap:
   - try an export-time bigram ablation
   - measure size first, then judge the BPB loss
5. Only if these storage passes fail badly should we spend more `8x` budget on further training/eval variants.

## Storage-Pass Readiness

Prepared helper:
- `records/codex_scylla_2/scylla_storage_pass.py`

Prepared launcher:
- `records/codex_scylla_2/run_storage_pass.sh`

Current storage-pass variants:
- `S0`: all-int6 (`mlp,attn,embed,other`)
- `S1`: `S0` + `LZMA_PRESET=9`
- `S2`: `S1` + export-time bigram ablation

Current code inspection:
- no obvious JEPA-style duplicate alias pair to drop
- most promising size lever is widening Scylla's int6 export policy
- next strongest free lever is stronger LZMA compression

Operational note:
- we do not yet have the Scylla `final_model.pt` checkpoint locally
- when the paused pod is resumed, pull `final_model.pt` from the best run and execute the storage-pass ladder there or locally

## April 7 Comparison Against High-Vocab Scylla-V2

Recent `scylla_2_claude` work established something useful but narrower than we first hoped:
- larger Scylla vocab is real
- the best corrected high-vocab path we found was `12288 bridge S1`
- but it still did not beat the old `P1` Scylla line

Best relevant numbers:
- old `P1`: `legal_ttt_exact = 1.12466737`
- `12288 bridge S1`: `storage_pass_legal_ttt_exact = 1.14725027`
- `8192 bridge S1`: `storage_pass_legal_ttt_exact = 1.15145267`

What this demonstrates:
- `12288` is the best new-stack vocab point we validated under the corrected pre-quant, full-val, storage-pass path
- but the old `P1` lane still leads by about `0.02258` BPB
- the remaining gap is mostly upstream of TTT rather than a missing eval-time trick

The most important lesson is that the new high-vocab stack drifted too far from the proven Scylla submission path:
- larger vocab helped
- but the rest of the stack never became as compression-compatible as `P1`

## April 7 Low-Risk Revival Branch

The lower-risk branch now is to revive the old `P1` lane and port frontier ideas one at a time instead of replacing the whole stack at once.

Prepared revival configs in `launch_sweep.sh`:
- `R0`: `P1`-style pre-quant TTT plus `XSA_LAST_N=11`
- `R1`: same as `R0`, but with `QK_GAIN_INIT=5.0`

These are intentionally conservative:
- keep the old winning Scylla legal path
- add the cheapest frontier attention changes first
- only consider moderate vocab increases after we know whether `R0`/`R1` help

Current recommended next run order:
1. `R0`
2. `R1`
3. only if promising, port a moderate vocab increase into this old legal path
