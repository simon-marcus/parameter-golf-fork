# Byte-Level JEPA Challenge

## Motivation
Will DePue wrote:

> i’ll send merch to anyone that can get a JEPA model to beat the parameter golf baseline! only rule is no tokenizer (use byte level) to be true to JEPA

This folder is for that challenge specifically.

The bar to clear is the March 17 Naive Baseline:
- `final_int8_zlib_roundtrip_exact val_bpb: 1.22436570`
- tokenizer-based baseline, not byte-level

The byte-level challenge here is stricter than our main JEPA non-record submission:
- no tokenizer
- use byte-level inputs only
- show that JEPA is a real contributor, not just an incidental add-on

## What We Already Know

### Strong result that does NOT satisfy this challenge
Our best JEPA submission so far is SP-1024 based, so it does not count for Will's byte-level challenge even though it is strong.

### Byte-level JEPA evidence we do have
The cleanest byte-level JEPA evidence came from the isolation lane in [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py).

Matched `8xH100` results:
- control: `1.36692884`
- JEPA `0.10`: `1.36182961`
- EMA control: `1.36487466`
- EMA JEPA `0.10`: `1.36169919`

So JEPA does help in the byte-level lane. The problem is magnitude: this is still far worse than `1.2244`.

## Recent JEPA Attempts Summary

### 1. Byte-level isolation lane
This is the cleanest scientific lane, based on [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py).

What worked:
- JEPA beat a matched byte-level control on `8xH100`.
- After adding `LeakyReLU(0.5)^2` and then model EMA, JEPA still beat the matched control.
- Best proven full-scale byte-level JEPA result:
  - EMA control: `1.36487466`
  - EMA JEPA `0.10`: `1.36169919`

What did not work:
- This is still far above the `1.22436570` target.
- Larger `12x512` and `12x576` versions collapsed badly on cheap `1xH100` screens, so the current byte-level lane is throughput-bound rather than parameter-starved.

### 2. Leader-stack JEPA lane
This is the stronger `sp1024` competition-facing lane, not eligible for the byte-level merch challenge.

What worked:
- Longer `1xH100` confirmation showed `JEPA_LOSS_WEIGHT=0.10` beating the matched leader-stack control.
- That promoted cleanly to `8xH100`.
- After a storage-only export pass, the resulting JEPA submission became submission-valid under `16MB` and landed around `1.123` BPB over 3 seeds.

Why it does not count here:
- it uses `sp1024`, so it is not byte-level
- it proves JEPA can help in a strong stack, but not under Will's "no tokenizer" rule

### 3. Patch-first byte JEPA lane
This is the newer attempt in [train_jepa_patched.py](/Users/simon/Code/parameter-golf/train_jepa_patched.py), meant to reduce effective sequence length by operating on internal byte patches.

Patch-first attempts so far:
- first patched family: too weak overall, JEPA hurt
- second patched family: JEPA helped internally, but total BPB was still much worse than the older byte-level lane

Latest patched matrix on `1xH100`, `180s`, `PATCH_SIZE=2`:
- `mixer control`: `2.53405611`
- `mixer jepa01`: `2.56703380`
- `attn control`: `2.57629430`
- `attn jepa01`: `2.59779192`

Interpretation:
- both patched variants are much worse than the older byte-level isolation lane
- in both `mixer` and `attn`, JEPA made the patched model worse rather than better
- patch-first remains plausible as a long-term direction, but the current implementation is not competitive

### 4. XSA-all transplant into the byte-level isolation lane
We then tested the cleanest transplant from PR #1019 first: XSA on all layers, while keeping the byte-level isolation lane and running JEPA versus control in parallel on separate `1xH100` pods.

Matched subset-screen results on `1xH100`, `180s`, using the byte260 S3 screening bundle (`3` train shards, `2` val shards):
- `xsa_control`: `2.66036151`
- `xsa_jepa01`: `2.60044021`

Interpretation:
- absolute quality was poor, so this subset screen does not justify promotion by itself
- but JEPA still beat the matched no-JEPA control inside the XSA-all lane
- this keeps the JEPA attribution story intact even though the transplant did not make the byte-level lane competitive

## Current Gap
To beat `1.2244`, the byte-level JEPA lane must recover roughly `0.137` BPB from our best current byte-level JEPA result.

That is too large to expect from another scalar JEPA sweep or a cosmetic architectural tweak. We need a more ambitious byte-level plan.

One important refinement from follow-up analysis: before we commit to a full patch-first rewrite, we should answer a cheaper structural question first:
- is the current byte-level JEPA lane parameter-starved, because it is only using about `9MB` of the `16MB` budget?
- or is it already throughput-bound, so adding capacity will just reduce useful training steps?

That question is cheap to test and changes what should happen next.

We ran that test on `1xH100`, `180s`, with matched byte-level JEPA screens:
- `control` (`9x512`): `1.87138830`
- `jepa01` (`9x512`): `1.84092584`
- `depth12_control` (`12x512`): `2.33624034`
- `depth12_jepa01` (`12x512`): `2.29014684`
- `depth12wide_control` (`12x576`): `2.77315789`
- `depth12wide_jepa01` (`12x576`): `2.74212553`

Conclusion:
- the larger byte-level variants were dramatically worse
- this lane is throughput-bound, not parameter-starved
- the current 1024-byte backbone is the wrong place to spend more parameters
- patch-first byte modeling is now the clear next step
- XSA-all did not rescue the byte-level lane on the subset screen, although JEPA still improved the matched XSA control

## Constraints
Allowed:
- byte-level inputs
- JEPA or JEPA-hybrid training objectives
- architecture changes
- recurrence / universal-transformer style reuse
- longer context or patching
- legal eval-time methods that do not use external data
- aggressive compression-aware training
- parameter sharing / banking / better optimizers

Not allowed for this specific challenge:
- tokenizer tricks
- subword vocabularies
- claiming success from our SP-1024 JEPA result

## Working Hypothesis
The current byte-level lane is leaving too much performance on the floor because byte-level modeling needs stronger structure than a direct transplant of the March 17 baseline. The most promising path is probably not "more JEPA weight" but "better byte-level representation + stronger reuse of context + JEPA as a regularizing/predictive pressure on top".

## Candidate Directions
These are the current candidate directions to investigate next:
- scale the current byte-level JEPA backbone up to use much more of the `16MB` budget
- byte patching / latent bytes: operate on groups of bytes while still scoring exact bytes
- stronger parameter sharing / recurrence so byte-level context goes further per parameter
- legal eval-time adaptation at byte level, only after the training recipe is credible
- compression-aware export from the start, not only at the end
- leader-stack ideas transplanted one-by-one into the byte-level JEPA lane

## Initial Attack Plan

The likely winning path is not a pure decoder-free JEPA. Parameter Golf is scored by BPB, so the most realistic contender is still a JEPA-hybrid that emits exact byte probabilities while using JEPA to shape the representation.

### Phase 1: move to patch-first byte modeling
The capacity test already answered the first structural question:
- simply making the current byte-level backbone larger is a bad trade
- throughput collapses before the extra parameters become useful

So the next architectural move is no longer tentative. It is the patch-first rewrite.

### Phase 2: make byte patches first-class
The current byte-level lane is probably too weak because bytes are semantically thin and sequences are too long.

Most plausible fix:
- keep raw bytes as the only input alphabet
- group them into fixed internal byte patches
- predict exact bytes with a small local decoder head
- put the JEPA objective on future patch latents rather than only per-byte hidden states

This remains tokenizer-free while giving the model a more meaningful unit of structure.

### Phase 3: add more effective depth per parameter
Byte-level modeling probably needs more effective depth per parameter than the March 17 family provides.

The most attractive options are:
- universal-transformer style recurrence
- shared-depth stacks over byte patches
- lightweight state-space or recurrent memory over patch summaries

These are promising because they can improve long-range byte modeling without exploding artifact size.

### Phase 4: only then consider byte-level eval-time methods
If the training recipe gets close:
- test stronger sliding-window byte eval
- then consider legal score-first adaptation / TTT methods

This should be a late-stage multiplier, not the first rescue attempt.

## Files To Watch
- [train_jepa.py](/Users/simon/Code/parameter-golf/train_jepa.py)
- [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py)
- [train_jepa_patched.py](/Users/simon/Code/parameter-golf/train_jepa_patched.py)
- [records/byte-level-jepa/target-embedding-autoresearch/JEPA_TARGET_EMBEDDING_AUTORESEARCH.md](/Users/simon/Code/parameter-golf/records/byte-level-jepa/target-embedding-autoresearch/JEPA_TARGET_EMBEDDING_AUTORESEARCH.md)
- [records/byte-level-jepa/target-embedding-autoresearch/program.md](/Users/simon/Code/parameter-golf/records/byte-level-jepa/target-embedding-autoresearch/program.md)
- [run_jepa_target_embedding_autoresearch.sh](/Users/simon/Code/parameter-golf/run_jepa_target_embedding_autoresearch.sh)
- [PLAN_AND_PROGRESS.md](/Users/simon/Code/parameter-golf/PLAN_AND_PROGRESS.md)

## Goal
Produce a true byte-level JEPA submission that can honestly say:
- no tokenizer
- JEPA helps against a matched byte-level control
- final score beats the `1.2244` Naive Baseline

## New Narrow Research Lane
We now have a dedicated target-embedding autoresearch lane focused on one specific question:
- can we improve JEPA by making the target patch embedding richer, while keeping the rest of the byte-level isolation lane fixed?

Entry points:
- [records/byte-level-jepa/target-embedding-autoresearch/JEPA_TARGET_EMBEDDING_AUTORESEARCH.md](/Users/simon/Code/parameter-golf/records/byte-level-jepa/target-embedding-autoresearch/JEPA_TARGET_EMBEDDING_AUTORESEARCH.md)
- [run_jepa_target_embedding_autoresearch.sh](/Users/simon/Code/parameter-golf/run_jepa_target_embedding_autoresearch.sh)
