# JEPA Target Embedding Autoresearch

## Purpose
This lane is for one narrow question:

- can we improve the **JEPA target embedding mechanism** in the byte-level isolation model enough to materially improve BPB, while keeping a clean `JEPA on` versus `JEPA off` comparison?

This is intentionally much narrower than "improve byte-level JEPA in general".

## Why This Lane Exists
The current byte-level isolation result is scientifically good but not competitive:
- matched `8xH100` EMA control: `1.36487466`
- matched `8xH100` EMA JEPA `0.10`: `1.36169919`

So JEPA clearly helps, but not enough to beat the `1.22436570` baseline.

One likely bottleneck is the current JEPA target:
- in [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py), the JEPA target for the next patch is currently built from the EMA token embedding table and then **mean pooled** across bytes in the patch
- this is clean and cheap, but it throws away byte order and local structure

That makes this a strong candidate for a constrained autoresearch lane:
- the change surface is narrow
- the scientific question is crisp
- we can still preserve the matched no-JEPA control

## Scope
Allowed mutation surface in the inner-loop probe:
- JEPA target patch aggregation
- local position handling inside target patches
- tiny target patch encoders
- light changes to the paired predictor head, if directly motivated by the target representation

Not in scope:
- changing the overall backbone family
- tokenizer changes
- patch-first full architecture rewrites
- eval-time tricks
- storage/export tricks
- unrelated optimizer rewrites

## Inner Loop
The actual inner loop is now [target_embedding_probe.py](/Users/simon/Code/parameter-golf/records/byte-level-jepa/target-embedding-autoresearch/target_embedding_probe.py), not full LM training.

It:
- consumes byte260 patch pairs directly
- trains only a tiny patch-level JEPA toy model
- evaluates held-out predictor/target quality
- emits a lower-is-better proxy score through the existing autoresearch parser

The probe score combines:
- normalized held-out prediction MSE
- retrieval accuracy penalty
- collapse penalty from low predictor variance

Additional diagnostics:
- held-out cosine similarity
- retrieval@1
- predictor and target std
- sensitivity to shuffling bytes inside a patch

## Success Criteria
The first goal is not "beat 1.2244 immediately".

The first goal is:
- discover stronger target-embedding mechanisms cheaply
- filter to a few promising target profiles
- then promote only those into real `1xH100` control-vs-JEPA runs

Promotion logic:
1. inner-loop probe autoresearch
2. short real `1xH100` control-vs-JEPA screen for promoted target variants
3. longer `1xH100` confirmatory run
4. only then `8xH100`

## Why Autoresearch Fits
This lane is well suited to a Karpathy-style autoresearch loop because:
- the search space is narrow enough to keep proposals disciplined
- failed ideas are cheap to discard
- the baseline is already stable and known-good
- we can constrain the prompt so the model does not wander into unrelated changes

## Practical Notes
Local setup on a strong Mac is now genuinely useful because the probe can run on:
- CPU
- MPS
- CUDA

There is also a synthetic-data mode for smoke testing the loop when local byte260 shards are unavailable.

The expectation is:
- iterate probe design locally
- use RunPod only when we want faster or larger probe throughput
- promote only a few target variants into actual LM training
