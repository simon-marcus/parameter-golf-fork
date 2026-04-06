# Leader-Stack JEPA With Storage-Only Fix

Status: in preparation

This folder is the packaging workspace for our JEPA submission candidate.

## What This Submission Is

This line of work started as a deliberate JEPA investigation rather than a generic leaderboard optimization exercise.

We wanted to answer two separate questions:

1. Does a JEPA-style auxiliary objective actually help `val_bpb` in this challenge setting?
2. If it does, can it survive inside a competitive stack rather than only in a toy ablation lane?

The resulting candidate is a **leader-stack JEPA hybrid**:
- base stack: March 23 leader-family training recipe
- JEPA mechanism: learned patch-latent predictor on top of the leader hidden states
- evaluation setting: `TTT_ENABLED=0` for the JEPA attribution work
- final storage fix: post-training export change only, no retraining

## Why We Believe JEPA Is Actually Helping

We intentionally kept a cleaner evidence lane in parallel with the competitive lane.

### 1. Isolation lane

In the isolation lane we used [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py), which was built to test JEPA with minimal confounds.

After adding `LeakyReLU(0.5)^2` and EMA, matched `8xH100` runs showed:
- control: `1.36487466`
- JEPA `0.10`: `1.36169919`

This established that JEPA could improve a matched control at full scale.

### 2. Leader-stack translation lane

We then moved to the competitive lane with [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py).

Short `1xH100` screens suggested multiple possible weights, but longer confirmatory runs were more informative:
- `600s` control: `2.22466503`
- `600s` JEPA `0.05`: `2.22707166`
- `600s` JEPA `0.10`: `2.19728869`
- `600s` JEPA `0.15`: `2.29532844`

So in the stronger leader stack, `JEPA_LOSS_WEIGHT=0.10` was the robust winner.

### 3. Full `8xH100` leader-stack result

Using the exact leader-stack JEPA `0.10` candidate on `8xH100`, we obtained:
- over-cap export: `final_int8_zlib_roundtrip_exact val_bpb: 1.12128254`

That run beat the naive baseline comfortably, but missed the artifact limit.

### 4. Storage-only fix

We then performed a storage-only pass from the finished checkpoint:
- no retraining
- same trained weights
- export change only

The winning bounded storage recipe was:
- remove duplicated top-level JEPA alias weights from export
- quantize `attn`, `embed`, `mlp`, and `other` to `int6`

That produced:
- estimated total artifact size: `15,919,760` bytes
- final exact: `1.12271348`

So the storage fix costs about `0.00143` BPB relative to the over-cap export while making the artifact submission-valid on size.

## What We Tried And What Failed

This was not a single lucky run. We kept a running experiment log and deliberately killed branches that did not support a clean JEPA story.

### Early byte-level JEPA hybrid

The first exploratory file was [train_jepa.py](/Users/simon/Code/parameter-golf/train_jepa.py).

What it showed:
- JEPA was alive
- JEPA could help short screens
- but the longer `600s` control beat the JEPA hybrid

That line was useful for learning, but it was not yet a convincing competition candidate.

### Minimal-diff isolation without stronger base improvements

The first clean NaiveBaseline-derived JEPA isolation lane did **not** immediately show a JEPA win.

What changed that:
- a learned predictor head alone was not enough
- `LeakyReLU(0.5)^2` materially strengthened the lane
- once that stronger base was in place, JEPA beat the matched control

This is important because it shows we did not simply credit JEPA for gains that only came from unrelated tuning.

### Leader-stack short-screen false positives

On the leader stack, short `180s` screens initially made `0.15` look attractive.

Longer `600s` `1xH100` confirmation showed that was misleading:
- `0.15` lost badly
- `0.05` also lost slightly
- `0.10` was the actual robust leader

That is why our final candidate uses `JEPA_LOSS_WEIGHT=0.10`.

## Durable Evidence

Primary planning and chronology:
- [PLAN_AND_PROGRESS.md](/Users/simon/Code/parameter-golf/PLAN_AND_PROGRESS.md)

Key implementation files:
- [train_jepa.py](/Users/simon/Code/parameter-golf/train_jepa.py)
- [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py)
- [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py)
- [leader_stack_storage_pass.py](/Users/simon/Code/parameter-golf/leader_stack_storage_pass.py)

Key `8xH100` artifacts:
- [jepa01_train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/jepa01_train.log)
- [storage_pass_allint6.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/storage_pass_allint6.log)
- [final_model.pt](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/final_model.pt)
- [final_model.int6.ptz](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/final_model.int6.ptz)

## Important Scope Note

This candidate is a strong JEPA contribution and a plausible competition submission path.

It is **not** the byte-level JEPA variant requested in Will DePue's merch note. That note explicitly asked for byte-level/no-tokenizer JEPA. Our strongest valid competition candidate so far is on the `sp1024` leader stack, not the byte-level lane.

So we should present this honestly as:
- a JEPA-based competition candidate
- plus a separate byte-level JEPA research line that established signs of life and clean JEPA attribution, but did not reach the same absolute score

## Current Packaging Status

The modeling and storage results are strong, but packaging still needs one more pass before a real PR:
- scored code currently spans [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py) plus the imported March 23 base script
- the official artifact rule expects all scored code to live in a single `train_gpt.py`
- the storage-only export fix must be folded into that single-file submission path
- the challenge expects logs from 3 independent runs; seed `1337` is complete and seeds `42` and `2026` are currently in flight

That packaging audit is tracked in [LEGALITY_CHECK.md](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-04-01_LeaderStack_JEPA_StoragePass/LEGALITY_CHECK.md).
