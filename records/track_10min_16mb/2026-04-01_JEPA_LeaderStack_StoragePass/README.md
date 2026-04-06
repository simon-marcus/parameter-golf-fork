# JEPA Leader-Stack + Storage-Only Int6 Export

This record captures our current JEPA-based competition candidate and the research path that led to it.

Current confirmed result from seed `1337` on `8x H100 SXM`:
- under-cap artifact estimate: `15,919,760` bytes
- final exact metric: `val_bpb = 1.12271348`

This beats the March 17 Naive Baseline (`1.22436570`) by a wide margin while preserving a matched ablation story showing that JEPA itself improves the underlying training recipe.

Two additional seeds are currently being run with the same training recipe and the same storage-only export pass:
- `SEED=42`
- `SEED=2026`

## What This Submission Is

This is not a pure byte-level JEPA submission. It is a competition-facing `sp1024` leader-stack model with an added JEPA auxiliary loss, followed by a storage-only post-training export change that reduces the final artifact under `16,000,000` bytes.

That distinction matters:
- The **JEPA research claim** is that JEPA helps inside this model family.
- The **competition claim** is that the resulting submission is stronger than the baseline while staying within the artifact and time limits.
- The **Will DePue byte-level JEPA merch challenge** is a separate target; this submission does **not** satisfy the “no tokenizer, byte level” condition for that challenge.

## Final Candidate Snapshot

Training model:
- Base stack: March 23 leader-family stack from [records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py)
- Wrapper with JEPA: [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py)
- JEPA setting: `JEPA_LOSS_WEIGHT=0.10`
- Tokenizer/data: `fineweb10B_sp1024` + `fineweb_1024_bpe.model`
- Eval mode for this candidate: `TTT_ENABLED=0`

Storage-only export pass:
- Script: [leader_stack_storage_pass.py](/Users/simon/Code/parameter-golf/leader_stack_storage_pass.py)
- Changes relative to the original March 23 export:
  - remove duplicate top-level JEPA alias weights (`jepa_in.weight`, `jepa_out.weight`) from the exported state
  - quantize `attn`, `mlp`, `embed`, and `other` floating tensors with the int6 path
- This pass starts from the trained checkpoint only. It does not retrain, does not use validation tokens to change weights, and does not introduce any new data source.

## Seed 1337 Result

From the `8xH100` run and storage-only follow-up:

Original trained-export result:
- `final_int8_zlib_roundtrip_exact val_bpb: 1.12128254`
- `Total submission size int6+lzma: 16187860`
- This was too large to submit.

Storage-only under-cap result from the same checkpoint:
- `storage_pass quant_bytes=15826872`
- `storage_pass code_bytes=92888`
- `storage_pass estimated_total=15919760`
- `storage_pass_final_exact val_bpb: 1.12271348`

So the storage fix cost about `0.00143` BPB while making the artifact submission-valid.

Durable logs:
- training log: [jepa01_train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/jepa01_train.log)
- original run driver: [jepa01_8x_driver_manual.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/jepa01_8x_driver_manual.log)
- storage-only export/eval log: [storage_pass_allint6.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/storage_pass_allint6.log)

## Why We Believe JEPA Is Doing Real Work

We tried to isolate JEPA’s effect rather than bury it inside a giant stack change.

### 1. Byte-level isolation lane

We built a separate JEPA isolation trainer on top of the March 17 Naive Baseline:
- [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py)

That lane established a clean JEPA-over-control win on `8xH100`:
- control: `1.36692884`
- `JEPA_LOSS_WEIGHT=0.10`: `1.36182961`

After adding full-model EMA, JEPA still won:
- control: `1.36487466`
- `JEPA_LOSS_WEIGHT=0.10`: `1.36169919`

This was our proof lane: same script, same hardware, same time budget, JEPA on beats JEPA off.

### 2. Leader-stack translation lane

We then moved into the stronger March 23 stack with `TTT_ENABLED=0` first, so the JEPA question remained interpretable:
- [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py)

Short `1xH100` screens suggested several viable weights, but longer `600s` confirmatory runs showed:
- control: `2.22466503`
- `weight:0.05`: `2.22707166`
- `weight:0.15`: `2.29532844`
- `weight:0.10`: `2.19728869`

So in the competition-facing leader stack, `0.10` was the only JEPA weight that held up at longer horizon.

### 3. Final 8xH100 candidate

We promoted exactly that setting to `8xH100`, then fixed storage after the fact from the saved checkpoint.

## What We Tried And What Failed

This JEPA result did not come from a single clean idea. We tried several dead ends and kept explicit logs because we wanted a credible JEPA story, not just a lucky number.

Key failed or inconclusive directions:
- exploratory byte-level JEPA hybrid in [train_jepa.py](/Users/simon/Code/parameter-golf/train_jepa.py): early positive screens, but not competitive at longer horizon
- minimal-diff isolation lane before the stronger MLP transplant: JEPA did not help
- tiny learned predictor head alone on the weak isolation lane: did not produce a robust win
- leader-stack short-screen weight `0.15`: looked best at `180s`, then lost badly at `600s`
- simple stronger-LZMA recompression: did not solve the artifact overage

Key things that helped:
- `LeakyReLU(0.5)^2`
- full-model EMA for export/eval
- choosing `JEPA_LOSS_WEIGHT=0.10` by longer-horizon validation, not by the cheapest screens alone
- storage-only export change from the saved checkpoint

The research trail is summarized in:
- [PLAN_AND_PROGRESS.md](/Users/simon/Code/parameter-golf/PLAN_AND_PROGRESS.md)

## Legality Summary

The detailed legality audit is in:
- [LEGALITY_CHECKLIST.md](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-04-01_JEPA_LeaderStack_StoragePass/LEGALITY_CHECKLIST.md)

Short version:
- metric uses the official SentencePiece byte-length LUT path, not a hardcoded token-to-byte ratio
- validation uses the full validation shard set
- evaluation is single-pass sliding-window scoring with `TTT_ENABLED=0`
- the storage-only export pass uses only the trained checkpoint, not evaluation tokens
- the main remaining packaging task is to flatten the current wrapper/base split into a single record-local `train_gpt.py`

That last point is important: the current research code is a wrapper over the March 23 file. Before submission, we need a self-contained record-local `train_gpt.py` that embeds the JEPA and storage-export logic directly.

## Files Intended For The Final Record Folder

To be finalized after the two additional seeds complete:
- `train_gpt.py` — flattened, self-contained version of the current winner
- `README.md` — this document, updated with 3-seed aggregate results
- `submission.json`
- `requirements.txt`
- `train_seed1337.log`
- `train_seed42.log`
- `train_seed2026.log`

## Current Status

Seed status:
- `1337`: complete, under-cap `1.12271348`
- `42`: running
- `2026`: queued

If the remaining seeds land similarly, this is a real JEPA-based record-track submission candidate rather than just an interesting non-record experiment.
