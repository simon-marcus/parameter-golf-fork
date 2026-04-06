# Legality Checklist

This file checks the current JEPA leader-stack candidate against:
- the official repository README rules in [README.md](/Users/simon/Code/parameter-golf/README.md)
- the unofficial guidance in Issue #1017, “A Field Guide to Valid Submissions”

The issue is not official policy, but it is useful as a structured review checklist. Where the issue and the README differ, the README governs.

## 1. Artifact Size

Official requirement:
- total code bytes plus compressed model bytes must be `<= 16,000,000`

Current status:
- original `jepa01` export on seed `1337` was **not valid** on size:
  - `16187860` bytes total
- storage-only export pass is **under the cap**:
  - `15826872` quant bytes
  - `92888` code bytes
  - `15919760` estimated total

Status:
- Pass for the storage-only export
- Fail for the original over-cap export

## 2. Single Self-Contained Script

Official rule/guidance:
- counted code should live in the record-local `train_gpt.py`
- the records folder should compile and run on its own

Current status:
- current research candidate is implemented as:
  - [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py)
  - which imports [records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py)
- the storage-only pass is currently a second script:
  - [leader_stack_storage_pass.py](/Users/simon/Code/parameter-golf/leader_stack_storage_pass.py)

Risk:
- this is acceptable for research, but it is not the cleanest record-package shape
- before final submission, we should flatten the wrapper and the storage-export logic into a single record-local `train_gpt.py`

Status:
- Research-valid
- Packaging task remains for final submission

## 3. Training Time

Official rule:
- training must fit within `10` minutes on `8xH100`

Current status:
- seed `1337` training stopped at:
  - `stopping_early: wallclock_cap train_time:599393ms`
- this is within the `600000ms` budget

Status:
- Pass

## 4. Evaluation Time

Official rule:
- evaluation must fit within an additional `10` minutes on the same hardware

Current status for seed `1337` storage-only under-cap export:
- int6 roundtrip eval: `7863ms`
- sliding-window exact eval: `76028ms`

This is well below the evaluation cap.

Status:
- Pass

## 5. No Network / No External Runtime Side Information

Official rule:
- the scored artifact must be self-contained
- no network calls, external downloads, or runtime side information during evaluation

Current status:
- run-time data staging from S3 was only used to prepare the pod before the run
- the scored model/eval path uses only local files
- no network calls are part of the training or evaluation scripts

Status:
- Pass

## 6. Validation Metric Correctness

Issue #1017 emphasizes:
- full normalized distribution
- score-before-update
- single left-to-right pass
- correct byte-level BPB calculation
- full validation split

Current status:
- tokenizer: SentencePiece `sp1024`
- byte accounting uses the official LUT path in the March 23 base stack:
  - `build_sentencepiece_luts()`
- validation uses the full `fineweb_val_*.bin` pattern, not a single shard
- final scored path is the fixed-model sliding-window eval with `TTT_ENABLED=0`

Status:
- Pass, subject to preserving the same code path in the flattened record-local script

## 7. JEPA And Causality

Potential concern:
- does the JEPA loss create any evaluation-time information leak?

Answer:
- no, because JEPA is a training-time auxiliary objective only
- during evaluation, the model is just a fixed predictor with ordinary forward passes
- no JEPA target is computed from validation tokens during scoring

Status:
- Pass

## 8. Storage-Only Export Pass

Potential concern:
- is the storage-only export pass an illegal transfer of compute from training into evaluation?

Our interpretation:
- the pass starts from the finished checkpoint only
- it uses no training or validation data
- it does not adapt the model to validation tokens
- it only changes serialization/quantization of the already-trained weights

Why this appears acceptable:
- the README explicitly warns that *data-consuming* state-modification procedures like GPTQ/Hessian calibration belong to training
- our storage-only pass is not data-calibrated
- it is closer to post-training compression than to evaluation-time learning

Remaining caution:
- for the cleanest submission, the storage-only export should be integrated into the final record-local `train_gpt.py` so the entire artifact-generation path is explicit in one script

Status:
- Likely pass
- Should still be clearly documented in the final README

## 9. Seed Brute Forcing

Official caution:
- brute-forcing seeds or optimizing directly on validation outcomes can be disqualifying

Current status:
- competition record proof uses the standard 3-seed pattern:
  - `1337`
  - `42`
  - `2026`
- these are conventional fixed seeds, not a large offline search

Status:
- Pass

## 10. JEPA Claim Scope

Important honesty point:
- this submission is JEPA-based, but not a pure byte-level no-tokenizer JEPA
- the JEPA contribution claim should therefore be:
  - JEPA helps inside this competition-facing leader stack
  - not that this is the “true byte-level JEPA” challenge solution

Status:
- Documentation requirement, not a rule violation

## Bottom Line

Current legal/compliance position:
- the metric path looks valid
- the under-cap storage-only export looks valid
- the main remaining submission task is packaging:
  - flatten the wrapper/base split
  - ship a single record-local `train_gpt.py`
  - include the three seed logs and final submission metadata
