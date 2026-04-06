# Legality Check: Leader-Stack JEPA Candidate

This document audits the current JEPA candidate against the official competition rules in [README.md](/Users/simon/Code/parameter-golf/README.md) and the unofficial guidance in issue [#1017](https://github.com/openai/parameter-golf/issues/1017).

## Scope

Candidate under audit:
- training path: [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py)
- storage-only export/eval path: [leader_stack_storage_pass.py](/Users/simon/Code/parameter-golf/leader_stack_storage_pass.py)
- launcher used for the main 8xH100 run: [launch_leader_stack_jepa_screen_runpod.sh](/Users/simon/Code/parameter-golf/launch_leader_stack_jepa_screen_runpod.sh)

Best completed `8xH100` seed so far:
- seed `1337`
- over-cap training/export result: `1.12128254` at `16,187,860` bytes
- under-cap storage-only result from the same checkpoint: `1.12271348` at estimated `15,919,760` bytes

## Official Rule Checks

### 1. Artifact must fit under `16,000,000` bytes
Rule source:
- [README.md:144](/Users/simon/Code/parameter-golf/README.md#L144)

Status:
- `PASS` for the storage-only export candidate

Evidence:
- [storage_pass_allint6.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/storage_pass_allint6.log)
  - `storage_pass quant_bytes=15826872`
  - `storage_pass code_bytes=92888`
  - `storage_pass estimated_total=15919760`

Notes:
- The original `jepa01` export was over cap.
- The under-cap candidate comes from:
  - dropping duplicated exported JEPA alias weights
  - quantizing `attn, embed, mlp, other` to `int6`

### 2. All counted code should live in a single `train_gpt.py`
Rule source:
- [README.md:146](/Users/simon/Code/parameter-golf/README.md#L146)
- issue [#1017](https://github.com/openai/parameter-golf/issues/1017), “Submission Structure & Constraints”, item 2

Status:
- `FAIL / MUST FIX BEFORE PR`

Why:
- The current candidate is split across:
  - [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py)
  - [records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py)
  - [leader_stack_storage_pass.py](/Users/simon/Code/parameter-golf/leader_stack_storage_pass.py)

Risk:
- The field guide explicitly frames “all scored code in a single `train_gpt.py`” as a hard validity condition.
- The current wrapper/import structure is excellent for research, but not yet a clean competition artifact.

Required fix:
- flatten the wrapper logic and the accepted storage export path into one submission-local `train_gpt.py`
- make that one file runnable inside the final record folder

### 3. Artifact must be self-contained, with no network calls or external downloads during evaluation
Rule source:
- [README.md:148](/Users/simon/Code/parameter-golf/README.md#L148)
- issue [#1017](https://github.com/openai/parameter-golf/issues/1017), “Submission Structure & Constraints”, item 3

Status:
- `PASS IN SPIRIT`, `PENDING FINAL FLATTENED SUBMISSION`

Why:
- Our experimental staging used S3 and RunPod prep scripts, but the candidate runtime itself does not need network access once the record folder is assembled.
- The actual model run on the pod used only local code, local dataset shards, and local tokenizer files.

Required final-state condition:
- the record folder must contain all code needed for training/export/eval
- no submission script may rely on S3, staging pods, or external helper scripts at runtime

### 4. Training must run in under 10 minutes on `8xH100` SXM
Rule source:
- [README.md:94](/Users/simon/Code/parameter-golf/README.md#L94)
- [README.md:170](/Users/simon/Code/parameter-golf/README.md#L170)

Status:
- `PASS` for the main training run we are promoting

Evidence:
- [jepa01_train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/jepa01_train.log)
  - `stopping_early: wallclock_cap train_time:599393ms`

Important caveat:
- The under-cap export was produced by a separate storage-only pass after the main training run.
- For a final competition submission, the accepted export path needs to be integrated into the scored `train_gpt.py` flow in a way that still respects the official time interpretation.

### 5. Evaluation must run in under 10 additional minutes
Rule source:
- [README.md:160](/Users/simon/Code/parameter-golf/README.md#L160)
- issue [#1017](https://github.com/openai/parameter-golf/issues/1017), “Compute”, item 6

Status:
- `PASS`

Evidence:
- main over-cap candidate:
  - [jepa01_train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/jepa01_train.log)
  - `final_int6_sliding_window_exact ... eval_time:93716ms`
- under-cap storage-only candidate:
  - [storage_pass_allint6.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/storage_pass_allint6.log)
  - `storage_pass_int6_sliding_window_exact ... eval_time:76028ms`

### 6. No evaluation-time training on validation data unless counted properly
Rule source:
- [README.md:160](/Users/simon/Code/parameter-golf/README.md#L160)
- issue [#1017](https://github.com/openai/parameter-golf/issues/1017), “When `val_bpb` Is Meaningful”

Status:
- `PASS`

Why:
- `TTT_ENABLED=0` in the promoted leader-stack JEPA path
- no validation-driven adaptation is used in the reported candidate
- the reported final metric is from ordinary quantized roundtrip/sliding-window eval, not from TTT

### 7. Three independent runs are required for a record PR
Rule source:
- [README.md:172](/Users/simon/Code/parameter-golf/README.md#L172)
- issue [#1017](https://github.com/openai/parameter-golf/issues/1017), “PR Contents”, item 11

Status:
- `PENDING`

State:
- seed `1337` is complete
- seeds `42` and `2026` are currently running in the chained follow-up on the same `8xH100` pod

## JEPA-Specific Claim Checks

### Can we legitimately claim “JEPA helps”?
Status:
- `YES`

Why:
- We built a clean isolation lane in [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py).
- On `8xH100`, matched control vs JEPA runs showed a positive delta in the leaky-MLP + EMA isolation lane.
- In the stronger leader-stack translation lane, longer `1xH100` confirmatory runs also showed JEPA `0.10` beating the matched control.

Evidence:
- [PLAN_AND_PROGRESS.md](/Users/simon/Code/parameter-golf/PLAN_AND_PROGRESS.md)
- [runpod_artifacts/2026-03-31_jepa_iso_8xh100_ema/records/tmp_control/train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-31_jepa_iso_8xh100_ema/records/tmp_control/train.log)
- [runpod_artifacts/2026-03-31_jepa_iso_8xh100_ema/records/tmp_jepa01/train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-31_jepa_iso_8xh100_ema/records/tmp_jepa01/train.log)

### Can we claim this submission is “true byte-level JEPA” under Will DePue’s merch condition?
Status:
- `NO`

Why:
- The current competition-facing candidate uses SentencePiece `sp1024`, not byte-level input.
- The merch condition quoted by Will requires no tokenizer and byte-level input.

What we can say instead:
- We ran a separate byte-level JEPA program and showed clean JEPA-positive evidence in the isolation lane.
- The promoted competition candidate is JEPA-enabled, but it is not the byte-level/no-tokenizer JEPA path referenced in the merch comment.

## Main Ambiguities / Risks

### A. Single-file legality
This is the biggest current risk.

We should assume reviewers will want:
- one submission-local `train_gpt.py`
- no imported research wrapper/base split for scored code

### B. Storage pass integration
The under-cap result currently comes from a post-training export/eval script, not from the main scored training script.

To reduce legality risk, the final PR should:
- merge the accepted storage export recipe into the final `train_gpt.py`
- emit the final under-cap metric directly from that one script

### C. JEPA attribution in the writeup
We should be precise:
- JEPA is a meaningful contributor in our evidence ladder
- but the final competition candidate is not “JEPA alone”
- it is JEPA added to a stronger leader stack, after isolation work established that JEPA itself was beneficial

## Bottom Line

Current status:
- `scientifically defensible JEPA contribution`: `YES`
- `strong competition candidate`: `YES`
- `submission-legal as currently laid out in files`: `NOT YET`

The candidate becomes much safer for PR submission if we do the following before packaging:
1. flatten wrapper + base + accepted storage export into one `train_gpt.py`
2. emit the under-cap result from that one file
3. finish the remaining two seeds and preserve the logs
