# Legality Check

This document is intentionally stricter than the narrative README. Its purpose is to answer: "What do we still need to verify or change before this is safe to submit?"

## Sources Checked

Official:
- [OpenAI challenge README](https://raw.githubusercontent.com/openai/parameter-golf/main/README.md)

Unofficial but useful:
- [Issue #1017: A Field Guide to Valid Submissions](https://github.com/openai/parameter-golf/issues/1017)

The issue is not an official rule source, but it is a useful checklist for catching easy mistakes.

## Current Candidate

Current best candidate path:
- training/eval wrapper: [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py)
- base imported script: [train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py)
- storage-only export/eval helper: [leader_stack_storage_pass.py](/Users/simon/Code/parameter-golf/leader_stack_storage_pass.py)

Current best full-scale result:
- under-cap exact `val_bpb`: `1.12271348`
- estimated artifact size: `15,919,760` bytes

## Rule-By-Rule Audit

### 1. Artifact must be <= 16,000,000 bytes

Status: currently satisfied by the storage-only export recipe.

Evidence:
- [storage_pass_allint6.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/storage_pass_allint6.log)

Logged values:
- `storage_pass quant_bytes=15826872`
- `storage_pass code_bytes=92888`
- `storage_pass estimated_total=15919760`

Risk:
- this is currently an evaluated offline storage pass, not yet the default export path of the candidate `train_gpt.py`

Required before submission:
- integrate this exact export logic into the final single-file submission artifact

### 2. All scored code must live in one `train_gpt.py`

Status: not yet satisfied.

Current issue:
- [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py) imports the March 23 base file dynamically
- that is convenient for research, but it is not the intended final packaging shape

Required before submission:
- flatten the wrapper + base into one self-contained `train_gpt.py`
- ensure the storage fix is inside that same file

### 3. No network calls or runtime side information

Status: satisfied by the intended final artifact shape, but must be rechecked after packaging.

Current runpod prep uses network for staging, but that is outside the submitted artifact and before timed evaluation.

What matters:
- the final submission code must not download anything during training or evaluation
- all needed code and weights must already be in the artifact

### 4. Training must fit the 10-minute 8xH100 budget

Status: satisfied.

Evidence:
- [jepa01_train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/jepa01_train.log)

Key lines:
- `step_avg:84.62ms`
- `stopping_early: wallclock_cap ... step:7083/20000`

### 5. Evaluation must fit the 10-minute additional budget

Status: appears satisfied.

Evidence:
- training run log: [jepa01_train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/jepa01_train.log)
- storage pass log: [storage_pass_allint6.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/storage_pass_allint6.log)

Observed eval times:
- over-cap run sliding-window exact eval: `93716ms`
- under-cap storage pass sliding-window exact eval: `76028ms`

These are comfortably below the 10-minute evaluation cap.

### 6. No compute transfer from evaluation into training

Status: current method is likely legal, but wording must be careful.

Why this matters:
- the official README and issue #1017 both treat state changes from validation data as evaluation compute
- if validation data is used to modify model state, that belongs to the evaluation protocol and must respect causal constraints

What we are doing:
- the storage-only pass loads the already trained checkpoint
- it changes only the quantization/export representation
- it does not fit weights on validation data
- it then evaluates the resulting fixed predictor

Current reading:
- this is best understood as part of model export/serialization, not adaptation on validation data
- because it does not use validation statistics to choose weights or tune the model, it does not look like forbidden compute transfer

Risk:
- if we were to choose the storage recipe *because* it looked best on validation after many repeated tries, that would drift toward offline val-set optimization

Required before submission:
- describe this as a predetermined export recipe, not as per-run tuning against the validation result
- avoid broad storage hyperparameter fishing directly against validation

### 7. Strict causal left-to-right evaluation

Status: satisfied for the current candidate.

Why:
- `TTT_ENABLED=0`
- final metric is the script's ordinary left-to-right/sliding-window evaluation path
- there is no eval-time weight adaptation in the submitted candidate path

This is an important simplification versus the March 23 legal-TTT ancestor.

### 8. Seed brute forcing / offline validation optimization

Status: needs disciplined presentation.

What is okay:
- challenge asks for logs from at least 3 independent runs
- we are running seeds `1337`, `42`, and `2026`

What is not okay:
- cherry-picking a seed because it looks best on validation without reporting the 3-seed picture

Required before submission:
- report all 3 seeds
- compute mean/std cleanly
- do not present only the single best seed as if that were the full evidence

### 9. JEPA claim scope

Status: must be phrased carefully.

What we can claim:
- JEPA improved a matched control in the isolation lane at full scale
- JEPA also improved the stronger leader stack at `0.10`
- the final candidate is therefore meaningfully JEPA-based, not just JEPA-branded

What we should not claim:
- that this is a pure JEPA model
- that this is the byte-level/no-tokenizer JEPA requested in Will DePue's merch comment

### 10. Byte-level organizer note

Status: not satisfied for the current strongest candidate.

Relevant fact:
- Will's merch note explicitly asked for a byte-level/no-tokenizer JEPA
- our strongest candidate is on `sp1024`

So:
- this candidate is a JEPA competition submission
- it is not the exact byte-level merch-targeted JEPA variant

## Bottom Line

If we had to summarize the current legality state in one sentence:

The current JEPA candidate looks **methodologically legal**, but it is **not yet packaging-legal for PR submission** until we flatten it into a single `train_gpt.py`, integrate the storage fix into that file, and finish the 3-seed reporting cleanly.

## Submission TODO

Before opening a real PR:
- flatten wrapper + imported base into one `train_gpt.py`
- integrate the storage fix into that single file
- rerun or verify the single-file path reproduces the under-cap exact result
- pull and store the seed `42` and `2026` logs
- write `submission.json`
- add final `requirements.txt` only if additional packages are truly needed
