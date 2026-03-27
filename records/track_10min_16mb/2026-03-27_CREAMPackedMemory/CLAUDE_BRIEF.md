# CREAM Brief For Claude Opus 4.6 High

You are helping prepare a record-quality submission for OpenAI parameter-golf
`track_10min_16mb`.

## Objective

Beat or match PR `#933` ("CacheMoney", `0.0804` BPB) as the main target.

Treat PR `#944` as a possible rules-regime shift, but do not optimize for it
yet. Treat our previous online-only cache line as reference material, not the
main attack.

## Repo / Workspace

- Repo root: `/Users/simon/Code/parameter-golf`
- Current date: `2026-03-27`
- Main new branch:
  - `/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-27_CREAM`

## Competitive Context

### PR #913
- Online cache breakthrough
- `0.0887` BPB
- Tiny model
- Phrase cache
- Temperature sharpening

### PR #933
- Main target
- `0.0804` BPB
- `7.47 MB` artifact
- `339s` eval on `8xH100`
- Tiny `4.2M` model in FP16
- Two-pass full-rescore
- Leave-one-out correction
- Phrase cache
- Online alpha calibration

### PR #944
- `0.01654407`
- Packed causal memory loaded from artifact
- Possible regime shift if organizers allow that interpretation
- Not current primary target

## What We Learned From Our Recent Work

Our online-only score-first cache line was useful research, but not the main
winning path against `#933`.

Fast-eval online run result:
- log:
  - `/Users/simon/Code/parameter-golf/logs/runpod_fast_eval/online_phrase_fast_seed1337.txt`
- final:
  - `val_bpb = 0.16115741`
- online eval time:
  - `973.2s`

Interpretation:
- the cache mechanism is real and powerful
- the descent curve was impressive
- but absolute score and runtime are not competitive enough against `#933`

## CREAM Current State

We created:
- `/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-27_CREAM`
- `/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-27_CREAMPackedMemory`

It is based on the strongest relevant local substrate:
- two-pass full-rescore
- leave-one-out n-gram scoring

We already patched `CREAM` toward a `#933`-style evaluator:
- phrase cache added
- temperature-sharpened pass-1 eval added
- cheap cache-blend calibration added
- more efficient per-position hashing added
- tiny-model launch defaults added

Key files:
- `/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-27_CREAM/train_gpt.py`
- `/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-27_CREAM/launch.sh`

Current launch defaults in `CREAM`:
- model:
  - `NUM_LAYERS=6`
  - `MODEL_DIM=256`
  - `NUM_HEADS=4`
  - `NUM_KV_HEADS=2`
  - `MLP_MULT=3.0`
- n-gram:
  - orders `2-16`
  - `16M` buckets
  - leave-one-out on
  - temperature `0.85`
  - `alpha_max` grid: `0.90,0.94,0.97,0.99`
  - entropy center grid: `2.5,3.0,3.5`
- phrase:
  - lengths `64,48,32,16`
  - `8M` buckets
  - phrase alpha grid: `0.980,0.990,0.995`

Static checks passed:
- `python3 -m py_compile` on `train_gpt.py`
- `bash -n` on launch scripts

## Local Reference Files

Use these as references:

- `/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-26_PPM_NgramRescore/train_gpt.py`
- `/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-26_PPM_LOO_NgramRescore/train_gpt.py`
- `/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-27_OnlinePhraseCacheCalibrated/train_gpt.py`
- `/Users/simon/Code/parameter-golf/logs/runpod_fast_eval/online_phrase_fast_seed1337.txt`

## What We Want From You

We want an aggressive, practical plan to beat `#933`.

Please answer:

1. What exact model/training recipe should we use if the cache does most of the
   work?
2. What cache engine design is most likely to beat `#933`?
3. How should leave-one-out be structured correctly and cheaply?
4. What phrase lengths and phrase-table strategy should we use?
5. How should online alpha calibration be improved?
6. Where is `#933` still leaving points on the table?
7. What first experiments should we run, in order, to get into the `0.08x`
   regime quickly?
8. What should we discard entirely?

## Constraints / Preferences

- Pragmatic
- competition-focused
- willing to discard old work
- specific and actionable
- clean-room implementation is allowed if it is the better choice

## Desired Output Format

Please structure your answer as:

1. concise verdict on the best path to beat `#933`
2. ranked plan of attack
3. concrete architectural recommendations
4. first 5 experiments to run
5. red flags or legality risks
