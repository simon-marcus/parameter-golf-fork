# CREAM: Cache Rules Everything Around Me

**Goal:** build a direct `#933` challenger around a tiny-model, two-pass,
leave-one-out cache-first evaluator.

## Current Direction

- tiny model first, not strong model first
- two-pass full-rescore from the start
- leave-one-out scoring as a hard requirement
- phrase cache and alpha calibration as first-class parts of the evaluator
- speed profiling in every eval phase

## Immediate Priorities

### 1. Rebuild toward `#933`
- target the `0.080x` regime first
- reproduce the core shape before trying to out-innovate it
- keep this branch as the main record attack unless the rules clearly shift toward
  packed-memory approaches

### 2. Keep only what helps the cache-first path
- anything from the older online-only line is reference-only unless it clearly
  improves the two-pass evaluator
- the neural model is here to support the cache blend, not to win on its own

### 3. Iterate on evaluator quality and speed together
- every serious run should log pass-1, cache-build, pass-2, and calibration time
- every scoring improvement has to survive runtime scrutiny

## Architecture
- current substrate copied from the leave-one-out two-pass PPM branch
- expected evolution:
  - much smaller base model
  - stronger phrase layer
  - explicit online alpha calibration on top of full-rescore

## Status

This branch is intentionally a fresh attack lane. It should diverge quickly from
the original PPM LOO branch toward:

- tiny FP16 storage if quantization is not buying enough
- `#933`-style temperature sharpening
- phrase lengths closer to `64/48/32/16`
- calibrated aggressive cache trust
- cleaner, auditable leave-one-out scoring

## Usage
```bash
# Smoke test (1xGPU, 3min)
bash launch.sh smoke

# Full run (8xGPU, 10min)
bash launch.sh base
```

## Multi-Seed Package
```bash
# Standard full package
bash launch_multiseed.sh

# Subset
SEEDS=1337,42 bash launch_multiseed.sh

# Smoke package
MODE=smoke SEEDS=1337,42,2025 bash launch_multiseed.sh
```

The multi-seed wrapper fixes:
- `RUN_ID=cream_seed<seed>`
- default seeds: `1337,42,2025`

This gives predictable remote log names such as:
```text
logs/cream_seed1337.txt
logs/cream_seed42.txt
logs/cream_seed2025.txt
```
