# BeatCacheMoney — Target: val_bpb < 0.080

**Goal: Beat PR #933 (CacheMoney, 0.0804 BPB)**

## Base
PPM_LOO_NgramRescore (0.0991 BPB on 8xH100, 144s eval)

## Run 1: Full features (FAILED — 0.1316 BPB)

All features enabled: phrase cache + T=0.85 + calibration + orders 2-20 + 32M buckets.

**Root cause of regression:** Phrase cache at 4M buckets with 62M val tokens = 14.8 entries/bucket.
Hash collisions make phrase probabilities pure noise. The phrase blend (alpha ≥ 0.44) overwrites
good n-gram estimates for 100% of tokens with this noise. See diagnosis in conversation.

Key stats from run 1:
- Training: 6166 steps, 97.33ms/step (SDPA fallback, no flash-attn-3)
- Neural: 1.1260 BPB sliding-window (comparable to baseline 1.1281)
- N-gram match: 100%, Phrase match: 100% (everything matches = red flag)
- Calibration: alpha_max=0.900, center=3.5, phrase_max=0.999
- Final: **0.1316 BPB** (33% regression from 0.0991 baseline)

## Run 2: N-gram only (PHRASE_ENABLED=0)

Changes from run 1:
- Phrase cache disabled
- Calibration grid widened: alpha_max ∈ [0.70, 0.80, 0.90, 0.95, 0.99]
- Everything else unchanged: orders 2-20, 32M buckets, T=0.85, LOO, calibration

Expected: ≤ 0.099 (match or beat baseline with more orders + larger tables + temperature + calibration)

## Changes from PPM_LOO_NgramRescore

1. **N-gram orders 2-20** (up from 2-12): 4 extra primes, extended ORDER_MULTS
2. **32M n-gram buckets** (up from 16M default / 4M in actual baseline run)
3. **Temperature sharpening T=0.85**: logits divided by T before softmax in Pass 1
4. **Alpha calibration**: grid search on 5% of tokens (5×4 = 20 combos n-gram only)
5. **Phrase cache** (DISABLED by default): needs ≥64M buckets for two-pass to avoid collision noise
6. **SDPA fallback**: flash_attn_interface import with F.scaled_dot_product_attention fallback
7. **USE_COMPILE gating**: all torch.compile calls gated on USE_COMPILE env var
