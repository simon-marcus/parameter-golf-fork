# PPM Leave-One-Out N-gram Rescore + Complementary Training

**Goal:** test how much of the full-cache win survives once each token's own
`(context,target)` contribution is removed before n-gram rescoring.

## Key Innovations

### 1. Decoupled Two-Pass N-gram Eval
- Pass 1: Standard sliding-window neural eval, storing per-token model_p and entropy
- Build: Complete n-gram cache from all tokens (vectorized numpy, ~5s)
- Pass 2: Rescore ALL tokens with the full cache, but **leave-one-out** for the
  current token before computing match eligibility and probability

### 2. Order 2-16 N-gram with 16M Buckets
- PR #846 uses order 2-9 with 4M buckets
- Higher order catches longer repetitive patterns
- 16M buckets reduces hash collisions at 64M tokens

### 3. Full-Build Cache With Self-Exclusion
- Keep the fast vectorized full-cache build
- Subtract the current token's own context/full counts at score time
- Preserves the speed advantage while removing the most obvious self-inclusion path

### 4. Complementary Training
- During training, bigram statistics are tracked
- Tokens predictable by bigrams are downweighted (COMPLEMENT_ALPHA=0.5)
- Neural model specializes on hard tokens that n-grams can't predict
- Enables higher eval-time alpha (0.85 vs 0.60)

### 5. Optimized Blending
- Entropy-adaptive alpha with order-dependent center shift
- Per-order multipliers: orders 2-3 suppressed (0.3x), orders 5+ boosted (2.0x)
- alpha_max=0.85 (vs PR #846's 0.60)

## Architecture
- Base: 11-layer transformer (LeakyReLU(0.5)^2, Parallel Muon, EMA, SWA)
- Quantization: GPTQ-lite int6 + lzma
- Eval: sliding window (stride 64) + n-gram two-pass rescore

## Timing Budget (estimated, 8xH100)
| Phase | Time |
|-------|------|
| Training | 600s |
| Diagnostic eval | ~3s |
| GPTQ export + roundtrip | ~25s |
| Pass 1: sliding-window neural | ~100s |
| N-gram cache build | ~5-10s |
| Pass 2: n-gram rescore | ~30-60s |
| **Total eval** | **~165-200s** |

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
- `RUN_ID=ppm_loo_seed<seed>`
- default seeds: `1337,42,2025`

This gives predictable remote log names such as:
```text
logs/ppm_loo_seed1337.txt
logs/ppm_loo_seed42.txt
logs/ppm_loo_seed2025.txt
```

The launcher defaults to:

```bash
NGRAM_LEAVE_ONE_OUT=1
```

Set `NGRAM_LEAVE_ONE_OUT=0` only for explicit A/B comparison against the older
full-self-inclusion behavior.
