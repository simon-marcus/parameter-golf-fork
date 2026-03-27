# PPM N-gram Rescore + Complementary Training

**Target: val_bpb < 0.14** (beating PR #846's 0.1434)

## Key Innovations

### 1. Decoupled Two-Pass N-gram Eval
- Pass 1: Standard sliding-window neural eval, storing per-token model_p and entropy
- Build: Complete n-gram cache from all tokens (vectorized numpy, ~5s)
- Pass 2: Rescore ALL tokens (not just 15/63 chunks) with full cache

### 2. Order 2-16 N-gram with 16M Buckets
- PR #846 uses order 2-9 with 4M buckets
- Higher order catches longer repetitive patterns
- 16M buckets reduces hash collisions at 64M tokens

### 3. Per-Token-Position Cache (via full build)
- PR #846 updates cache per 1M-token chunk (63 chunks)
- We build the complete cache in one vectorized pass, then score all positions
- Eliminates cold-start entirely (every token scored with full cache)

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
