# Online Phrase Cache + Lightweight Calibration

**Goal:** beat the new online-cache frontier with a cleaner evaluator than full
two-pass rescoring and a stronger base model than the tiny-baseline approach.

## Thesis

The current frontier moved from "make the neural model better" to "use the
validation stream as a legal online memory." This candidate pivots accordingly:

- keep the stronger 11-layer PPM-style model from our recent submissions
- discard full-cache two-pass rescoring as the primary record path
- score the stream once, legally, with online cache updates only after each
  scored microchunk
- stack a long-phrase cache on top of the n-gram cache
- calibrate cache trust online from early already-scored tokens, then freeze the
  chosen blend parameters for the rest of eval

## Key Innovations

### 1. Score-first online cache updates
- tokens are scored first
- only after a scored chunk finishes do its tokens enter the cache
- default cache update granularity is `8192` tokens, not `131072+`
- this reduces the chunk-level cold-start penalty without reintroducing
  two-pass legality questions

### 2. Online n-gram cache
- orders `2-12`
- order-adaptive entropy gating
- stronger trust for higher-order matches, suppression for weak low-order hits

### 3. Separate long-phrase cache
- probe lengths `64,56,48,36,28,20,16`
- separate tables per phrase length to avoid cross-length contamination
- blended after n-grams, so exact long-repeat structure can dominate when found

### 4. Temperature-sharpened evaluator
- cache eval sharpens model logits with `NGRAM_TEMPERATURE=0.85`
- this improves the neural prior before cache blending

### 5. Lightweight online alpha calibration
- collects early scored tokens only
- grid-searches a tiny set of `ngram_alpha_max`, `ngram_entropy_center`, and
  `phrase_alpha_max` choices
- applies the selected settings only to later tokens, preserving score-first
  ordering

### 6. Stronger base model retained
- unlike the tiny-baseline online cache approach, this keeps our recent
  larger model family, quantization path, and complementary-training setup

## Defaults

- `NGRAM_MODE=online`
- `NGRAM_ONLINE_CHUNK_TOKENS=8192`
- `NGRAM_NUM_BUCKETS=8388608`
- `NGRAM_CALIBRATION_ENABLED=1`
- `NGRAM_CALIBRATION_MIN_TOKENS=131072`
- `NGRAM_CALIBRATION_MAX_TOKENS=524288`
- `PHRASE_ENABLED=1`
- `PHRASE_NUM_BUCKETS=4194304`
- `PHRASE_PROBE_LENGTHS=64,56,48,36,28,20,16`

## Usage

```bash
# Smoke test
bash launch.sh smoke

# Full run
bash launch.sh base
```

## Multi-Seed Package

```bash
# Standard 3-seed package
bash launch_multiseed.sh

# Custom subset
SEEDS=1337,42 bash launch_multiseed.sh
```

The multi-seed wrapper writes logs under run IDs like:

```text
online_phrase_cal_seed1337
online_phrase_cal_seed42
online_phrase_cal_seed2025
```

## Notes

- `NGRAM_MODE=two_pass` still exists as a fallback A/B path.
- `NGRAM_LEAVE_ONE_OUT` remains available for explicit testing, but the default
  online mode does not rely on it.
