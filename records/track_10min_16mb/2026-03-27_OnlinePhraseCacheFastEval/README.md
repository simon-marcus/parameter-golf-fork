# Online Phrase Cache + Fast Eval

**Goal:** preserve the strong online-cache descent from the calibrated branch
while making evaluation operationally viable for rapid iteration.

## Thesis

The calibrated branch showed that the online phrase-cache stack learns fast, but
the default `8192`-token microchunks made evaluation too slow for practical
record iteration. This fork keeps the same scorer and shifts only the runtime
knobs:

- keep the stronger 11-layer PPM-style model from our recent submissions
- keep the legal score-first online cache path
- keep the long-phrase cache and online alpha calibration
- increase the online update granularity aggressively to cut Python overhead
- shorten the calibration prefix so tuning finishes earlier

## Key Innovations

### 1. Coarser score-first online cache updates
- tokens are scored first
- only after a scored chunk finishes do its tokens enter the cache
- default cache update granularity is `65536` tokens
- this intentionally trades some within-chunk adaptation for much better eval
  throughput without reintroducing two-pass legality questions

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

### 5. Shortened online alpha calibration
- collects early scored tokens only
- grid-searches a tiny set of `ngram_alpha_max`, `ngram_entropy_center`, and
  `phrase_alpha_max` choices
- now defaults to a smaller `65536-131072` token prefix
- applies the selected settings only to later tokens, preserving score-first
  ordering

### 6. Stronger base model retained
- unlike the tiny-baseline online cache approach, this keeps our recent
  larger model family, quantization path, and complementary-training setup

## Defaults

- `NGRAM_MODE=online`
- `NGRAM_ONLINE_CHUNK_TOKENS=65536`
- `NGRAM_NUM_BUCKETS=8388608`
- `NGRAM_CALIBRATION_ENABLED=1`
- `NGRAM_CALIBRATION_MIN_TOKENS=65536`
- `NGRAM_CALIBRATION_MAX_TOKENS=131072`
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
online_phrase_fast_seed1337
online_phrase_fast_seed42
online_phrase_fast_seed2025
```

## Notes

- `NGRAM_MODE=two_pass` still exists as a fallback A/B path.
- `NGRAM_LEAVE_ONE_OUT` remains available for explicit testing, but the default
  online mode does not rely on it.
- This fork is intended to measure the speed-vs-score tradeoff before trimming
  phrase lengths or removing other scorer components.
