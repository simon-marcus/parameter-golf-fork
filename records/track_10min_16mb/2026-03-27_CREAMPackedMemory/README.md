# CREAM Packed Memory

**Goal:** open a direct attack lane on the packed-memory frontier established by
PRs `#944` and `#962`, while keeping the simpler `CREAM` branch as a fallback.

## Thesis

The old two-pass cache-first game is no longer enough on its own. This branch
persists a compact hashed training n-gram memory into the artifact itself, then
uses that memory to warm-start a single-pass causal eval.

The first version is intentionally simple:

- tiny transformer prior
- packed order-`2..9` training cache stored in the artifact
- single-pass causal eval seeded from that packed cache
- online cache updates after scoring each chunk
- no phrase cache in the reported path yet

## Current Defaults

- `NGRAM_MODE=packed_online`
- `ARTIFACT_NGRAM_EXPORT=1`
- `ARTIFACT_NGRAM_MIN_ORDER=2`
- `ARTIFACT_NGRAM_MAX_ORDER=9`
- `ARTIFACT_NGRAM_BUCKETS=32768`
- `NGRAM_ONLINE_CHUNK_TOKENS=131072`
- `PHRASE_ENABLED=0`

## What This Branch Is Testing

1. Whether packed training memory inside the artifact can dominate the tiny
   neural prior even without phrase memory.
2. Whether a compact fixed-budget hashed cache can buy leaderboard-scale gains
   before more elaborate gating or packing tricks.
3. Whether the artifact byte budget is better spent on memory than on model.

## Usage

```bash
# Smoke test
bash launch.sh smoke

# Full run
bash launch.sh base
```

## Notes

- This is the first packed-memory attack lane, not the final design.
- Expected next steps:
  - better value-aware packing
  - a smarter learned gate
  - sparse premium higher-order memory
  - explicit artifact budget search between model bytes and memory bytes
