# Codex Scylla 2

## Purpose

This folder is a dedicated Scylla-v2 eval-time adaptation lab.

The tokenizer problem is solved:
- Scylla-v2 is byte-exact
- the full exported bundle passes strict audit
- `val_bpb` can now be computed correctly

The remaining problem is conversion of Scylla's strong training and quantization behavior into a strong final legal score.

## Working Thesis

Scylla is not failing as a tokenizer. It is being adapted with a TTT recipe that is still too SP-shaped.

What that means in practice:
- Scylla benefits from richer byte coverage and stronger pre-TTT quality
- but the current TTT pass does not harvest enough of that advantage
- the most plausible fixes are in the adaptation regime, not the tokenizer itself

The most important Scylla-specific hypotheses are:
- larger chunks help because byte-native tokens need more context per update
- freezing embeddings helps because byte embeddings carry low-level codebook structure, not document-specific semantics
- freezing more early blocks helps because those layers are still composing bytes into meaning
- stable TTT minibatches may matter more than noisy ones because byte-native fallback tokens are semantically flat

## Design Principles

This workspace is intentionally not a broad autoresearch lane.

For now it is a disciplined hand-sweep lab:
- small number of high-value configs
- clear hypotheses per config
- same legal score-first TTT protocol
- same exact Scylla-v2 bundle

If this finds a stable good region, we can then build a narrow config-search autoresearch loop on top of it.

## Phase 1 Matrix

All runs use the exact Scylla-v2 bundle and the same legal score-first TTT path.

| Run | Chunk Tokens | LR | Epochs | Freeze Blocks | Freeze Emb | Batch Seqs | Hypothesis |
|---|---:|---:|---:|---:|---:|---:|---|
| C0 | 34304 | 0.0015 | 3 | 2 | 0 | 32 | Current best smoke control |
| C1 | 34304 | 0.0015 | 3 | 2 | 1 | 32 | Pure embedding-freeze isolation test |
| C2 | 65536 | 0.0015 | 3 | 2 | 1 | 32 | Larger byte span with minimal other changes |
| C3 | 65536 | 0.0010 | 2 | 4 | 1 | 32 | Calmer richer updates with more lower-stack protection |
| C4 | 98304 | 0.0010 | 2 | 4 | 1 | 32 | Very large byte span per adaptation step |
| C5 | 65536 | 0.0010 | 2 | 6 | 1 | 64 | Late-layer-only adaptation with extra-stable TTT gradients |

The core creative bet here is not "turn random knobs." It is:
- move adaptation upward in the stack
- stop rewriting byte-level codebooks during evaluation
- give each TTT update more bytes and more stable gradient signal

## How To Run

Single config on a `1xH100` pod:

```bash
ssh root@HOST -p PORT "cd /workspace/parameter-golf && \
  USE_COMPILE=0 NPROC_PER_NODE=1 MAX_WALLCLOCK_SECONDS=180 \
  bash records/codex_scylla_2/launch_sweep.sh C3"
```

Full sweep:

```bash
ssh root@HOST -p PORT "cd /workspace/parameter-golf && \
  USE_COMPILE=0 NPROC_PER_NODE=1 MAX_WALLCLOCK_SECONDS=180 \
  bash records/codex_scylla_2/launch_sweep.sh ALL"
```

Analysis:

```bash
python3 records/codex_scylla_2/analyze_sweep.py records/codex_scylla_2/runs
```

## Promotion Rule

Promote only if a config beats the current SP smoke baseline and also improves on Scylla control `C0`.

Only then should we spend `8xH100` budget.

## PR1019 Transfer Pair

To compare the cleanest imports from PR `#1019`, use this matched pair:

| Run | Chunk Tokens | LR | Ep | Freeze Blocks | Freeze Emb | Prequant | XSA Last N | Warmdown | Purpose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T0 | 34304 | 0.0010 | 3 | 2 | 0 | 1 | 4 | 3500 | Current Scylla control on the modern eval path |
| T1 | 34304 | 0.0010 | 3 | 2 | 0 | 1 | 11 | 4000 | PR `#1019` transfer test: XSA-all + longer warmdown |

These are intentionally matched on:
- tokenizer
- exact dataset bundle
- TTT regime
- batch geometry
- seed

So the only material differences are:
- `XSA_LAST_N=11` instead of `4`
- `WARMDOWN_ITERS=4000` instead of `3500`

## Notes

- The script in this folder is a Scylla-specific fork of the legal score-first TTT script.
- The main additional lever relative to the standard promoted script is `TTT_FREEZE_EMBEDDINGS`.
- This workspace is the right place to refine Scylla-specific TTT before building any narrow autoresearch loop.
