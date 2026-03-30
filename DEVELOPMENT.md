# Project

This repository is an OpenAI Parameter Golf entry: train the best language model that fits in a 16MB artifact and trains within the competition budget, judged by post-quantization `val_bpb` on FineWeb.

The current project focus is tokenizer selection and tokenizer/runtime infrastructure:

- remove unnecessary runtime dependencies from scoring-critical paths
- search for tokenizers that help a tiny artifact-constrained model, not just tokenizers that look efficient in isolation
- promote only the strongest tokenizer candidates into proxy model validation

# Strategy

## Working assumptions

- Cheap tokenizer metrics are useful only as a screening signal.
- The real arbiter is proxy-trained `postquant_val_bpb`, and then full competition-style training.
- Large-vocab “wins” are suspicious unless they survive equal-shape or equal-budget proxy validation.
- The most promising tokenizer regime is now small TokenMonster vocabularies, not larger SentencePiece variants.

## What we tried

- SentencePiece autoresearch aligned to FineWeb and budget-aware scoring.
  - Result: useful exploration, but it saturated and did not transfer well in proxy validation.
- Broad TokenMonster sidecar screening.
  - Result: small TokenMonster variants looked promising; larger TokenMonster variants did not hold up well.
- Proxy-calibrated mixed-family autoresearch.
  - Result: enough evidence to stop broad family search and narrow to TokenMonster-only exploration.
- TokenMonster-only autoresearch.
  - Result: after broadening away from tiny resize-only local search, the best line became `english-1024-clean-v1` with light pruning/shrinking.

## Current thesis

The best current tokenizer candidate is a lightly pruned TokenMonster derivative:

- base: `english-1024-clean-v1`
- current best local candidate: `autoresearch/tokenmonster_discovery/experiments/0054/candidate.vocab`
- screening result:
  - `score = 4.073924`
  - `vocab_size = 998`
  - `tokens_per_byte = 0.405436`
  - `dead_vocab_frac = 0.0200`

Interpretation:

- the improvement appears to come from removing wasted vocab capacity while preserving the strong `clean-v1` tokenization behavior
- this is exactly the kind of tokenizer change that could matter under Parameter Golf’s artifact budget
- multi-seed proxy confirmation now supports promotion:
  - `0054_candidate`: mean postquant `2.197508`
  - `english-1024-clean-v1`: mean postquant `2.203397`
  - baseline `sp1024`: mean postquant `2.279679`

# Current Work and Next Steps

## Current status

The promoted tokenizer path has already cleared its first real training gate.

Completed:

- exported the local matched-data bundle for `0054_candidate`
  - `/Users/simon/Code/parameter-golf-local/tm0054_full_export`
- exported runtime scoring metadata
  - `tokenizers/candidate.meta.npz`
- ran a first real training comparison on the local 10-train-shard bundle
  - baseline `sp1024`: postquant `1.30739032`
  - `tm0054_candidate`: postquant `1.25801162`

Interpretation:

- `tm0054_candidate` beat baseline on a real training run, not just on tokenizer screening or proxy model selection
- the artifact was also slightly smaller than baseline on that run
- tokenizer work is now a real competition lever, not just an exploratory side lane

## Immediate result

The first legal ablation ladder is complete on the matched local bundle.

Results:

- baseline `sp1024`
  - roundtrip: `2.78423303`
  - sliding-window: `2.76644428`
  - legal TTT: `1.67267635`
- `tm0054_candidate`
  - roundtrip: `2.97167024`
  - sliding-window: `2.94326425`
  - legal TTT: `1.61602134`

Interpretation:

- `tm0054_candidate` is worse on plain roundtrip and sliding-window eval under the legal stack
- `tm0054_candidate` is materially better on `legal_ttt_exact`
- the tokenizer is therefore not a universal model-quality improvement here, but it does appear to improve the score-first legal TTT path specifically

This is the result that matters most for competition strategy, because the strongest current legal path is the legal TTT path.

## Current execution plan

The active launcher path is:

- `records/track_10min_16mb/2026-03-29_PPM_LOO_TokenizerBundle/train_gpt.py`
  - derived from the best legal `PPM_LOO` stack
  - n-gram pass can be disabled cleanly
  - backported to support metadata-driven tokenizer LUT loading so it can score `tm0054`
- `modal_train_legal_tokenizer_bundle.py`
  - mounts both the baseline bundle and the exported `tm0054` bundle
  - runs baseline and `tm0054` through the same legal recipe
  - records:
    - `final_int6_roundtrip_exact`
    - `final_int6_sliding_window_exact`
    - `legal_ttt_exact`

## Current next step

Confirm the legal-TTT improvement with at least one more seed on the same ladder.

If the second seed holds:

- promote `tm0054_candidate + legal TTT` into the current best legal competition path
- stage the tokenizer bundle on the real competition training environment
- run the stronger seed / scale confirmation
- if stable, make `tm0054_candidate` the active tokenizer for record-style legal runs

That second-seed confirmation did hold:

- seed `1337`
  - baseline legal TTT: `1.67267635`
  - `tm0054_candidate`: `1.61602134`
- seed `2024`
  - baseline legal TTT: `1.67977535`
  - `tm0054_candidate`: `1.62609262`

So the active promotion decision is now:

- promote `tm0054_candidate + legal TTT`
- run a stronger 8xH100 legal-only confirmation using the `PPM_LOO`-derived script with `NGRAM_ENABLED=0`

If `tm0054_candidate` only wins on standard/sliding but not on legal TTT:

- keep it as the best tokenizer for plain model quality
- inspect whether TTT-specific adaptation interacts badly with the tokenizer
- compare against `english-1024-clean-v1` as the fallback TokenMonster control

## Ongoing lanes

- Main TokenMonster lane:
  - disciplined, broadened after stagnation
  - current best source of useful candidates
- Creative TokenMonster lane:
  - intentionally more exploratory
  - useful as a challenger source, not the main decision-maker unless it clearly beats the main lane

## Work in progress

The immediate infrastructure task is now the full competition-data `tm0054` bundle, not the smaller 10-shard local export.

Current reality:

- the live RunPod `tm0054` runs are still using the matched local subset:
  - `10` train shards
  - `1` val shard
- that is good enough for promotion and ladder work
- it is not the final legal competition dataset path

Full-data target for this repo:

- baseline source family: published `sp1024` export
- training prefix: `80` train shards
- validation: full fixed val split (`1` val shard in the current published export)
- tokenizer runtime paths preserved exactly:
  - `TOKENIZER_PATH -> candidate.vocab`
  - `TOKENIZER_META_PATH -> candidate.meta.npz`

Builder path:

- [build_tm0054_competition_bundle.py](/Users/simon/Code/parameter-golf/data/build_tm0054_competition_bundle.py)
- this:
  - downloads or reuses the published `sp1024` baseline prefix
  - retokenizes it in shard order into `fineweb10B_tm0054`
  - preserves validation ordering
  - validates shard headers plus expected train/val counts

Target output bundle:

- `/Users/simon/Code/parameter-golf-local/tm0054_competition_export/datasets/fineweb10B_tm0054/`
- `/Users/simon/Code/parameter-golf-local/tm0054_competition_export/tokenizers/candidate.vocab`
- `/Users/simon/Code/parameter-golf-local/tm0054_competition_export/tokenizers/candidate.meta.npz`
- `/Users/simon/Code/parameter-golf-local/tm0054_competition_export/manifest.json`

Next command path after bundle validation:

- switch the legal 8xH100 launchers from the 10-shard local export to the full validated `tm0054` competition bundle
- rerun the promoted legal TTT recipe on that full-data path
