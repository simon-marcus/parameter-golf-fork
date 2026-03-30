# Tokenizer Work

This document summarizes the tokenizer work in this repo for Parameter Golf.

## Goal

We are not trying to build the most elegant tokenizer in isolation.
We are trying to find the tokenizer that helps a very small model do best on the competition metric:

- final post-quantization `val_bpb`
- under a hard artifact budget
- with very limited training time

That means tokenizer work is only useful if it survives contact with the model.

## What We Built

### 1. Runtime tokenizer metadata

We removed the runtime dependency on `sentencepiece` for fixed-tokenizer competition runs.

Relevant files:
- [train_gpt.py](/Users/simon/Code/parameter-golf/train_gpt.py)
- [data/export_sentencepiece_meta.py](/Users/simon/Code/parameter-golf/data/export_sentencepiece_meta.py)
- [data/export_tokenmonster_meta.py](/Users/simon/Code/parameter-golf/data/export_tokenmonster_meta.py)

The runtime only needs exact per-token metadata for `val_bpb` accounting:
- token byte length
- leading-space behavior
- boundary/control behavior

So we now export/load compact metadata artifacts instead of requiring SentencePiece at runtime.

### 2. Cheap tokenizer autoresearch

We first built a cheap loop around reconstructed FineWeb text:

- [autoresearch_tokenizer.py](/Users/simon/Code/parameter-golf/autoresearch_tokenizer.py)
- [watch_tokenizer_autoresearch.py](/Users/simon/Code/parameter-golf/watch_tokenizer_autoresearch.py)
- [data/extract_text_sample_from_shards.py](/Users/simon/Code/parameter-golf/data/extract_text_sample_from_shards.py)
- [data/evaluate_tokenizer_on_text_sample.py](/Users/simon/Code/parameter-golf/data/evaluate_tokenizer_on_text_sample.py)

This loop searched tokenizer efficiency on a deterministic multi-shard FineWeb sample.
It initially optimized a cheap screening score built from:

- `tokens_per_byte`
- `dead_vocab_frac`
- estimated vocab-linked artifact cost

### 3. Mixed-backend proxy validation

We then added a proxy harness to test whether tokenizer wins survive model training:

- [proxy_validate_tokenizers.py](/Users/simon/Code/parameter-golf/proxy_validate_tokenizers.py)
- [modal_proxy_validate_tokenizers.py](/Users/simon/Code/parameter-golf/modal_proxy_validate_tokenizers.py)
- [data/build_proxy_tokenizer_dataset.py](/Users/simon/Code/parameter-golf/data/build_proxy_tokenizer_dataset.py)
- [data/aggregate_proxy_tokenizer_results.py](/Users/simon/Code/parameter-golf/data/aggregate_proxy_tokenizer_results.py)

This can compare:
- SentencePiece candidates
- TokenMonster vocabularies

under the same short proxy training recipe.

### 4. Proxy-aware tokenizer autoresearch

We then built a second lane:

- [autoresearch_tokenizer_proxy.py](/Users/simon/Code/parameter-golf/autoresearch_tokenizer_proxy.py)
- [program_tokenizer_proxy.md](/Users/simon/Code/parameter-golf/program_tokenizer_proxy.md)

The first version used a tiny regression fit from proxy outcomes and overfit badly.
The current direction is to consume aggregate proxy results directly instead of trusting a tiny regression.

## What We Learned

### Cheap tokenizer metrics were not enough

The first SentencePiece lane found tokenizer candidates that looked much better than baseline on cheap text metrics.
That was real as a tokenizer-only result.

But proxy validation showed those wins did not transfer cleanly into better trained-model BPB.

### Larger vocabularies were too optimistic

Both larger SentencePiece and larger TokenMonster vocabularies looked attractive on tokenizer-only efficiency.
In proxy runs, many of those candidates were worse once model learning and budget pressure mattered.

### Small TokenMonster is the most promising direction so far

From the completed short multi-seed proxy sweeps in:
- [proxy_calib_short_s1337_v2/summary.json](/Users/simon/Code/parameter-golf/modal_logs/proxy_calib_short_s1337_v2/summary.json)
- [proxy_calib_short_s2024_v2/summary.json](/Users/simon/Code/parameter-golf/modal_logs/proxy_calib_short_s2024_v2/summary.json)
- aggregated in [proxy_calib_aggregate_v2.json](/Users/simon/Code/parameter-golf/modal_logs/proxy_calib_aggregate_v2.json)

the best region is:
- `english-1024-consistent-v1`
- `english-1024-balanced-v1`
- `english-1024-clean-v1`

The best tested small SentencePiece candidates are behind those TokenMonster `1024` variants on this proxy recipe.

## Things We Tried And Rejected

### Rejected or deprioritized

- Large-vocab SentencePiece (`4096`, `6144`, `6400+`)
  Cheap metrics liked them, proxy runs did not.

- Large-vocab TokenMonster (`4096`, `8000`)
  Also looked good on cheap screening, but proxy results were clearly worse.

- Treating tokenizer-only score as enough
  This was the central mistake in the first phase.

- Tiny regression on six proxy datapoints
  It overfit and produced misleading rankings.

- Continuing to push SentencePiece larger and larger
  That search saturated.

### Weak or unhelpful knobs

- `input_sentence_size` increases beyond the already-large sample
  Often produced identical tokenizers.

- many `character_coverage` tweaks at larger SentencePiece vocab sizes
  Usually neutral or worse.

- `unigram` as the main replacement for `bpe`
  Not competitive in the tested regimes.

## Current Best Working Hypothesis

If we want a useful tokenizer prediction pipeline, it should:

- be anchored to repeated proxy results, not just tokenizer-only metrics
- stay in the small-vocab regime first
- treat TokenMonster `1024` and `2048` as the most promising search region
- keep SentencePiece mainly as a control band, not the main optimization frontier

## Current Next Step

Use the aggregated proxy results as the scoring backbone for future tokenizer autoresearch, rather than fitting a fragile regression from too few runs.
