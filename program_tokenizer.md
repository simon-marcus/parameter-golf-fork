# Parameter Golf Tokenizer Program

## Objective
Discover tokenizer candidates that are better than the current `sp1024` baseline for this challenge.

The working proxy objective is to improve tokenizer efficiency on FineWeb text while respecting vocabulary budget:
- lower `tokens_per_byte`
- lower `dead_vocab_frac`
- lower estimated vocab-linked artifact cost
- reasonable `vocab_size`

This is a screening lane, not the final leaderboard metric. A tokenizer only graduates if it later helps short-run and full-run BPB.
Prefer tokenizers that win enough on compression to justify the embedding/head budget they consume.

## Baseline
- Control tokenizer: current SentencePiece `sp1024`
- Text source: reconstructed FineWeb text sample decoded from the existing tokenized shards

## Allowed Experiment Types
1. Train a new SentencePiece tokenizer on the reconstructed sample
2. Re-evaluate the baseline on a larger or different sample
3. Compare a provided external tokenizer if a compatible evaluator is added later

## Priorities
1. Establish a reproducible baseline score for `sp1024`
2. Search nearby vocab sizes first: 896, 1024, 1152, 1536, 2048
3. Prefer one-parameter changes per experiment
4. Do not assume smaller vocab is better; measure it
5. Do not spend time on huge vocabularies unless cheap metrics are already clearly better

## Anti-Goals
- Do not claim a tokenizer is better based on theory alone
- Do not retrain the full dataset before cheap metrics justify it
- Do not mix tokenizer changes with model changes in this lane
