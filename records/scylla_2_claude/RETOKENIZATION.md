# Scylla Retokenization Notes

This folder now has a staged Scylla vocab ladder beyond `4096`, with validated bundle archives for:

- `6144`
- `8192`
- `12288`
- `16384`

## What Was Done

For each target vocab, I:

1. Built a resized TokenMonster base vocab.
2. Converted it into a byte-native Scylla YAML tokenizer with:
   - `charset none`
   - `normalization none`
   - `capcode 0`
   - missing byte tokens added explicitly
3. Retokenized the local FineWeb export from the existing `fineweb10B_sp1024` source bundle.
4. Verified the resulting dataset and tokenizer with `verify_runpod_data_ready.sh`.
5. Ran a strict byte audit with `data/audit_tokenmonster_bundle.py`.
6. Archived the dataset + tokenizer into `.tar.zst` bundles and uploaded them to S3.

This is the data-prep step needed before running the next Scylla vocab sweep on GPUs.

## Why

The earlier Scylla ladder (`1536 -> 2048 -> 3072 -> 4096`) improved monotonically and `4096` became the best current Scylla base. That made higher vocab the obvious next axis to test, but each larger vocab needs its own retokenized FineWeb bundle. The tokenizer files are small; the retokenized dataset archive is the real artifact.

## Important Detail

The larger TokenMonster base families do not map to runtime vocab the same way as the smaller bases. For the larger families used here, runtime vocab came out as:

- `runtime_vocab = resize + 142`

The working builds used:

- `6144`: base `english-8000-clean-v1`, resize `6002`
- `8192`: base `english-16000-clean-v1`, resize `8050`
- `12288`: base `english-16000-clean-v1`, resize `12146`
- `16384`: base `english-32000-clean-v1`, resize `16242`

These were verified against actual tokenizer runtime vocab before retokenizing the full corpus.

## Validation Contract

All uploaded bundles passed:

- exact shard/tokenizer preflight
- `bad_docs: 0`
- `meta_overcount_frac: 0.0`
- `decoded_drift_frac: 0.0`

That means the exported token metadata is byte-exact against the original source docs and safe to use for Scylla training runs.

## Current S3 Artifacts

Bucket prefix:

- `s3://parameter-golf-staging-094651608775/data/archives/`

Objects:

- `scylla_v2_v6144.tar.zst`
- `scylla_v2_v6144.manifest.tsv`
- `scylla_v2_v8192.tar.zst`
- `scylla_v2_v8192.manifest.tsv`
- `scylla_v2_v12288.tar.zst`
- `scylla_v2_v12288.manifest.tsv`
- `scylla_v2_v16384.tar.zst`
- `scylla_v2_v16384.manifest.tsv`

## Launch Reminder

Do not assume a fixed train shard count across vocab sizes. Read the manifest for each bundle:

- `6144`: `train_shards=9`, `val_shards=1`
- `8192`: `train_shards=9`, `val_shards=1`
- `12288`: `train_shards=8`, `val_shards=1`
- `16384`: `train_shards=8`, `val_shards=1`

These counts should be passed through to the RunPod launcher instead of hardcoding the old ladder values.
