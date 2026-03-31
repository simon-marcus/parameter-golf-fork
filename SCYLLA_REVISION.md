# Scylla Revision

## Issue

The current Scylla / `tm0054` submission path has two separate correctness problems:

1. `candidate.meta.npz` does not compute validation bytes correctly for TokenMonster token streams.
   The current exporter treats each token's standalone decoded UTF-8 length as `base_bytes`, with all-zero
   `has_leading_space` and `is_boundary_token`. That overcounts bytes for modifier-token sequences.

2. More importantly, the current TokenMonster bundle is not byte-identical to the fixed FineWeb validation text.
   The winning `tm0054` vocabulary reports `normalization = "NFD"`, so the retokenized validation stream can differ
   from the original source bytes even when decoded with TokenMonster's own decoder.

## Why This Matters

The competition metric is tokenizer-agnostic `val_bpb` on the fixed FineWeb validation set.
Changing token boundaries is allowed. Changing the underlying validation byte stream is not.

So a custom tokenizer path must satisfy both of these conditions:

- exact denominator: score must divide by the true source-byte count of the fixed validation corpus
- exact text preservation: the tokenized validation stream must decode back to the same byte stream

The current Scylla bundle fails both checks.

## Immediate Plan

1. Add a first-class audit step for TokenMonster bundles:
   - compare source validation bytes
   - compare metadata-derived bytes
   - compare exact TokenMonster-decoded bytes

2. Make bundle generation fail fast for normalizing TokenMonster vocabularies.
   A vocabulary with `normalization != None` is unsafe for exact competition submission unless there is a separate,
   proven exact byte reconstruction path.

3. Reassess Scylla after running the audit on the full validation shard.

## Expected Outcome

This revision is likely to show that the current `tm0054` submission cannot be defended as-is.
If so, the tokenizer line remains promising as research, but a leaderboard-valid revision will require
an exact byte-preserving tokenizer path rather than the current normalized TokenMonster bundle.

## Exploratory Result

Two "cheap salvage" ideas were tested and rejected:

- rewriting the existing `tm0054` vocabulary header to `normalization: none`
- rewriting it further to `charset: none` and `normalization: none`

These variants improve some cases, but they do not make the existing vocabulary byte-exact on validation text.
In particular, sample roundtrips still fail on:

- some plain-ASCII documents with +/- 1 byte drift
- non-ASCII cases that fall through to TokenMonster's byte-token path

So the likely repair is not a header tweak. It is a new byte-native TokenMonster vocabulary path,
trained or constructed for exact byte preservation from the start.

## Current Stage

The dedicated byte-native discovery lane has now mapped the obvious local repair space for converted
`1024`-scale TokenMonster bases:

- the least-bad region is `capcode=2`
- `english-1024-clean-v1` is the current best invalid frontier
- current best exactness is still:
  - `187 / 200` sampled validation docs exact
  - `13` bad docs
  - `byte_drift = -95`

Important negative result:

- switching between `clean`, `balanced`, and `consistent` under `capcode=2` does not improve exactness
- narrow single-token deletions like `ing`, `.C`, ` mo`, ` po`, ` -`, `ng`, ` re` also do not improve exactness
- narrow two-token deletions also plateau at the same `13 / -95` boundary

So the current converted-family search is no longer bottlenecked on broad strategy. It is bottlenecked on
understanding the exact remaining failure mechanisms.

## Plan For This Stage

The next stage is document-first, not token-first:

1. inspect and cluster all 13 remaining bad docs from the least-bad frontier
2. identify whether they fall into a small number of byte-level mechanisms
   - capcode marker stripping
   - multi-byte UTF-8 fallback corruption
   - punctuation / boundary token interactions
3. only after clustering, design the next mutation space around those concrete mechanisms

If the 13 docs all reduce to one or two structural bugs, the next lane should target those bugs directly.
If they do not, that is evidence that converted normalized TokenMonster families are fundamentally the wrong substrate,
and we should shift effort to a from-scratch byte-native vocabulary build instead.

## First Failure Clustering Result

Using the current least-bad frontier (`english-1024-clean-v1`, `capcode=2`), all 13 remaining bad docs
fall into the same broad bucket:

- `13 / 13` are `non_ascii_or_utf8`

Representative failures:

- `c2 b7` decodes as `c2 20` in docs `9` and `18`
- `c3 b8` decodes as `c3 79` in doc `19`
- multi-byte CJK text is heavily truncated in doc `80`
- `c3 ad` decodes as `c3 74` in doc `90`
- `c3 af` decodes as `c3 76` in doc `119`
- `c3 bc` decodes as `c3 74` in doc `140`

This is important because it means the lane has moved past generic token deletion.
The remaining problem is specifically a byte-native / UTF-8 fallback interaction, likely involving
high-byte tokens under `capcode=2`, not a general ASCII boundary issue.

## Current Deletion-Only Findings

After pivoting the lane toward the `13` non-ASCII failures, we manually screened the remaining literal
high-byte suspects taken directly from the failing docs:

- `\x84` -> `186 / 200` exact, `14` bad docs, `byte_drift = -97`
- `¥` -> `187 / 200` exact, `13` bad docs, `byte_drift = -97`
- `¦` -> `187 / 200` exact, `13` bad docs, `byte_drift = -97`
- `©` -> `176 / 200` exact, `24` bad docs, `byte_drift = -113`
- `°` -> `184 / 200` exact, `16` bad docs, `byte_drift = -101`
- `Ã` -> `184 / 200` exact, `16` bad docs, `byte_drift = -116`

Paired local deletions do not help either:

- `¥ + ¦` -> `187 / 200` exact, `13` bad docs, `byte_drift = -99`
- `¥ + \x84` -> `186 / 200` exact, `14` bad docs, `byte_drift = -99`
- `¦ + \x84` -> `186 / 200` exact, `14` bad docs, `byte_drift = -99`
- `¥ + °` -> `184 / 200` exact, `16` bad docs, `byte_drift = -103`

Earlier high-byte deletions such as `Â`, `â`, and `\x94` were materially worse and are now treated as
explicitly harmful.

## Current Conclusion

Deletion-only search has saturated.

The remaining failures are not removable junk tokens that disappear cleanly under one or two targeted
deletions. Some deletions are catastrophic (`Â`, `â`, `\x94`), some are neutral plateaus (`Å`, `¥`, `¦`),
and the rest make exactness worse without fixing the bad-doc cluster.

So the next repair phase should not spend more budget on literal deletions alone. The remaining mismatch
looks structural in the byte-token / UTF-8 handling path, which means the next useful mutation space is
likely one of:

1. a from-scratch byte-native TokenMonster vocabulary build, or
2. a richer edit space than deletion alone, such as deliberately replacing or reintroducing exact byte
   fallback coverage
