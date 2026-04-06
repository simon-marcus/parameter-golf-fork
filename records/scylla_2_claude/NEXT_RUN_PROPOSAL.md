# Next Run Proposal

## Recommendation

Use a two-stage follow-up rather than another broad blind vocab ladder.

The latest results say:

- `12288` is best end-to-end.
- `6144` is the strongest backup and may still be the best pre-TTT base.
- The likely optimum is somewhere in the `10k-14k` region.
- `12288` is benefiting unusually strongly from TTT, so that axis deserves direct tuning.

## Stage A: Resolve The Optimum Shape

Run 4 x `1xH100` in parallel:

1. `6144` repeat with a second seed
2. `12288` repeat with a second seed
3. `10240` new vocab point
4. `14336` new vocab point

### Why

- The `6144` vs `12288` gap is only about `0.0034` BPB, so it should be checked for seed stability.
- `10240` and `14336` bracket the apparent optimum much better than `8192` and `16384`.
- This stage answers the main open structural question before spending runs on TTT micro-tuning.

### Expected outcome

This should tell us whether:

- `12288` is robustly best
- `6144` is statistically tied and cheaper/faster
- the real sweet spot lies between `10240` and `14336`

## Stage B: Tune The Winning Region

After Stage A, run another 4 x `1xH100` batch centered on the best Stage A vocab.

If `12288` remains best, I would run:

1. `12288` control repeat
2. `12288` with smaller `TTT_CHUNK_BYTES`
3. `12288` with larger `TTT_CHUNK_BYTES`
4. `12288` with lower `TTT_LR`

### Suggested settings

Current control:

- `TTT_CHUNK_BYTES=131072`
- `TTT_LR=0.001`

Suggested Stage B variants:

- chunk-small: `TTT_CHUNK_BYTES=98304`
- chunk-large: `TTT_CHUNK_BYTES=196608`
- lr-low: `TTT_LR=0.0007`

### Why

- The main surprise in the data is that `12288` wins because of stronger TTT improvement, not because it is best before TTT.
- That makes TTT chunk geometry and adaptation strength the most promising immediate leverage.

## Bundle Prep Needed

Before Stage A, prepare and validate:

- `scylla_v2_v10240.tar.zst`
- `scylla_v2_v14336.tar.zst`

with matching manifests and exact shard counts.

## If We Want The Smallest Possible Next Step

If we want to minimize prep and decide fast, the minimal next batch is just:

1. `6144` repeat
2. `12288` repeat

on 2 x `1xH100`.

That is the fastest way to check whether the current winner is robust before spending time building `10240` and `14336`.

## Bottom Line

My preferred next sequence is:

1. Build `10240` and `14336`
2. Run Stage A on `6144`, `10240`, `12288`, `14336`
3. Pick the winner
4. Run a 4-way TTT sweep at that winning vocab

That is the highest-information path from the current results.
