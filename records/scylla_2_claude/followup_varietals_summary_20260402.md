# Follow-Up Varietals Summary

Date: 2026-04-02

## Bases

Standard base:

- `sp_xsa_all`
- Stage 1 score: `1.35526573`

Scylla base:

- `scylla_mlp35_warmdown900`
- Stage 1 score: `1.37660171`

## Pulled Data

Local pulled follow-up logs:

- `records/scylla_2_claude/pulled_followup_runs/xq0m5tbqvn786r/std_adapter/`
- `records/scylla_2_claude/pulled_followup_runs/fwsk8itvz65n4l/std_causal/`
  Note: the earlier stop interrupted the first pull attempt, so the detailed
  per-run logs were summarized from remote inspection and are not fully present
  locally.
- `records/scylla_2_claude/pulled_followup_runs/xwxi1bgdojj8lh/scylla_adapter/`
- `records/scylla_2_claude/pulled_followup_runs/mplu4wieiey7w0/scylla_causal/`

## Standard Adapter TTT

Base: `sp_xsa_all`

- `top4_r32`: `1.35669510`
- `top4_r64`: `1.35571459`
- `all_r32`: `1.35714084`
- `all_r64`: `1.35572511`

Best:

- `top4_r64`: `1.35571459`

Delta vs base:

- `+0.00044886` BPB worse

Conclusion:

- Adapterized TTT on the standard base did not beat the best Stage 1 standard run.

## Standard Legal Causal Cache

Base: `sp_xsa_all`

- `o24_light`: `1.38459763`
- `o24_strong`: `1.40857779`
- `o34_light`: `1.37919631`
- `o34_conservative`: `1.36923587`

Best:

- `o34_conservative`: `1.36923587`

Delta vs base:

- `+0.01397014` BPB worse

Conclusion:

- Exact causal cache was substantially worse than the standard base.

## Scylla Adapter TTT

Base: `scylla_mlp35_warmdown900`

- `top4_r32`: `1.37614388`
- `top4_r64`: `1.37620588`
- `all_r32`: `1.37746769`
- `all_r64`: `1.37761870`

Best:

- `top4_r32`: `1.37614388`

Delta vs base:

- `-0.00045783` BPB better

Conclusion:

- Adapterized TTT gave a very small improvement on the best Scylla base.
- The gain is real but small on this 1xH100 harness.

## Scylla Legal Causal Cache

Base: `scylla_mlp35_warmdown900`

- `o24_light`: `1.40053481`
- `o24_strong`: `1.42370239`
- `o34_light`: `1.39688203`
- `o34_conservative`: `1.38714083`

Best:

- `o34_conservative`: `1.38714083`

Delta vs base:

- `+0.01053912` BPB worse

Conclusion:

- Exact causal cache was clearly worse than the Scylla base.

## Overall Read

Ranking of best follow-up results:

1. Standard base, no follow-up: `1.35526573`
2. Standard adapter TTT best: `1.35571459`
3. Scylla adapter TTT best: `1.37614388`
4. Scylla base, no follow-up: `1.37660171`
5. Standard causal cache best: `1.36923587`
6. Scylla causal cache best: `1.38714083`

Main takeaways:

- The strongest result remains the plain standard-tokenizer base from Stage 1.
- Adapterized TTT is at least plausibly useful on Scylla, but only as a small-gain refinement.
- The exact legal causal-cache variants did not pay off on either base.

## Pod Cleanup

Stopped after pull completion:

- `mplu4wieiey7w0`
- `xwxi1bgdojj8lh`

Stopped earlier after completed result capture:

- `fwsk8itvz65n4l`
- `xq0m5tbqvn786r`
