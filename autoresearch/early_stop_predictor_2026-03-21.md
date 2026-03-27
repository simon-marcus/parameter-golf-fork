# Early-Stop Predictor, 2026-03-21

## Goal

Estimate the final `600s` `val_bpb` of an `8xH100` run from an earlier stopping point such as `120000ms`.

## Data

- Source pool:
  - `origin/main` record logs
  - fetched PR refs for `#64, #168, #179, #180, #198, #205, #208, #232, #236, #244, #264, #272, #274, #286, #294, #305, #329`
  - current worktree logs
- Dedupe method:
  - SHA1 of full log text
- Inclusion rule:
  - only logs with `stopping_early: wallclock_cap`
  - `train_time` in `[540000, 660000]`
  - parseable final `val_bpb`
- Resulting dataset:
  - `60` unique `600s` runs
  - `37` directory-level groups

## Important Split

There are two materially different regimes.

`mainstream`
- ordinary training/eval runs
- includes sliding-window and TTT
- excludes the strongest "paid bits" / eval-data artifacts:
  - paid prefix
  - correction table
  - val-only/data-access runs

`all`
- everything, including those eval/data artifacts

This split matters because early training traces do not contain enough information to predict large eval-only gains from paid prefix / correction-table / train-on-val style methods.

## Model

Default predictor:
- `ExtraTreesRegressor`
- grouped cross-validation by directory, not by individual seed
- features available by cutoff:
  - `last_step`
  - `last_step_avg_ms`
  - `steps_per_sec`
  - `train_loss_last`
  - `train_loss_min`
  - `train_loss_mean_last5`
  - `train_loss_delta_last5`
  - `train_loss_slope_last5`
  - `val_bpb_last`
  - `val_bpb_delta_last3`
  - `model_params`
  - `train_batch_tokens`
  - `train_seq_len`
  - `embed_lr`
  - `matrix_lr`
  - `scalar_lr`
  - `flag_ttt`
  - `flag_sliding`

95% intervals:
- conformal-style empirical interval
- half-width = 95th percentile of grouped CV absolute residuals

## Results

### All Runs

These include paid-prefix / val-only / correction-table style logs.

| Cutoff | Runs | Groups | MAE | RMSE | 95% half-width |
|---|---:|---:|---:|---:|---:|
| `60s` | 60 | 37 | `0.0513` | `0.0897` | `0.2233` |
| `120s` | 60 | 37 | `0.0401` | `0.0698` | `0.1894` |
| `180s` | 60 | 37 | `0.0368` | `0.0673` | `0.1886` |
| `300s` | 60 | 37 | `0.0356` | `0.0680` | `0.1894` |

Interpretation:
- this is too wide to be useful for frontier decisions
- the main reason is eval-side artifacts that are not inferable from early training trajectory alone

### Mainstream Runs

These exclude paid-prefix / val-only / correction-table logs.

| Cutoff | Runs | Groups | MAE | RMSE | 95% half-width | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| `60s` | 52 | 33 | `0.0152` | `0.0185` | `0.0308` | `0.942` |
| `120s` | 52 | 33 | `0.0146` | `0.0179` | `0.0324` | `0.942` |
| `180s` | 52 | 33 | `0.0137` | `0.0172` | `0.0326` | `0.942` |
| `300s` | 52 | 33 | `0.0153` | `0.0187` | `0.0361` | `0.942` |

Interpretation:
- `120s` is already about as informative as `300s`
- for mainstream runs, a practical estimate is:
  - `predicted_final_bpb ± 0.032` at `120s`
- this is good enough to kill obviously bad directions
- it is not good enough to resolve tiny deltas like `0.003 BPB`

## Practical Recommendation

Use the predictor in two modes:

1. `mainstream` for ordinary recipe exploration.
   - This is the useful regime.
   - A `120s` screen is worthwhile.

2. `all` only as a sanity check.
   - Expect wide intervals.
   - Do not rely on it for paid-prefix / correction-table / val-data methods.

Decision rule for mainstream screens:
- if the `120s` prediction is worse than your current best by more than about `0.04 BPB`, kill it
- if it is better by less than about `0.01 BPB`, do not trust the proxy alone
- if it is better by `0.03+ BPB`, it is a real promotion candidate for `8xH100`

## Caveats

- The predictor estimates the final logged `val_bpb`, not submission validity.
- Size and eval-time failures are separate constraints and are not modeled here.
- Very novel eval-side techniques can break the mapping from early training trace to final score.
- Grouped CV by directory is intentionally conservative; seed-only validation would look better but would be misleading.

## Usage

Summary:

```bash
uv run --with numpy --with scikit-learn \
  python autoresearch/early_stop_predictor.py summary
```

Predict a run from its first `120s`:

```bash
uv run --with numpy --with scikit-learn \
  python autoresearch/early_stop_predictor.py predict \
  --log records/track_10min_16mb/2026-03-20_LeaderCore10L_ValidEval_TempOnly_Int8Search/runpod_tmp_base/train.log \
  --cutoff-ms 120000 \
  --mode mainstream
```
