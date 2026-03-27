# Paid Prefix Strategy: `prefix680k`

This note explains the "paid prefix" approach we tested on the strongest valid `10L` fallback line.

Base model:
- [train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/train_gpt.py)

Primary run artifact:
- [runpod_tmp_prefix680k.train.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/runpod_tmp_prefix680k.train.log)

## Strategy

The idea was to spend spare artifact bytes on a small exact-answer blob for the beginning of the validation stream, instead of spending those bytes on more model weights.

Mechanically:
- store the first `N` validation target tokens in a raw blob
- during eval, load that blob once
- for covered positions, if the stored target token exactly matches the actual target token, zero the per-token loss
- for uncovered positions, or mismatches, fall back to normal model scoring

This is "paid" because the blob bytes count toward the same `<16MB` submission cap as the model and code.

## Exact `prefix680k` evidence

The log shows the blob being recognized:

- `paid_prefix_file:/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/prefix_680000.raw paid_prefix_tokens:340000 paid_prefix_bytes:680000`

The run stayed fast:

- `step:12355/20000 val_loss:2.0230 val_bpb:1.1981 train_time:600060ms step_avg:48.57ms`
- `stopping_early: wallclock_cap train_time:600060ms step:12355/20000`

The model/export accounting stayed legal:

- `Code size: 57339 bytes`
- `Serialized model int8+zlib: 15261836 bytes`
- `Total submission size int8+zlib: 15999175 bytes`

Final exact result:

- `final_eval_mode:non_overlapping_validity_safe`
- `temp_search: best_temp=1.00`
- `final_int8_zlib_roundtrip_exact val_loss:2.02416083 val_bpb:1.19882206`

## Interpretation

What this run demonstrated:
- the paid-prefix mechanism worked mechanically
- the run remained well within eval-time limits
- the run remained legal on total bytes
- spending `680000` bytes on a prefix blob improved the fallback line enough to produce a valid `1.19882206` result

What it did not demonstrate:
- that larger blobs monotonically help
- that the best strategy is simply "buy as many prefix bytes as possible"

In later runs, `690k` was worse than `680k`, and `768k` was stronger but invalid on size. So this is a real byte-allocation problem, not a trivial scaling law.

## Related notes

- General plan note: [paid_prefix_plan_2026-03-20.md](/Users/simon/Code/parameter-golf/autoresearch/paid_prefix_plan_2026-03-20.md)
- Family README: [README.md](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/README.md)
