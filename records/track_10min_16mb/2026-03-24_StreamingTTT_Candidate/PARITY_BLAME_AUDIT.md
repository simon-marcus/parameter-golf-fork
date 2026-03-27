# March 23 Parity / Blame Audit

Scope:
- Leader baseline: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
- Current baseline: `chunk_full` full 8x run from `runpod_artifacts/2026-03-25_chunk_full_8x/train.log`

Goal:
- Explain the remaining gap between March 23 leader seed `1337` and current `chunk_full`
- Separate mismatches into:
  - Bucket A: train-quality controls
  - Bucket B: export-path controls
  - Bucket C: legal-TTT controls

Observed gap:

| Metric | Leader | Current | Gap |
|---|---:|---:|---:|
| `post_ema` bpb | `1.1369` | `1.1372` | `+0.0003` |
| `final_int6_roundtrip_exact` bpb | `1.14518023` | `1.14532575` | `+0.00015` |
| `final_int6_sliding_window_exact` bpb | `1.12165091` | `1.12173272` | `+0.00008` |
| `legal_ttt_exact` bpb | `1.11922988` | `1.11955351` | `+0.00032` |

## Blame Table

| Bucket | Mismatch | Leader value | Current value | Source location or log evidence | Plausibility |
|---|---|---|---|---|---|
| A | Effective train length under the wallclock cap | Stops at `step:7179/9000` before EMA | Stops at `step:7181/9000` before EMA | Leader [train_seed1337.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log#L66), [train_seed1337.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log#L70); current [train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-25_chunk_full_8x/train.log#L65), [train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-25_chunk_full_8x/train.log#L69); same wallclock-based warmdown logic in [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1660) and [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1693) | `~3e-4` |
| A | Activation control surface simplified | Configurable activation knobs exist, but default path is still `leaky_relu_sq` | Slimmed code hard-wires `leaky_relu_sq`, log confirms `activation_mode:leaky_relu_sq neg_slope:0.5` | Leader activation config and wiring at [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L94), [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L743), [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1538); current hard-wired path at [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L742), [current train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-25_chunk_full_8x/train.log#L16) | `negligible` |
| A | TTT default surface changed in code, but not used during training | `TTT_FREEZE_BLOCKS` exists as an eval-time control | Replaced by `TTT_PARAM_MODE`, `TTT_BLOCK_START`, `TTT_BLOCK_END`, `TTT_LR_SCHEDULE`; training path does not consume them | Leader [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L102); current [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L99) | `negligible` |
| B | Int6 row calibration, clip search, and scaling path | `quantize_int6_per_row()` tries `0.9990..1.0`, uses `scale=row_clip/31`, `clamp_min(1/31)`, `float16` | Same | Leader [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1298); current [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1337) | `negligible` |
| B | Mixed export policy and compressor | `{"mlp","attn"}` use int6, others int8 or passthrough, `lzma preset=6` | Same | Leader [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1842), [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1855), [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1859); current [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1875), [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1888), [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1892) | `negligible` |
| B | Exported eval-model reconstruction surface | Rebuild passes explicit activation knobs into exported eval model | Slimmed code removes those knobs and rebuild only passes `activation_neg_slope`, which still implies `leaky_relu_sq` | Leader [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1878); current [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1911) | `negligible` |
| B | Observed export drift after identical codec path | `roundtrip=1.14518023`, `sliding=1.12165091`, `int6+lzma=15887928` | `roundtrip=1.14532575`, `sliding=1.12173272`, `int6+lzma=15771716` | Leader [train_seed1337.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log#L73), [train_seed1337.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log#L76), [train_seed1337.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log#L78); current [train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-25_chunk_full_8x/train.log#L72), [train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-25_chunk_full_8x/train.log#L75), [train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-25_chunk_full_8x/train.log#L77) | `~1e-4` |
| C | Legal-TTT chunk size, epochs, and LR schedule | `chunk_tokens=32768`, `ttt_lr=0.002`, `ttt_epochs=3`, inline chunk-cosine | Same effective values, explicit `TTT_LR_SCHEDULE=chunk_cosine` | Leader [train_seed1337.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log#L80), [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1237); current [train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-25_chunk_full_8x/train.log#L80), [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1160) | `negligible` |
| C | Trainable-param surface during TTT | `freeze_blocks=0`, `unfrozen=26928220`, `frozen=0` | `param_mode=full`, `unfrozen=26928220`, `frozen=0` | Leader [train_seed1337.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log#L80), [train_seed1337.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log#L81); current [train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-25_chunk_full_8x/train.log#L79), [train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-25_chunk_full_8x/train.log#L80) | `negligible` |
| C | TTT adaptation mode bit | Explicit `base_model.train()` before chunk adaptation | Candidate had been staying in `eval()` during TTT SGD until the March 25 parity fix | Leader [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1199), [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1234); fixed current [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1245), [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1278) | `~1e-4` |
| A | In-training validation cadence under the wallclock cap | Default `VAL_LOSS_EVERY=4000`, visible `step:0` and `step:4000` validation pauses in the leader log | Runpod launcher had been forcing `VAL_LOSS_EVERY=0` until the March 25 parity fix | Leader default [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L37), leader log [train_seed1337.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log#L38), [train_seed1337.log](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log#L57); fixed launcher [launch_streaming_ttt_candidate_runpod.sh](/Users/simon/Code/parameter-golf/launch_streaming_ttt_candidate_runpod.sh#L32) | `~1e-4` |
| C | Logging and metric bookkeeping around TTT | `ttt_sliding:start`, `ttt_sliding:done` | `ttt:start`, `ttt:done`, plus `scored_tokens` and `adapted_tokens` | Leader [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1157), [leader train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py#L1283); current [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1230), [current train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-24_StreamingTTT_Candidate/train_gpt.py#L1319) | `negligible` |

## Conclusions

1. Bucket A is almost clean, but not perfectly clean.
   The highest-value train-side mismatches are now:
   - wallclock-step drift under the 10-minute cap
   - in-training validation cadence under that same cap

   Before the March 25 parity fix, the Runpod launcher was removing the leader's periodic validation pauses by forcing `VAL_LOSS_EVERY=0`.
   - leader stops at `7179`
   - current `chunk_full` stops at `7181`
   - despite two extra steps, current `post_ema` is still worse by about `3e-4`

2. Bucket B appears structurally clean.
   The quantizer, clip search, mixed int6/int8 policy, and `lzma` settings match. The observed roundtrip/sliding drift is small and consistent with slightly different incoming weights rather than a changed codec path.

3. Bucket C is closer to clean than before, but not fully proven yet.
   Chunk size, epochs, LR shape, and effective full-parameter adaptation match the leader behavior. But until March 25, the candidate's chunked legal TTT had also been adapting in `eval()` mode rather than `train()` mode, which changes fake-quant behavior during TTT.

4. The current evidence does not justify a broad quantization pivot as the next run package.
   The best-supported reading is:
   - some small train-quality drift exists before export
   - export/TTT add only smaller secondary deltas

## Derived Run Package

Run 1:
- Exact parity-cleaned `chunk_full` baseline
- Objective: remove any hidden drift and see whether `post_ema` moves first

Run 2:
- Same as Run 1 plus one tiny train-quality adjustment most likely to affect `post_ema`
- Best candidates:
  - a small LR nudge
  - a small late-QAT threshold shift
  - a small SWA timing tweak

Run 3:
- Only if Bucket A looks clean or stalls
- Same as best prior run plus matched export-aware QAT for the current exact int6 exporter

## Stop Condition

If `post_ema` does not improve, do not spend the next 8x cycle on broader quantization or new TTT modes.
