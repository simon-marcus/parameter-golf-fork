# Parameter Golf — Experiment Tracker

## Goal
Beat the baseline **1.2244 BPB** (post-int8-quantization) on FineWeb validation, within 16MB / 10 min on 8×H100.

## Current Status
- **Infrastructure**: 1×H100 SXM on RunPod ($2.69/hr), autoresearch loop running
- **Baseline (1-GPU, 3 min)**: 1.6419 BPB — our iteration benchmark
- **Baseline (8-GPU, 10 min)**: 1.2244 BPB — official submission benchmark
- **Best so far**: 1.6201 BPB (experiment #2, deeper/narrower 11L×448d)

## Experiment Log

| # | Change | BPB | vs Best | Size (bytes) | Kept? | Notes |
|---|--------|-----|---------|--------------|-------|-------|
| 0 | Baseline (9L, 512d, 4KV, tied emb) | 1.6419 | — | 8,899,669 | ✓ | Official config, 334 steps in 3 min |
| 1 | SiLU activation instead of ReLU² | 1.6968 | +0.055 | 8,835,877 | ✗ | Worse — ReLU² seems better for this regime |
| 2 | 11 layers × 448 dim (vs 9×512) | 1.6201 | -0.022 | 8,769,774 | ✓ | Deeper/narrower helps! New best. |
| 3+ | (autoresearch running with Opus) | — | — | — | — | — |

## Ideas to Try

### High Priority (likely to help)
- [ ] Learning rate sweep (matrix_lr, scalar_lr, tied_embed_lr)
- [x] More layers at same param count (depth vs width tradeoff) → **helped!** (exp #2)
- [ ] Sequence length changes (shorter = more steps in time budget)
- [ ] Weight sharing / layer tying (effective depth for free)
- [ ] MLP expansion ratio tuning

### Medium Priority
- [ ] Different attention configs (more/fewer heads, different KV head counts)
- [ ] Warmdown schedule tuning
- [ ] Batch size adjustments
- [ ] Gradient clipping
- [ ] Logit softcap value

### Speculative / High Risk
- [ ] Depth recurrence (run same layers multiple times)
- [ ] Mixture of experts
- [ ] Quantization-aware training
- [ ] Sequence length warmup (start short → go long)
- [ ] Alternative optimizers or Muon hyperparameters

## What We've Learned
- ReLU² outperforms SiLU in this small-model/short-training regime (exp #1)
- Deeper/narrower (11×448) beats shallower/wider (9×512) at similar param count (exp #2)
- On 1×H100, step time is ~540ms (vs ~44ms on 8×H100), so we get ~334 steps in 3 min
- Artifact size is well under 16MB budget (~8.8MB) — room to grow the model
- Cycle time with Claude Code: ~6 min per experiment (~10 experiments/hour)
- Now using Opus with high effort for smarter proposals

## Infrastructure Notes
- Pod: RunPod secure cloud, 1×H100 SXM 80GB, pod ID `dfmh6pu2s39twj`
- SSH: `ssh root@216.243.220.216 -p 15584`
- Auth: Claude Max subscription (OAuth)
- Autoresearch output: `tail -f /workspace/parameter-golf/autoresearch.out`
- History: `/workspace/parameter-golf/autoresearch/history.jsonl`
- Logs: `/workspace/parameter-golf/autoresearch/logs/`
- Experiment snapshots: `/workspace/parameter-golf/autoresearch/experiments/NNNN/`
