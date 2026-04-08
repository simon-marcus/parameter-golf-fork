# Experiment 11

**Date:** 2026-04-07T14:37:23.590775+00:00
**Lane/Stage:** representation/discovery
**Result:** KEPT
**val_bpb:** 0.2375
**Artifact size:** N/A bytes
**Model params:** N/A
**Last step:** 400
**Pre-quant val_bpb:** N/A
**Quantization gap:** N/A
**Eval time:** 7018 ms
**Peak memory:** None MiB
**Gate reason:** improved_val_bpb (0.23925176 -> 0.2375)
**Propose time:** 48.3s
**Train time:** 9.9s

## Change
Upgrade Predictor from GELU MLP blocks to SwiGLU gated blocks (gate*up→down with SiLU activation). Each block now uses multiplicative gating (SiLU(gate(x)) * up(x) → down(x)) instead of GELU(fc1(x)) → fc2(x). This adds one extra linear per block (3 vs 2 projections) for modestly more parameters but richer nonlinear capacity, matching the target encoder's SwiGLU style. The multiplicative gating should improve the predictor's ability to learn the current→target mapping, benefiting both MSE and retrieval.

## Diff from previous best
Identical to current best
