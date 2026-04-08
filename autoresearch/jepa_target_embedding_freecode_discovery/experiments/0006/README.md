# Experiment 6

**Date:** 2026-04-07T14:31:12.429386+00:00
**Lane/Stage:** representation/discovery
**Result:** KEPT
**val_bpb:** 0.2403
**Artifact size:** N/A bytes
**Model params:** N/A
**Last step:** 400
**Pre-quant val_bpb:** N/A
**Quantization gap:** N/A
**Eval time:** 8835 ms
**Peak memory:** None MiB
**Gate reason:** improved_val_bpb (none -> 0.2403)
**Propose time:** 42.3s
**Train time:** 11.9s

## Change
Add weight decay (0.02) to AdamW optimizer for regularization and boost predictor learning rate to 1.5x base lr. Weight decay should improve held-out probe_score by reducing overfitting, while the higher predictor lr compensates for the regularization drag on the predictor's current→target mapping capacity.

## Diff from previous best
Identical to current best
