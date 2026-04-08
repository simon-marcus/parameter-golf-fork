# Experiment 10

**Date:** 2026-04-07T14:35:49.288019+00:00
**Lane/Stage:** representation/discovery
**Result:** KEPT
**val_bpb:** 0.2393
**Artifact size:** N/A bytes
**Model params:** N/A
**Last step:** 400
**Pre-quant val_bpb:** N/A
**Quantization gap:** N/A
**Eval time:** 6602 ms
**Peak memory:** None MiB
**Gate reason:** improved_val_bpb (0.24028241 -> 0.2393)
**Propose time:** 84.9s
**Train time:** 9.4s

## Change
Rebalance loss weights from (0.3 MSE, 0.35 infonce, 0.35 online_infonce) to (0.5 MSE, 0.35 infonce, 0.15 online_infonce). The probe_score metric is dominated by MSE and retrieval against the EMA target, but 35% of training loss was going to online_infonce which aligns with the online target encoder — not directly measured by the score. Shifting weight from online_infonce to MSE should better align training gradients with the evaluation metric, improving both MSE and retrieval. | retry: Rebalance loss weights from (0.5 MSE, 0.25 infonce, 0.25 online_infonce) to (0.5 MSE, 0.35 infonce, 0.15 online_infonce). The probe_score metric is dominated by MSE and retrieval against the EMA target, but 25% of training loss was going to online_infonce which aligns with the online target encoder — not directly measured by the score. Shifting weight from online_infonce to infonce should better align training gradients with the evaluation metric, improving both MSE and retrieval accuracy.

## Diff from previous best
Identical to current best
