# Parameter Golf Research Program — JEPA Hybrid

## Objective
Explore byte-level JEPA-style training in parameter golf without disturbing the main record path.

## Current Approach
- Use `byte260` input data and exact byte-level BPB accounting.
- Keep a causal decoder path so evaluation stays directly compatible with the challenge metric.
- Add an auxiliary JEPA loss that predicts the latent embedding of the next byte patch from the current causal state.
- Treat this as a non-record research lane first, not an immediate leaderboard candidate.

## Why This Hybrid Exists
- A pure JEPA does not naturally emit calibrated byte probabilities, but the challenge score is BPB.
- This hybrid gives us a clean bridge:
  - representation learning signal from JEPA
  - exact likelihood-based evaluation from the causal head
- If the hybrid shows promise, we can move toward more aggressive JEPA variants later.

## Milestones
1. Confirm the byte-level trainer runs end-to-end and exports under the cap.
2. Beat a byte-level causal-only control with the same backbone.
3. Sweep the JEPA-specific knobs:
   - `JEPA_LOSS_WEIGHT`
   - `PATCH_SIZE`
   - `TARGET_EMA`
   - `JEPA_VAR_WEIGHT`
4. Only after stable gains, test more "true JEPA" directions:
   - patch decoder instead of token decoder
   - target encoder beyond plain EMA embeddings
   - masked-target or multi-future prediction

## First Experiments
1. `JEPA_LOSS_WEIGHT=0.0` control run on `train_jepa.py`
2. `JEPA_LOSS_WEIGHT=0.1`
3. `JEPA_LOSS_WEIGHT=0.2`
4. `PATCH_SIZE=16`
5. `TARGET_EMA=0.99` vs `0.995`

## Success Criteria
- Training remains stable.
- Post-quantization BPB does not regress badly relative to the byte-level control.
- Export size remains comfortably under 16MB.
- JEPA loss decreases while CE/BPB do not collapse.

## Avoid
- Comparing against the tokenizer-optimized mainline run as if this were the same regime.
- Treating a JEPA auxiliary loss win as proof that pure JEPA will work.
- Launching large expensive runs before the byte-level control and small sweeps are stable.
