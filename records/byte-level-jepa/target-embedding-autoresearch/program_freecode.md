# JEPA Target Embedding Research Program (Freecode Mode)

You are improving the JEPA target-embedding prediction mechanism through the focused probe in [target_embedding_probe.py](/Users/simon/Code/parameter-golf/records/byte-level-jepa/target-embedding-autoresearch/target_embedding_probe.py).

## Objective
Improve the held-out JEPA target/predictor proxy score (`val_bpb` in this probe) by writing code-level changes that strengthen target-embedding prediction quality, while keeping this as a cheap probe loop.

## What Is Different In This Mode
You are allowed a broader coding surface than constrained mode. You may edit:
- `TargetPatchEncoder`
- `Predictor`
- target/predictor loss shaping for this probe
- patch-pair construction and lightweight probe diagnostics
- tiny supporting changes in `ProbeModel` if needed

You are NOT limited to only one class, but changes must still be coherent and attributable.

## Hard Constraints
- Do not change tokenizer or alphabet (`VOCAB_SIZE=260` remains byte-level).
- Do not convert this into full LM training.
- Keep runtime in the same order of magnitude as current probe runs.
- Do not add external dependencies.
- Do not edit files outside this probe for the proposal.

## Design Guardrails
- Prefer one coherent mechanism per experiment (even if it touches multiple nearby functions).
- Keep parameter growth modest; avoid large architecture bloat.
- Preserve compatibility with existing log parsing (`final_int8_zlib_roundtrip_exact ... val_bpb:` line must still appear).
- Keep the data contract unchanged (`fineweb_train_*.bin`, `fineweb_val_*.bin`).

## Suggested High-Value Directions
- Better intra-patch structure modeling for target embeddings.
- Better predictor-target alignment geometry (normalization/projection/contrastive variants).
- Better anti-collapse behavior with minimal side effects.
- Better retrieval behavior without destabilizing the score.

## Avoid
- unrelated broad optimizer sweeps
- touching random hyperparameters for noise
- multi-concept grab-bags in one proposal
- code churn that hurts interpretation

## What Good Looks Like
Strong proposals in this mode should:
- open up new behavior not reachable through tiny constrained edits
- still be small enough to evaluate quickly and compare cleanly
- produce stable measurable movement in probe score and diagnostics
