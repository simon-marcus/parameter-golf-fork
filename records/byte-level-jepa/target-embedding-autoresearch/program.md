# JEPA Target Embedding Research Program

You are improving the JEPA target embedding mechanism through the focused probe in [target_embedding_probe.py](/Users/simon/Code/parameter-golf/records/byte-level-jepa/target-embedding-autoresearch/target_embedding_probe.py).

## Objective
Improve the held-out JEPA target/predictor proxy score by changing only the **target embedding / target patch representation** machinery.

## Core Hypothesis
The current JEPA target is too weak because it mean-pools EMA byte embeddings across the next patch, losing:
- byte order
- local compositional structure
- richer patch semantics

Better target embeddings may let JEPA teach a stronger predictive representation.

## Hard Constraints
- Do not change the tokenizer or input alphabet.
- Do not broaden this into full LM training.
- Do not turn this into a patch-first whole-model rewrite.
- Do not add eval-time tricks or storage tricks.
- Do not make broad unrelated optimizer changes.

## Preferred Mutation Surface
Prefer changes in this order:
1. target patch aggregation method
2. local positional encoding inside target patches
3. tiny target patch encoder
4. tiny paired predictor improvement if needed

## Good Candidate Ideas
- mean pooled byte embeddings + learned local position offsets
- tiny MLP reducer over per-byte target embeddings
- tiny causal conv over bytes inside the target patch
- tiny intra-patch attention reducer
- EMA target patch encoder with very small parameter count
- lightly normalized or projected target latents before JEPA loss

## Avoid
- changing unrelated hyperparameters just to search noise
- large parameter increases
- edits outside `TargetPatchEncoder` and, if truly necessary, `Predictor`
- changing many concepts at once

## Proposal Rules
- Make exactly one conceptual change per experiment.
- Keep edits surgical.
- Favor small target-embedding changes over bigger rewrites.
- If a proposal needs multiple code edits, they must all support the same single conceptual change.

## What Good Looks Like
The best proposals should plausibly improve the JEPA target representation without muddying attribution.

We are looking for:
- stronger local semantics in the target patch embedding
- better held-out predictor/target alignment
- better retrieval and anti-collapse behavior
- a better chance that promoted variants will later help `JEPA on` beat the matched `JEPA off` control in the real byte-level lane
