# Parameter Golf TokenMonster Program

## Objective
Search only within the small-vocab TokenMonster regime that proxy validation supports.

Current evidence says:
- `english-1024-balanced-v1` is the strongest current frontier
- `english-1024-clean-v1` and `english-1024-consistent-v1` are close alternatives
- `english-2048-clean-v1` is weaker and mostly a control
- SentencePiece is no longer the active search frontier

## Allowed Experiment Types
1. `tokenmonster_eval`
   - evaluate one known TokenMonster vocabulary reference
2. `tokenmonster_build`
   - create a local TokenMonster `.vocab` candidate by pruning/editing a base TokenMonster vocabulary

## Priorities
1. Stay near the winning `1024` TokenMonster family
2. Favor simpler, more atomic vocabularies
3. Prefer pruning/removing awkward composite tokens over increasing vocab size
4. Keep candidates easy to proxy-validate later
5. Avoid getting trapped in tiny resize-only local search once progress stalls

## Candidate-Building Heuristics
- prune very long decoded tokens
- prune obviously composite multiword tokens
- prune tokens that mix leading space with punctuation-heavy fragments
- optionally resize down modestly after pruning
- try different base vocabularies within the proven `1024` family, not only `balanced-v1`
- try qualitatively different pruning rules, not only resize changes
- when stuck, prefer bolder structure edits over another `1014-1020` micro-step

## Stagnation Policy
- If several consecutive experiments fail to improve the frontier, switch out of local hill-climbing mode
- Do not keep proposing near-duplicate resize sweeps in the same narrow band
- When stagnating, prioritize:
  - changing base vocab (`balanced`, `clean`, `consistent`)
  - changing pruning families
  - testing more atomic vocab edits
  - testing controls with no resize but selective deletions

## Anti-Goals
- do not return to broad SentencePiece exploration
- do not chase larger vocabularies without direct proxy evidence
- do not claim cheap tokenizer metrics are the final judge
