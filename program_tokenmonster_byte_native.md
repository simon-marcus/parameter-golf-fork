# Parameter Golf TokenMonster Byte-Native Program

## Objective
Discover a TokenMonster-derived tokenizer path that is valid for competition use:

1. it must preserve the exact FineWeb validation byte stream
2. it must then be judged on tokenizer efficiency and eventual model quality

This lane is not allowed to ignore exactness problems in exchange for a prettier cheap score.

## Research Priority
The old Scylla / `tm0054` path failed because it relied on a normalized UTF-8 TokenMonster vocabulary.
This lane searches only for byte-native TokenMonster variants that can survive exactness audits.

## Allowed Experiment Types
1. `byte_native_build`
   - derive a YAML-backed TokenMonster variant from an existing base vocab
   - use `charset:none`
   - use `normalization:none`
   - optionally delete specific raw-token patterns before auditing

2. `byte_native_eval`
   - audit an existing YAML-backed byte-native candidate

## Hard Gate
No candidate is allowed onto the frontier unless it passes the exactness gate first.

Required checks:
- sample exactness audit on validation docs
- low or zero byte drift
- no obvious normalization regression

If exactness is poor, the candidate is a rejection no matter how attractive the token stats look.

## Search Heuristics
- start from the best prior TokenMonster bases:
  - `tm0054`
  - `tm0045`
  - `english-1024-clean-v1`
  - `english-1024-balanced-v1`
  - `english-1024-consistent-v1`
- prefer byte-native YAML variants over `.vocab` persistence
- treat this as a repair loop, not a creative tokenizer-design loop
- act on the least-bad exactness frontier first
- carry forward strong negative evidence:
  - if a class of edits explodes to ~200 bad docs, stop repeating it
- prefer narrow local edits over global mode changes
- after single-token repairs plateau, allow at most two tightly related safe deletions in one experiment
- when the remaining failures are non-ASCII, prefer literal high-byte crumbs from the failing docs before touching likely UTF-8 lead-byte coverage
- look for raw-token pathologies around:
  - capcode + byte-token interactions
  - quote / dash punctuation cases
  - space-deletion modifier interactions
  - specific multi-character tokens implicated in the remaining bad docs
- if a family shows the same exactness failure repeatedly, move away from it

## Anti-Goals
- do not claim success on a tokenizer that fails byte-exactness
- do not rely on `candidate.meta.npz` tricks to paper over text drift
- do not treat converted normalized vocabs as safe unless the audits actually pass
- do not keep retrying broad capcode toggles once they are proven catastrophic
- do not delete broad token classes like all uppercase letters, spaces, or all capcode markers unless the audit evidence specifically justifies it
