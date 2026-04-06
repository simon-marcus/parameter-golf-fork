# Legality Memo

Date: 2026-03-27

## Summary

As of March 27, 2026, the competition's effective legality standard has narrowed
substantially.

The two major changes are:

- two-pass eval methods are being treated as leakage
- hashed n-gram / cache methods are being treated as invalid unless they produce
  a properly normalized probability distribution

Primary sources:

- Illegal submissions megathread, judge bulk-closure comment:
  https://github.com/openai/parameter-golf/issues/677#issuecomment-4145868416
- Judge explanation of the normalization issue:
  https://github.com/openai/parameter-golf/issues/677#issuecomment-4144203157
- Discussion of "distribution doesn't sum to 1" / cleanup RFC:
  https://github.com/openai/parameter-golf/pull/886

## Ruled Illegal

These appear clearly disallowed now.

### Two-pass eval submissions

- Judge closure rationale on our PRs:
  "Two-pass submissions like these leak eval tokens, since on the second pass
  you're evaling tokens you've trained on in the first."
- This directly kills the core logic behind our `#870` and `#881`.
- Source context:
  https://github.com/openai/parameter-golf/issues/677#issuecomment-4145868416

### Hashed n-gram caches whose probabilities are not renormalized into a true distribution

- The judge explanation is explicit: if
  `sum_i P_ngram(token_i | previous tokens)` is not `1`, the BPB is incorrect.
- The same comment says hashing/collisions create correlations that require
  extra renormalization.
- Source:
  https://github.com/openai/parameter-golf/issues/677#issuecomment-4144203157

### Methods that look at the target token to decide blending or available experts

- The judge comment on `#982` says valid systems must build a fixed distribution
  from previous tokens only, then score the actual next token under that
  distribution.
- The closure summary quoted in discussion also says these methods "look ahead
  to the target token to mix probabilities and therefore leak eval tokens."
- Related discussion:
  https://github.com/openai/parameter-golf/pull/886

### Illegal TTT and illegal GPTQ

- The bulk-closure comment explicitly says many PRs were closed for "illegal
  TTT and GPTQ."
- Source:
  https://github.com/openai/parameter-golf/issues/677#issuecomment-4145868416

### Specific PR families effectively invalidated

- The bulk-closure comment explicitly lists `#870`, `#881`, `#913`, `#944`,
  `#988`, and many others as closed/reviewed in that sweep.
- Source:
  https://github.com/openai/parameter-golf/issues/677#issuecomment-4145868416

## Ruled Legal, Or At Least Explicitly Described As Potentially Legal

This is the most important positive statement currently on record.

### Score-first TTT is explicitly legal in principle

- The README says:
  "you are only allowed to test-time train on validation set tokens you've
  already evaluated your model on, since those tokens have already been
  graded!"
- `@0hq` also clarified that evaluation is over the concatenated token stream,
  and that:
  "You're allowed to use any preceding tokens from the evaluation set that
  you've already been tested on in any way you'd like."
- `@valerio-oai` explicitly said:
  "TTT is a valid approach in theory, but very easy to unintentionally leak val
  data into."
- A merged record example exists:
  `#549` ("Legal Score-First TTT"), which `@valerio-oai` approved as
  "Looks legal."
- Sources:
  - README:
    https://raw.githubusercontent.com/openai/parameter-golf/main/README.md
  - `#402` / `@0hq`:
    https://github.com/openai/parameter-golf/issues/402#issuecomment-4112325938
  - `#402` / `@valerio-oai`:
    https://github.com/openai/parameter-golf/issues/402#issuecomment-4119313230
  - `#549`:
    https://github.com/openai/parameter-golf/pull/549
  - `#549` approval:
    https://github.com/openai/parameter-golf/pull/549#issuecomment-4118263181

### Pre-quant TTT is not specifically banned

- I have not found a maintainer comment or README rule saying that doing TTT on
  full-precision / pre-quantization weights is illegal as a category.
- The enforced distinction has been causal ordering:
  - legal: score-first / backward-looking TTT
  - illegal: adapt on validation tokens, then report scores after that
    adaptation
- The strongest example is `#1006`. That PR explicitly used pre-quant AdamW
  TTT, but the compliance objection was adapt-then-score, not "pre-quant."
- The author of `#1006` explicitly conceded that the reported score used the
  non-compliant adapt-then-score path.
- So the best current reading is:
  pre-quant TTT appears legal if and only if it still obeys the same
  score-first causality rule as any other TTT.
- Sources:
  - `#1006` discussion:
    https://github.com/openai/parameter-golf/pull/1006
  - comment flagging adapt-then-score:
    https://github.com/openai/parameter-golf/pull/1006#issuecomment-4149164167
  - author concession:
    https://github.com/openai/parameter-golf/pull/1006#issuecomment-4149293113

### N-grams are not banned in principle

- The judge explanation says "n-grams should be perfectly legal if":
  - any cache crossing the train/eval boundary is counted toward artifact size
  - any eval-time-built cache respects causality
  - BPB is computed correctly, which requires the distribution to sum to `1`
- Source:
  https://github.com/openai/parameter-golf/issues/677#issuecomment-4144203157

So the currently safest eval-time interpretation is:

- exact or auditable causal cache/memory is potentially legal
- but only if it defines a proper normalized next-token distribution from past
  tokens only

## Not Yet Clearly Adjudicated

These areas are still uncertain.

### `#933`

- It is still open, which means it has not been explicitly closed in the same
  way as `#870`, `#881`, `#913`, `#944`, etc.
- But that is not the same as being approved.
- Given the clarified rules, `#933` still looks exposed because it is both:
  - two-pass
  - hash-cache based
- So the best current classification is: not yet adjudicated, but vulnerable.
- PR:
  https://github.com/openai/parameter-golf/pull/933
- Megathread:
  https://github.com/openai/parameter-golf/issues/677

### Exact non-hashed causal caches

- The judge explanation strongly suggests these could be valid if normalized and
  causal.
- There is not yet a merged exemplar that settles the standard operationally.

### Artifact-passed train-time cache structures

- In principle, the judge comment allows train-to-eval caches if they count
  against artifact size.
- But the cache still has to define a correct normalized distribution.
- So "artifact memory" is not itself illegal; the problem has been the
  incorrect probabilistic construction.

### Pre-quant TTT still has a separate budget risk

- Even if pre-quant TTT is causal, the surrounding export path can still become
  illegal if it sneaks in extra post-training compute or data use.
- The clearest maintainer warning is around GPTQ / calibration after the 600s
  training cap, especially if training data is re-opened during export/eval.
- So the practical standard is:
  - pre-quant TTT itself is a causality question
  - calibration / export / artifact-construction steps are a separate budget
    and data-access question
- Source:
  https://github.com/openai/parameter-golf/issues/677#issuecomment-4123231934

### A second leaderboard

- The judge comment says they are still considering one.
- That matters because some invalidated ideas may become useful again in another
  category.
- Source:
  https://github.com/openai/parameter-golf/issues/677#issuecomment-4145868416

## Practical Reading Of The Rules Now

Safest high-confidence moves:

- train-time improvements
- strict causality
- fixed distribution built from previous tokens only
- exact normalization before likelihood is accumulated
- no second-pass reuse of eval tokens
- if using TTT, keep it score-first and make the logged path obviously so
- treat pre-quant TTT as acceptable only if the export/calibration path is also
  clean under the 600s + eval rules

High-risk moves:

- two-pass rescoring
- hashed cache distributions without exact renormalization
- target-conditioned mixing
- anything that relies on bucket collisions to "work"
- any TTT path that adapts on tokens before scoring them
- any post-training GPTQ / calibration / export stage that reuses training data
  or quietly extends the training budget

## Implication For Us

- `#870` and `#881` are out for the main leaderboard.
- `CREAM`-style two-pass cache work is probably not a viable final submission
  path either.
- A legally safer path now is:
  - stronger train-time model work
  - or exact causal online memory with explicit normalization
