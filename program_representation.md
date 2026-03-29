# Parameter Golf Representation Program

## Objective
Figure out whether tokenizer and representation changes can buy materially better BPB-per-byte
than our current `sp1024` setup under the actual challenge constraints:
- 16MB total artifact
- 10 minutes on 8xH100s
- tokenizer-agnostic BPB on FineWeb validation
- legality-first evaluation

## Why This Lane Exists
We moved too quickly into eval-time tricks. The current judge guidance favors slowing down,
understanding the dataset, and exploring train-time and representation ideas that can survive review.

The representation question is not "can we retokenize FineWeb?" It is:

1. Does FineWeb structure suggest a smaller or different vocabulary would save enough params/FLOPs to matter?
2. Which byte-heavy structures are poorly served by the current tokenizer?
3. If we changed tokenization, what model budget would that free?
4. Are there clear format-heavy or repetition-heavy regimes that justify specialized modeling?

## Research Questions

### Priority 1: Measure the current tokenizer honestly
1. What are the train/val bytes-per-token under `sp1024`?
2. Which token classes dominate byte mass?
3. How repetitive is validation under exact context matching?
4. Which token pieces look obviously bad: URLs, markup, numbers, boilerplate fragments?

### Priority 2: Translate observations into budget decisions
5. If vocab shrinks or grows, how many embedding/head params move?
6. Is the likely compute/artifact win large enough to justify retokenization work?

### Priority 3: Promote only evidence-backed hypotheses
7. Only propose tokenizer retraining if the profiler says the current tokenizer is leaving real money on the table.
8. Only propose architecture changes if they directly exploit measured FineWeb structure.

## Near-Term Outputs
- `data/profile_fineweb.py`: structural/token-class/repetition profile
- `data/tokenizer_tradeoff_report.py`: vocab-budget math
- `autoresearch_representation.py`: uses the above outputs to generate ranked next experiments

## Anti-Goals
- Do not chase eval-time loopholes in this lane.
- Do not assume bigger vocab or smaller vocab is automatically better.
- Do not retokenize the full corpus before a cheap profiler says it is worth doing.
