# Progress Notes

## 2026-04-06: High-Vocab Scylla Ladder

The full vocab ladder is now:

- `1536`: `1.54825555`
- `2048`: `1.54296076`
- `3072`: `1.53764772`
- `4096`: `1.52635984`
- `6144`: `1.50132960`
- `8192`: `1.51102757`
- `12288`: `1.49793940`
- `16384`: `1.51062312`

The curve is no longer monotonic. Scylla clearly wants a much larger vocab than `4096`, but the current optimum is interior rather than "bigger is always better."

### Main conclusions

- The `1536 -> 4096` trend was real and continued strongly to `6144`.
- `12288` is the current best point.
- `8192` and `16384` are both regressions relative to `6144` and `12288`.
- The gain from `4096` to `12288` is large: about `0.02842` BPB.
- The gain from `6144` to `12288` is small but likely real: about `0.00339` BPB.

### Important detail

The win is not coming purely from the base model.

From the pulled logs:

- `6144` final sliding-window exact: `1.51197674`
- `6144` final legal TTT exact: `1.50132960`
- `12288` final sliding-window exact: `1.51374972`
- `12288` final legal TTT exact: `1.49793940`

So `6144` is slightly better before TTT, but `12288` gets a larger TTT lift and wins overall. That means `12288` is the best current end-to-end stack under the present byte-aware TTT setup, not simply the best pre-TTT base model.

### Training-budget effect

At the 600s cap, completed steps drop as vocab rises:

- `1536`: `779`
- `4096`: `753`
- `6144`: `753`
- `8192`: `739`
- `12288`: `722`
- `16384`: `700`

This suggests part of the `16384` regression is likely undertraining and larger softmax/embed overhead rather than a purely tokenizer-quality failure.

### Size trend

Compressed submission size rises steadily:

- `4096`: `10.81 MB`
- `6144`: `11.44 MB`
- `8192`: `11.86 MB`
- `12288`: `12.80 MB`
- `16384`: `13.76 MB`

So the regressions at `8192` and `16384` are not primarily cap failures. They look more like a quality/training/compression tradeoff issue.

### Current read

- `12288` is the new best Scylla baseline.
- `6144` is the strongest simpler fallback.
- `8192` is not the sweet spot in the current recipe.
- `16384` is likely too far for the present 10-minute training recipe unless the training/eval stack is retuned around it.
