# Representation Discovery — Memo 005

## 1. Verdict

**A tokenizer change is not justified now.** The profiler data shows the current `sp1024` tokenizer is reasonably efficient for FineWeb's structure, and the parameter budget freed by vocab changes is too small to move the needle compared to architecture/training improvements in the core lane.

## 2. What the Current Profile Implies About FineWeb Structure

**FineWeb validation is overwhelmingly alphabetic English text.** Alpha tokens carry 94.2% of byte mass at 2.57 bytes/token — decent compression. Only 861 of 1024 vocab entries are observed in the 200K-token val sample, meaning ~160 tokens are dead weight, but at 512 bytes each (d_model=512), that's only ~80KB — noise against a 16MB budget.

**Repetition is moderate, not extreme.** At context length 2, 78% of positions have a repeat and the top-1 next-token probability is 0.51 — meaning bigram statistics are useful but far from deterministic. By context 4, only 19% of positions repeat, and by 8 it's 3%. This is not a dataset where n-gram caching or repetition modeling will dominate. *Inference*: FineWeb is diverse enough that the model genuinely needs to learn language structure, not just memorize patterns.

**Numeric and byte-fallback tokens are costly per byte.** Numerics average 1.0 bytes/token (26.5K tokens, 1.1% of bytes) — each digit burns a full prediction slot. Byte-fallback tokens (3.2K tokens, 0.14% of bytes) are similarly 1.0 bytes/token. Combined, these are ~1.2% of byte mass but consume ~3% of token slots. This is a real but small inefficiency.

## 3. What It Implies About Tokenizer Strategy

**Vocab shrinkage (512–768) saves 131K–262K params.** At int8 quantization that's 128–256KB freed in the artifact. Against a 16MB budget this is 1.6% at best. To justify retokenization, the freed params would need to buy more BPB improvement than whatever core-lane experiments could do with those same params — unlikely given diminishing returns on embeddings vs. transformer layers.

**Vocab growth (1536–2048) costs 262K–524K params** for better compression. The bytes/token ratio would improve (fewer predictions needed), but the extra embedding cost is substantial. At 2048 vocab, embeddings alone are 1MB int8 — that's 6.25% of the artifact budget on embeddings alone. *Inference*: growth past 1536 is probably net-negative under the 16MB constraint.

**The sweet spot is likely near where we already are (1024).** The 861/1024 utilization rate suggests mild over-provisioning but not enough to act on. A hypothetical 896-token vocab would save ~65K params (~64KB) — not worth the retokenization cost.

**The big vocab row (28,416) is informational only.** It would consume 14.5MB on embeddings alone, leaving ~1.5MB for the entire transformer. Dead on arrival.

## 4. Ranked Next Experiments

1. **Measure actual BPB contribution by token class.** The profiler shows byte mass, but not how much cross-entropy each class contributes. Numeric tokens at 1.0 bytes/token may be easy to predict (digits in years, prices follow patterns) or hard. Run the current best model on a tagged val sample and attribute loss by class. *This is the missing data point for any tokenizer decision.*

2. **Dead-token pruning feasibility check.** 163 unused vocab entries × 512 dim = 83K params. Quantify: if we zero those rows and reclaim the artifact space for more transformer layers, what's the param-equivalent gain? This is a code-only change (mask unused rows before export), no retokenization needed.

3. **Byte-fallback token analysis.** The 96 unique byte-fallback tokens handling 3.2K occurrences — what bytes are these? If they're mostly UTF-8 continuation bytes from non-English text, this is irrelevant noise. If they're ASCII characters that should have been merged, there's a tokenizer bug.

4. **Sequence-length impact of current tokenizer.** At 2.41 bytes/token, a 1024-context model sees ~2,467 bytes of context. Compute: what context length does the model actually train at, and is the bytes-per-token ratio creating a bottleneck vs. what the model could learn with longer effective context?

5. **Bigram-aware embedding initialization.** The repetition profile shows bigram context is informative (entropy drops from 5.0 to 1.9 bits). *Inference*: initializing embeddings with bigram statistics (rather than random) could accelerate early training. This is a cheap train_gpt.py change, no tokenizer rebuild needed.

## 5. What Not to Do Yet

- **Do not retrain a tokenizer.** The data doesn't justify it. The savings are marginal and the infrastructure cost (rebuild shards, validate, retrain model from scratch) is high.
- **Do not increase vocab size.** Every extra token costs 512 bytes in the artifact. The compression gains don't pay for themselves under 16MB.
- **Do not build an n-gram cache for eval-time.** The repetition profile shows this tops out at ~3% coverage at context-8. The program explicitly says no eval-time loopholes in this lane.
- **Do not chase the byte-fallback tokens** without first checking experiment 3 above. They're 0.14% of byte mass — optimizing them is noise unless they're symptomatic of a larger issue.
- **Do not propose architecture changes** until experiment 1 (loss attribution by token class) provides evidence that specific token classes are disproportionately driving val_bpb.
