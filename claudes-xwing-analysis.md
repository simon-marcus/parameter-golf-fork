# PR #800 Analysis: X-WING — The Breakthrough and What's Next

### The Core Innovation

**X-WING is not a training innovation — it's an eval-time compute technique.** The neural model itself trains to **~1.12 BPB** (similar to other top entries like #782). The massive jump to **0.5644 BPB** happens entirely at evaluation time by combining the model with n-gram statistics built from the validation data.

### How It Works (Mechanically)

**1. Standard training** → produces a ~1.12 BPB transformer (11L, 512d, GPTQ int6, etc.)

**2. At eval time**, a sliding-window pass over the validation set:
- For each position, compute `p_model` from the neural model's softmax
- Look up n-gram statistics from hash tables (orders 2-7) using backoff
- N-gram probability: `p_ngram = full_count / ctx_count` from the highest matching order
- Blend: **`p_final = (1 - α) × p_model + α × p_ngram`**

**3. Score-first compliance**: Tokens are scored *before* they update the n-gram tables. This is backward-looking, not oracle access.

**4. The key fix over previous work (PR #782)**: All 8 GPU ranks share **identical** n-gram tables. Previously each rank built tables from 1/8 of the data. Shared tables give 8× more statistical context. This single change closed a **0.37 BPB gap** (0.936 → 0.564).

**5. Cubric**: Per-order adaptive scaling — tracks which n-gram orders outperform the model and boosts/suppresses them. Orders 5-7 get boosted to 1.88-2.0×, while noisy orders 2-3 get suppressed to 0.30-0.45×.

**6. Entropy-adaptive alpha**: When the model is confident (low entropy), trust the model more (small α). When uncertain (high entropy), trust n-grams more (large α, up to 0.70). Uses a sigmoid mapping from entropy to α.

### Class of Change

This is a **non-parametric eval-time ensemble** — combining a parametric neural model with a non-parametric n-gram model built online from the test distribution. Closely related to:

| Approach | Description | Relation to X-WING |
|----------|-------------|---------------------|
| **KN/Modified KN n-gram LMs** | Classic smoothed n-gram models | X-WING uses raw counts + hash tables instead of proper smoothing |
| **Cache LMs** (Grave et al. 2017) | Boost recently-seen tokens | X-WING is a generalization — full n-gram context, not just unigram cache |
| **kNN-LM** (Khandelwal et al. 2020) | Retrieval-augmented LM via embedding similarity | Higher quality but much more expensive than hash tables |
| **Test-Time Training (TTT)** | Gradient updates to model at eval time | Parametric adaptation vs X-WING's non-parametric approach |
| **PPM/CTW** | Compression algorithms with multi-order context mixing | More principled probability estimation than ad-hoc backoff |

### What X-WING Leaves on the Table

**A. Blending sophistication (biggest opportunity)**

The linear blend `(1-α)p_model + αp_ngram` is the simplest possible combination. Alternatives:

1. **Log-linear / product-of-experts**: `p ∝ p_model^(1-α) × p_ngram^α` — this is theoretically better for combining independent evidence sources
2. **Per-token learned gating**: The blend weight could be a function of the model's hidden state, not just scalar entropy → a 2-layer MLP on the last hidden state could learn when to trust n-grams
3. **Multi-order mixture**: Instead of backoff (use highest available order), blend ALL available orders simultaneously with learned/adaptive weights
4. **Bayesian model averaging**: Use the validation data to compute posterior weights over model vs n-gram

**B. N-gram probability estimation**

- Raw `count/context_count` is MLE with no smoothing. **Kneser-Ney smoothing** would give much better probability estimates, especially for rare contexts
- Hash collisions degrade accuracy — 8M buckets for 62M tokens with orders 2-7 means significant collision rates. **Perfect hashing** or **count-min sketch** could help
- **Continuation counts** (how many distinct contexts a word appears in) are ignored — this is what makes KN smoothing so powerful

**C. Higher orders / longer context**

- They stop at order 7. **Orders 8-10** would capture longer patterns, especially for formulaic text
- The bottleneck is hash table size and collision rates — more buckets or a tiered approach could support higher orders

**D. Cubric is ad-hoc**

- The ±3% / ±5% thresholds with 0.3-2.0 clamps are hand-tuned heuristics
- **Online learning** (e.g., exponential weights / Hedge algorithm) for per-order blend weights would be more principled and adaptive
- **Bayesian online learning** with proper regret bounds

**E. The neural model itself is standard**

- Training achieves ~1.12 BPB. A better base model directly improves the ensemble.
- Their architecture includes some tricks we should examine: **leaky_relu_sq** (leaky ReLU squared activation with slope 0.5), **XSA** (cross-sequence attention on last 4 layers), **GPTQ** quantization (better than our int8)
- **The n-gram technique is fully orthogonal to training improvements** — any training BPB improvement we make translates directly to the final blended score

**F. Eval time budget**

- Training: 600s, eval: ~225s. They have `NGRAM_EVAL_MAX_SECONDS=300`
- If eval could be made faster (GPU-accelerated n-gram lookup instead of numpy), more sophisticated techniques become feasible
- **Multiple passes** over the validation data could improve table quality

### Most Promising Alternative Approaches (Same Class)

**1. PPM (Prediction by Partial Matching)**
- A well-studied compression algorithm that naturally handles multi-order context mixing
- Achieves optimal regret bounds for stationary sources
- Would replace the ad-hoc backoff + Cubric with a theoretically grounded mixture
- Could be implemented efficiently in numpy/torch

**2. CTW (Context Tree Weighting)**
- Bayesian model averaging over all context depths simultaneously
- No hyperparameters for blending — the math handles it
- Proven optimal for tree sources; competitive in practice
- More expensive per-token but potentially much better probability estimates

**3. kNN-LM hybrid**
- Store (context_embedding, next_token) pairs during eval
- At each position, find k-nearest neighbors by embedding similarity
- Much richer context matching than exact n-gram prefix matching
- Could run on GPU for speed

**4. N-gram + TTT hybrid**
- Use n-gram statistics to identify which tokens are "easy" (high n-gram confidence)
- Spend TTT gradient budget only on "hard" tokens where the model struggles
- Best of both worlds: cheap non-parametric correction + expensive parametric adaptation

**5. Ensemble of diverse small models**
- Instead of one model + n-grams, train 2-3 diverse small models that fit in 16MB together
- Ensemble their predictions at eval time
- Diversity could come from different architectures, different training data orders, different tokenizers

### Bottom Line for Us

The strategic move is clear:

1. **Implement n-gram eval** — this is the single highest-ROI change available. Even a basic implementation should cut our BPB by ~40-50%
2. **Use proper smoothing** (Kneser-Ney or CTW) instead of raw counts — likely worth 0.01-0.03 BPB over their approach
3. **Keep improving training** — every 0.01 BPB training improvement translates to the final score
4. **Explore PPM/CTW as the blending framework** — more principled than the Cubric heuristic
5. **GPTQ for quantization** — if we're not already doing this, it frees artifact bytes for more parameters

The n-gram approach is essentially "free lunch" eval-time compute — it's building a compression model from the test distribution and ensembling it with the neural model. The question isn't whether to adopt it, but how to do it better than X-WING.