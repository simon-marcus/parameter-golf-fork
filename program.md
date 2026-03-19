# Parameter Golf Research Program

## Objective
Minimize validation bits-per-byte (BPB) on FineWeb, subject to:
- 16MB total artifact size (code + compressed int8+zlib weights)
- 10 minutes training on 8×H100s
- Scored on post-quantization roundtrip BPB

Current baseline: 1.2244 BPB (9-layer, 512-dim, 4 KV heads, tied embeddings)
Baseline uses ~15.8MB of 16MB budget. Step time ~43ms on 8×H100. Reaches step ~13780 in 10 min.

## Key Insight
This is a compression challenge. The question is: how much language understanding can you pack
into 16MB of int8+zlib compressed weights, trained in 10 minutes?

Three levers: (1) better architecture per parameter, (2) better training per step,
(3) better compression per parameter.

## Research Priorities

### Phase 1: Hyperparameter Tuning (low risk, quick wins)
- Learning rate sweeps for matrix_lr, scalar_lr, tied_embed_lr
- Model shape: try different depth/width tradeoffs (e.g., more layers at smaller dim)
- MLP expansion ratio
- Number of KV heads
- Sequence length (longer sequences = more context but fewer steps)
- Batch size adjustments
- Warmdown schedule tuning

### Phase 2: Architecture Modifications (medium risk, higher potential)
- Weight sharing across layers (e.g., share every other layer's weights → more effective depth for free)
- Depth recurrence: run the same set of layers multiple times per forward pass
- Different MLP activations (GELU, SiLU vs current ReLU²)
- Mixture of experts with top-k routing
- Different attention variants
- Adjust the U-Net skip connection strategy

### Phase 3: Training Tricks (medium risk)
- Sequence length warmup (start short, increase during training → more steps early)
- Gradient accumulation schedule
- Better learning rate schedule shapes
- Stochastic depth / layer dropout during training

### Phase 4: Compression (high potential, careful testing needed)
- Quantization-aware training (simulate int8 noise during training)
- Lower-bit quantization schemes (int4 with learned codebooks)
- Structured pruning before quantization
- Better compression-friendly weight distributions

## Experiment Guidelines
- Make ONE change per experiment so we can isolate effects
- Always explain the hypothesis: why should this help?
- Monitor artifact size — if you're well under 16MB, consider using the budget
- If well over 16MB, focus on making the model smaller or compressing better
- Failed experiments are valuable — note what didn't work and why
- When a hyperparameter sweep finds an optimum, combine it with the best known config
- Prioritize changes that are likely to help given what we've learned so far
