# Record candidate: CaseOps + Matrix-LR 0.028 + Phased TTT 3500

**val_bpb: 1.06109** (3-seed mean, std 0.00130) | **15.90 MB max** | 8xH100 SXM | 600s train | score-first TTT eval

This is a small final-push stack on the modern CaseOps/LQER/SmearGate line: keep the strong #1855 architecture intact, nudge `MATRIX_LR` from `0.026` to `0.028`, and spend more of the still-available eval budget by raising `PHASED_TTT_PREFIX_DOCS` to `3500`. Not a philosophical reinvention; more like finding one last clean turn of the screw before the clock hits zero.

## 3-Seed Results

| Seed | Steps | Pre-quant BPB | Quantized BPB | TTT BPB | Artifact bytes | Eval time |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4,994 | 1.06350701 | 1.07204922 | **1.05925746** | 15,896,241 | 410.7s |
| 1337 | 4,987 | 1.06629011 | 1.07475931 | **1.06193899** | 15,897,721 | 407.4s |
| 2026 | 4,989 | 1.06659353 | 1.07490342 | **1.06208636** | 15,893,558 | 460.7s |
| **Mean** | **4,990** | **1.06546355** | **1.07390398** | **1.06109427** | **15,895,840** | **426.3s** |

Std over the three TTT BPBs is `0.00130`. The best seed reached `1.05925746` BPB with the same legal TTT path.

## Key Techniques

1. CaseOps SP8192 tokenizer and byte-sidecar path, using the lossless caps reserved tokenizer.
2. 11-layer 512d XSA stack with U-Net skips, parallel decoder, depth recurrence, SparseAttnGate, BOS-fixed SmearGate, and LeakyReLU(0.5)^2 MLP.
3. Polar-Express Newton-Schulz Muon plus the tuned quant/compression stack: GPTQ int6 matrices, int7 embeddings, int8 row gate, LQER asymmetric rank-4 correction, and per-group `lrzip` + brotli compression.
4. This submission's final deltas: `MATRIX_LR=0.028` and `PHASED_TTT_PREFIX_DOCS=3500`.
5. Score-first phased TTT stays on the post-quant model and scores every chunk before any update.

## Compliance / Legality

- [x] Training is capped at 600s on 8xH100; logs show `599.55s` to `599.64s`.
- [x] Eval including TTT is under 600s; observed range `407.4s` to `460.7s`.
- [x] All artifacts are under 16,000,000 bytes; max observed `15,897,721`.
- [x] TTT is score-first and single-pass: each chunk is evaluated before adaptation and not rescored.
- [x] No validation tokens are used for training or pre-quant adaptation.
- [x] No SLOT.
- [x] No n-gram cache and no logit bias.
- [x] No ETLB.

## Credits

This stands directly on #1855's CaseOps/LQER/SmearGate stack and keeps its main architecture intact. Credits also to the lineage called out there: #1797, #1787, #1736, #1729, #1667, #1626, #1610, #1586, #1530, #1344, #493, #478, #315, and #289. The incremental contribution here is the final Matrix-LR / TTT-prefix validation and packaging.

## Reproduction

```bash
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
VOCAB_SIZE=8192 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
TTT_ENABLED=1 \
PHASED_TTT_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=3500 \
PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 \
MATRIX_LR=0.028 \
MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 \
ATTN_CLIP_SIGMAS=13.0 \
EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 \
TTT_CHUNK_SIZE=48 \
WARMUP_STEPS=20 \
MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 \
WARMDOWN_FRAC=0.85 \
BETA2=0.99 \
TTT_BETA2=0.99 \
TTT_WEIGHT_DECAY=0.5 \
TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=0.5 \
GPTQ_CALIBRATION_BATCHES=16 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=50 \
GATED_ATTN_QUANT_GATE=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 \
LQER_ASYM_ENABLED=1 \
LQER_RANK=4 \
LQER_FACTOR_BITS=4 \
LQER_ASYM_GROUP=64 \
LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 \
COMPRESSOR=pergroup \
NCCL_NET=Socket \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Test Plan

- 3-seed validation on 8xH100 SXM: seeds `42`, `1337`, `2026`.
- Confirmed all train runs stop under the 600s wallclock cap.
- Confirmed all TTT eval runs finish under the 600s eval budget.
- Confirmed all compressed artifacts are below 16 MB.
- Confirmed score-first TTT ordering and no SLOT / n-gram cache / ETLB.
