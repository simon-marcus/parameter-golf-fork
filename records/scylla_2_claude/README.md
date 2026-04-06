# Scylla-v2 Byte-Native TTT Sweep

## Problem

Scylla-v2 (1125-token byte-native TokenMonster) trains strongly and quantizes well, but the current score-first TTT regime only improves BPB from ~1.1343 to ~1.1315 on 8xH100 — far weaker than the SP1024 legal-TTT path (1.1218 → 1.1194).

Mild retuning on 1xH100 smoke (TTT_CHUNK_TOKENS=34304, TTT_LR=0.0015) beat the SP smoke baseline, but did not transfer to 8x.

## Root Cause Analysis

The TTT recipe is implicitly tuned to SP1024 token geometry. Key mismatches for byte-native:

1. **Chunk byte coverage**: A 32768-token chunk covers ~45KB for Scylla vs ~120KB for SP1024. Each TTT update sees less than half the semantic context.

2. **Update density**: With thin chunks and 3 epochs per chunk, Scylla overfits to local byte-level n-gram statistics rather than learning distributional adaptation.

3. **Embedding adaptation**: Byte embeddings represent raw bytes (0x00-0xFF), not semantic subwords. Adapting them during TTT adds noise with no document-specific signal.

4. **Early layer disruption**: Blocks 0-5 compose bytes into meaning (learned tokenizer). Adapting them during TTT is like changing the tokenizer mid-evaluation.

## Code Changes

`train_gpt_legal_ttt.py` modifications from the 2026-03-31 Scylla-v2 8x baseline:

- **New `TTT_FREEZE_EMBEDDINGS` env var**: When set to 1, freezes all non-block/non-bank parameters during TTT: `tok_emb`, `lm_head`, `bigram`, `smear`, `ve_shared`, `ve_layer_scales`, `final_norm`, `skip_weights`, `mtp_heads`.

- **Bank-aware block freezing**: Instead of only freezing `blocks.N.*` parameters, the freeze logic now creates gradient masks for bank parameters (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`) that zero out gradients for frozen block slices. This is correct because banks are contiguous 3D tensors indexed by block — the forward pass needs all slices, but gradients should only flow through unfrozen blocks.

## Experimental Matrix (Phase 1: 1xH100 Smoke)

| Run | Chunk Tokens | LR     | Epochs | Freeze Blocks | Freeze Emb | Hypothesis |
|-----|-------------|--------|--------|---------------|------------|------------|
| A1  | 65536       | 0.0015 | 3      | 2             | 0          | Byte-equivalent chunk size (2x current) |
| A2  | 65536       | 0.001  | 2      | 2             | 0          | Richer chunks, fewer epochs |
| A3  | 65536       | 0.001  | 2      | 6             | 0          | Late-layer-only adaptation |
| A4  | 65536       | 0.001  | 2      | 6             | 1          | + frozen embeddings (key hypothesis) |
| A5  | 96000       | 0.001  | 2      | 6             | 1          | Even larger chunks |
| A6  | 34304       | 0.0015 | 3      | 2             | 0          | Current best (control) |

Run order: A6 first (control), then A1-A5. A6 uses the known-best smoke settings for comparison.

## How to Run

### On a 1xH100 pod:

```bash
# Sync this directory to the pod
rsync -avz -e "ssh -p PORT -o StrictHostKeyChecking=no" \
  records/scylla_2_claude/ root@HOST:/workspace/parameter-golf/records/scylla_2_claude/

# Run all configs sequentially
ssh root@HOST -p PORT "cd /workspace/parameter-golf && \
  USE_COMPILE=0 NPROC_PER_NODE=1 MAX_WALLCLOCK_SECONDS=180 \
  bash records/scylla_2_claude/launch_sweep.sh ALL"

# Or run a single config
ssh root@HOST -p PORT "cd /workspace/parameter-golf && \
  USE_COMPILE=0 NPROC_PER_NODE=1 MAX_WALLCLOCK_SECONDS=180 \
  bash records/scylla_2_claude/launch_sweep.sh A4"
```

### Analyze results:

```bash
# Pull results
rsync -avz -e "ssh -p PORT -o StrictHostKeyChecking=no" \
  root@HOST:/workspace/parameter-golf/records/scylla_2_claude/runs/ \
  records/scylla_2_claude/runs/

# Compare
python3 records/scylla_2_claude/analyze_sweep.py
```

## Phase 2: 8xH100 Promotion

Pick the top 2 configs from Phase 1 by `legal_ttt_exact` BPB. Run on 8xH100 with full 10-minute budget:

```bash
DATA_ROOT_MODE=tmp NPROC_PER_NODE=8 MAX_WALLCLOCK_SECONDS=600 \
  bash records/scylla_2_claude/launch_sweep.sh A4
```

## Phase 3: Narrow Autoresearch (conditional)

If Phase 2 identifies a stable winning region, build a config-search autoresearch loop:
- Mutation space: `{LR: [0.0005, 0.0008, 0.001, 0.0012], epochs: [1, 2, 3]}`
- Fixed: chunk size and freeze config from Phase 2 winner
- Objective: minimize `legal_ttt_exact` BPB
