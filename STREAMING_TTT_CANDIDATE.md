# Streaming TTT Candidate

Goal: turn the March 23, 2026 legal-TTT record into a stronger legal online-adaptation candidate, aligned with the OpenAI clarification that type-2 token-stream TTT is valid.

## Core Claim

The March 23 record uses a conservative legal protocol:

1. score a whole chunk under `torch.inference_mode()`
2. train on that already-scored chunk
3. move to the next chunk

This is legal, but stricter than necessary. Under the published clarification, a stronger legal target is:

1. score the newly revealed suffix of the current window
2. immediately adapt on exactly that already-scored suffix
3. advance by `stride`

No future-token leakage is allowed into either:

- the metric accumulation
- the loss that drives each optimizer step
- the ordering of windows/documents

## Legality Invariant

At evaluation step `k`, the update may depend only on losses for tokens whose contribution has already been added to the official metric.

Equivalent implementation rule:

- `optimizer.step()` may only use a loss mask over the suffix that was just accumulated into `loss_sum` / `byte_count`

Invalid patterns:

- backprop on the full window when only the suffix is already scored
- using future tokens from the current chunk/window in the adaptation loss
- reordering validation documents or windows

## Candidate Algorithms

### A. Token-Stream Streaming TTT

This is the primary candidate.

For each sliding window start `ws`:

1. build `x = val_tokens[ws:end]`, `y = val_tokens[ws+1:end+1]`
2. run forward pass
3. define `scored_start = 0 if ws == 0 else max(wlen - stride, 0)`
4. accumulate metric only on `nll[scored_start:wlen]`
5. backprop only on `nll[scored_start:wlen]`
6. step TTT optimizer
7. advance to `ws + stride`

This is the cleanest reading of valid type-2 token-stream TTT.

### B. Doc-Based Streaming TTT

Secondary candidate, only if BOS boundaries are reliably available in the exported validation stream.

For each document:

1. reset model/optimizer adaptation state
2. run the same streaming loop inside that document only

Reason to test it:

- may reduce harmful cross-document adaptation on a shuffled validation set

Reason to defer it:

- boundary recovery may be ambiguous in the current pipeline

## Implementation Shape

Target code path: the March 23 record's `eval_val_sliding_ttt(...)`-style evaluator, not the root `train_gpt.py` baseline.

Replace:

- chunk-level two-phase `SCORE -> TRAIN`

With:

- window-level `SCORE_SUFFIX -> TRAIN_ON_SAME_SUFFIX`

### Exact Forward/Backward Rule

For a batch of windows:

1. compute `logits = model.forward_logits(x_batch)`
2. compute unreduced token NLL with `F.cross_entropy(..., reduction="none")`
3. for each sample `i`, define `suffix = nll[i, scored_start_i:wlen_i]`
4. add `suffix.sum()` to evaluation totals
5. if adaptation is enabled, define:

```python
adapt_loss = (
    sum(suffix.sum() for suffix in scored_suffixes)
    / sum(suffix.numel() for suffix in scored_suffixes)
)
```

6. backprop only that `adapt_loss`
7. optimizer step

Important:

- the same scored suffix may be used for both metric accumulation and the legal update
- the update must happen only after accumulation
- there is no need for `torch.inference_mode()` in the streaming version, because legality comes from the suffix mask, not from a hard phase split

## Parameter-Subset Strategy

Do not start with full-model streaming TTT as the only candidate. It is expensive and likely not quality-optimal per second.

Test three subsets:

### `full`

- all parameters adapt

Purpose:

- apples-to-apples comparison against the March 23 record

### `late_blocks`

- last 2-4 blocks only
- block-local scales/mixes/norms
- optional output head / tied embedding scale if already present

Purpose:

- likely best quality-per-time baseline

### `control_plus`

- later blocks
- `attn_scale`
- `mlp_scale`
- `resid_mix`
- `q_gain`
- VE/bigram/output-side small modules

Purpose:

- middle ground between March 20 micro-TTT and March 23 full-model TTT

Recommendation:

- make the subset selectable by env var, not by branch-specific code edits

## Optimizer Strategy

Start simple:

- `SGD(momentum=0.9)`

Why:

- this matches the March 23 legal TTT recipe
- easier to compare protocol changes before optimizer changes

Second wave:

- lower-LR SGD for high-frequency updates
- optional AdamW on small subsets only

Expected change from chunked TTT:

- because updates happen every `stride` tokens instead of every `chunk_size`, LR should likely be reduced from the March 23 chunked default

Initial guesses:

- `TTT_LR=2e-4` for `full`
- `TTT_LR=5e-4` for `late_blocks`
- `TTT_LR=1e-3` for `control_plus`

These are starting points, not tuned values.

## Scheduling

The March 23 chunked evaluator uses chunk-index cosine decay. That schedule does not map cleanly to streaming updates.

For the first candidate, use one of:

### `constant`

- easiest to reason about
- best for establishing whether the protocol itself helps

### `token_cosine`

- decay by fraction of validation tokens already scored

Preferred first implementation:

- `constant`

Only add schedule complexity after a clear gain is observed.

## Batching and Compute

Streaming TTT risks doubling eval compute if implemented naively.

To control cost:

- score and adapt in the same forward pass
- use unreduced per-token NLL
- avoid a separate no-grad scoring pass
- batch windows exactly as the sliding evaluator already does

Expected trade:

- more frequent optimizer steps
- fewer wasted tokens than chunked TTT
- lower latency to exploit newly scored evidence

## First Run Matrix

### Screen Runs

Purpose: identify legal protocol wins cheaply before full 600s timing runs.

Use:

- `1xH100`
- `USE_COMPILE=0`
- `SCREEN_SECONDS=180` if the harness supports it

Candidates:

1. `stream_full_lr2e4`
2. `stream_late4_lr5e4`
3. `stream_late2_lr5e4`
4. `stream_control_lr1e3`
5. `chunk_baseline_repro`

Success criteria:

- any streaming variant beats chunked legal TTT at matched artifact + matched eval legality
- eval time remains plausibly inside the 10-minute budget on 8xH100

### Full Runs

Promote only the top 1-2 screen winners.

Use:

- `8xH100`
- `DATA_ROOT_MODE=tmp`
- full 600s train budget

Suggested initial full-run order:

1. `late4` streaming TTT
2. `control_plus` streaming TTT
3. `full` streaming TTT only if one of the smaller subsets does not clearly dominate

## Logging Requirements

Add explicit log lines that make legality auditable:

- TTT mode: `chunk_score_first` or `stream_suffix`
- subset: `full`, `late4`, `control_plus`, etc.
- scored token count
- adapted token count
- confirmation that `adapted_token_count <= scored_token_count`
- eval-time per phase

Helpful periodic line:

```text
ttt_stream step_windows:123 scored_tokens:8192 adapted_tokens:8192 lr:0.0005 bpb_so_far:1.1203
```

## Acceptance Criteria

A candidate is worth keeping only if all are true:

1. legal by the invariant above
2. final eval remains under the competition time cap
3. artifact remains under 16 MB
4. post-quantized bpb improves against the current chunked legal-TTT baseline

## Main Risks

### Risk 1: hidden future-token leakage

Mitigation:

- build the suffix mask from the exact scored region
- never backprop on tokens earlier or later than that mask

### Risk 2: too many optimizer steps

Mitigation:

- start with selective subsets
- reduce LR
- optionally step every `k` windows while still restricting the loss to already-scored suffixes

### Risk 3: online updates destabilize late in eval

Mitigation:

- constant low LR first
- clip grads
- optional cap on total update norm per step

## Recommended First Candidate

If only one candidate is implemented first, it should be:

- token-stream streaming TTT
- `late_blocks` subset with last 4 blocks unfrozen
- `SGD(momentum=0.9)`
- constant LR `5e-4`
- same `seq_len=2048`
- same `stride=64`

Reason:

- strongest expected quality/time trade
- lower implementation and runtime risk than full-model online adaptation
- clean comparison against the March 23 chunked legal TTT

## Non-Goals For First Pass

Do not mix these into the first experiment:

- doc-reset mode
- optimizer changes beyond SGD
- activation changes
- architecture changes
- quantization changes

First establish whether the protocol upgrade alone is real.
