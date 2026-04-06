# PR #1019 Transfer Notes For Scylla

This note evaluates which elements of the current winning stack in PR `#1019`
look clean to adopt into the Scylla / TokenMonster line, and which ones are
likely to be separate projects.

## Short Answer

Best clean imports:
- `XSA_LAST_N=11`
- `WARMDOWN_ITERS=4000`
- keep the broader export-side quantization mindset

Most promising but not clean:
- Full Hessian GPTQ with AR self-generated calibration

Probably not the right import:
- dropping TTT
- relying on `lzma preset=9` as a meaningful lever
- blindly copying the `3072 x 112` bigram setting

## Why

Scylla differs from PR `#1019` in three important ways:
- tokenizer geometry is different
- pre-quant legal TTT is load-bearing for Scylla
- Scylla's current under-cap path already drops `bigram.*` in the serialized artifact

That means we should separate ideas that improve the trained model from ideas
that only make sense in the PR `#1019` export/eval stack.

## Clean Ports

### 1. XSA On All 11 Layers

PR `#1019` applies XSA across all 11 layers. Scylla already has the same control
surface:
- [train_gpt_legal_ttt.py](/Users/simon/Code/parameter-golf/records/codex_scylla_2/train_gpt_legal_ttt.py#L96)
- [train_gpt_legal_ttt.py](/Users/simon/Code/parameter-golf/records/codex_scylla_2/train_gpt_legal_ttt.py#L1056)

Current Scylla default is `XSA_LAST_N=4`.

This is the cleanest architectural import because:
- zero parameter cost
- no tokenizer coupling
- no export-format coupling
- low code risk

This should be the first training-side adoption from PR `#1019`.

### 2. Warmdown = 4000

PR `#1019` uses `WARMDOWN_ITERS=4000`.

Scylla already exposes the same knob:
- [train_gpt_legal_ttt.py](/Users/simon/Code/parameter-golf/records/codex_scylla_2/train_gpt_legal_ttt.py#L52)
- [train_gpt_legal_ttt.py](/Users/simon/Code/parameter-golf/records/codex_scylla_2/train_gpt_legal_ttt.py#L2263)

This is low-risk and should be bundled with the XSA-all ablation.

### 3. Export-Side Broad Quantization Mindset

PR `#1019` wins partly because its quantization/export path is stronger than the
older mixed-int6 baseline.

Scylla already benefits from a broader export-side quantization pass in the
storage ladder, even though it is not GPTQ yet. That principle is compatible.

So the transferable idea is not "copy their exact quantizer now", but:
- keep treating export as an optimization surface
- expect real gains from better post-training quantization

## Conditional Ports

### 4. BigramHash Retuning

PR `#1019` uses `BIGRAM_VOCAB_SIZE=3072`, `BIGRAM_DIM=112`.

Scylla has the same family of module:
- [train_gpt_legal_ttt.py](/Users/simon/Code/parameter-golf/records/codex_scylla_2/train_gpt_legal_ttt.py#L94)
- [train_gpt_legal_ttt.py](/Users/simon/Code/parameter-golf/records/codex_scylla_2/train_gpt_legal_ttt.py#L821)

But this is not a clean copy-paste because:
- Scylla uses a different tokenizer and different bigram statistics
- current Scylla under-cap export drops `bigram.*` entirely with no final legal-TTT regression
- the PR `#1019` bigram choice is tuned for an SP1024 + no-TTT + GPTQ stack

So bigram retuning is a valid Scylla research direction, but not the first one.

### 5. Selective Post-Training Pruning

PR `#1019` uses selective `+1/-1` pruning by reconstruction error.

This looks more compatible with Scylla than full GPTQ because it can live in the
export/storage path. But it still touches the same now-stable path that got
Scylla under cap, so it should be treated as a medium-risk export project.

## Separate Project

### 6. Full Hessian GPTQ With AR Self-Generated Calibration

This is the biggest upside idea in PR `#1019`.

Why it is attractive:
- likely improves quantization quality materially
- may shrink artifact size and reduce quantization gap at the same time
- AR self-generated calibration avoids legality problems

Why it is not a clean transplant:
- Scylla currently uses a simpler mixed-int6 export path
- Scylla's best result depends on pre-quant legal TTT
- GPTQ changes the quantization stack, export stack, and likely the best eval ordering

So GPTQ is probably the highest-upside Scylla follow-up from PR `#1019`, but it
should be treated as its own project rather than a quick stack merge.

## What Not To Copy

### 7. Dropping TTT

PR `#1019` is a no-TTT stack. That does not transfer cleanly.

Scylla's current best result depends on pre-quant score-first TTT. So "drop TTT"
would remove one of Scylla's current strengths rather than import one of PR
`#1019`'s strengths.

### 8. LZMA Preset 9 As A Major Lever

We already tested stronger LZMA in the Scylla storage passes and it was not a
meaningful size win. So that part of PR `#1019` should not receive more focus.

## Recommended Adoption Order

1. `XSA_LAST_N=11`
2. `WARMDOWN_ITERS=4000`
3. Re-test Scylla with current best pre-quant TTT path
4. If still promising, evaluate a careful bigram retune
5. In parallel or after that, start a separate Scylla-GPTQ branch using AR
   self-generated calibration

## Working Thesis

The best clean lesson from PR `#1019` is not "copy the whole winner".

It is:
- port the zero-cost training improvements first
- keep Scylla's successful pre-quant TTT
- treat GPTQ as the real longer-horizon transfer
