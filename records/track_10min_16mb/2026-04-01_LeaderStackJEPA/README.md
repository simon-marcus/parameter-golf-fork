# Leader-Stack JEPA

This folder is the draft submission package for our JEPA contribution.

The core claim is not just that we got a good number. The core claim is that we methodically tested whether JEPA itself was helping, separated that from stronger surrounding tricks, and then translated the best JEPA setting into a competitive `8xH100` candidate.

## Status

Current best completed `8xH100` result from seed `1337`:
- over-cap direct export: `1.12128254`
- under-cap storage-only export from the same checkpoint: `1.12271348`

Current storage-valid candidate:
- estimated total artifact bytes: `15,919,760`
- final exact `val_bpb`: `1.12271348`

This beats the March 17 naive baseline target of `1.2244`.

Remaining work before PR:
- finish the additional seeds `42` and `2026`
- flatten the current wrapper/import/export flow into a single submission-local `train_gpt.py`

See the legality audit in [LEGALITY_CHECK.md](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-04-01_LeaderStackJEPA/LEGALITY_CHECK.md).

## What We Wanted To Show

The organizers explicitly asked for JEPA and other weird ideas, even if they were not instantly SOTA. We treated that as both:
- a scientific challenge: does JEPA actually help in this benchmark?
- and a packaging challenge: can we turn that into a real contest candidate without muddying attribution?

We therefore ran two lanes in parallel:

1. `Proof lane`
   - isolate JEPA as cleanly as possible
   - compare matched control vs JEPA-on runs
   - use this to justify saying “JEPA helps”

2. `Competition lane`
   - add JEPA to a stronger stack
   - only promote if JEPA still helps after the translation
   - optimize toward an actual contest-worthy `val_bpb`

## Research Path

### 1. Byte-level exploratory JEPA hybrid
We started with [train_jepa.py](/Users/simon/Code/parameter-golf/train_jepa.py), a byte-level JEPA hybrid:
- `byte260` input
- causal backbone so BPB remains well-defined
- auxiliary JEPA loss on future byte-patch latents

This taught us the mechanics, but the long-horizon results were not strong enough to promote directly.

### 2. Isolation lane: prove JEPA is doing real work
We then built [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py), a minimal-diff JEPA lane based on the March 17 NaiveBaseline family.

The key discipline here was matched A/B testing:
- same script
- same data
- same hardware
- same wallclock
- only JEPA weight changed

Important outcome:
- weak early isolation variants did not help
- once the lane was strengthened with `LeakyReLU(0.5)^2`, JEPA began beating the matched control
- model EMA preserved that JEPA-over-control gain

Most important full-scale proof:
- `8xH100` EMA isolation control: `1.36487466`
- `8xH100` EMA isolation JEPA `0.10`: `1.36169919`

That is the cleanest evidence we have that JEPA itself was contributing signal in this repo.

### 3. Leader-stack translation
After proving JEPA in isolation, we translated it into the stronger March 23 leader family using [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py).

We were careful here:
- `TTT_ENABLED=0` during JEPA translation tests
- the initial goal was pre-TTT merit, not credit leakage from eval-time adaptation

Short screens were noisy and sometimes misleading. Longer confirmatory `1xH100` runs were more trustworthy.

The key confirmatory result:
- leader-stack control, `600s`, `1xH100`: `2.22466503`
- leader-stack JEPA `0.10`, `600s`, `1xH100`: `2.19728869`

That was strong enough to justify `8xH100` promotion.

### 4. `8xH100` competition run
On `8xH100`, seed `1337`, the exact JEPA leader candidate produced:
- `final_int8_zlib_roundtrip_exact val_bpb: 1.12128254`

This was excellent, but slightly over the size cap:
- `Total submission size int6+lzma: 16187860`

That shifted the problem from modeling to storage engineering.

### 5. Storage-only rescue from the saved checkpoint
Rather than retrain immediately, we ran a bounded export/eval pass from the saved checkpoint using [leader_stack_storage_pass.py](/Users/simon/Code/parameter-golf/leader_stack_storage_pass.py).

The accepted bounded fix was:
- drop duplicate exported JEPA alias weights
- quantize `attn, embed, mlp, other` to `int6`

Result:
- estimated total: `15,919,760`
- final exact: `1.12271348`

This is slightly worse than the over-cap export, but only by about `0.00143` BPB, and it is under the artifact cap.

## Why We Believe JEPA Is Actually Helping

We do **not** want to overclaim. The final candidate is not “pure JEPA alone.” It is JEPA inserted into a stronger stack.

So the argument must be layered:

1. In the isolation lane, matched `control` vs `JEPA` runs on `8xH100` showed JEPA helping.
2. In the leader-stack translation lane, longer `1xH100` confirmatory runs again showed JEPA helping.
3. Only then did we promote the JEPA-enhanced leader stack to `8xH100`.

That is why we think it is fair to say:
- JEPA was a real difference-maker in our research program
- even though the final competition candidate also depends on a stronger surrounding stack

## What We Will And Will Not Claim

### We will claim
- We implemented and tested JEPA seriously in this competition.
- We built a clean evidence ladder to check whether JEPA itself was helping.
- JEPA showed positive matched-control gains in both an isolation lane and a translated leader-stack lane.
- We converted that into an `8xH100` under-cap candidate at `1.12271348`.

### We will not claim
- That this is a pure JEPA model.
- That this is the byte-level/no-tokenizer JEPA path requested in Will DePue’s merch comment.
- That every improvement in the final candidate comes uniquely from JEPA.

The byte-level/no-tokenizer condition matters. Our strongest competition candidate is a SentencePiece-based leader stack, so it should not be pitched as satisfying the merch condition.

## Main Files

Research:
- [train_jepa.py](/Users/simon/Code/parameter-golf/train_jepa.py)
- [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py)
- [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py)
- [leader_stack_storage_pass.py](/Users/simon/Code/parameter-golf/leader_stack_storage_pass.py)
- [PLAN_AND_PROGRESS.md](/Users/simon/Code/parameter-golf/PLAN_AND_PROGRESS.md)

Durable logs:
- [jepa01_train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/jepa01_train.log)
- [storage_pass_allint6.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-04-01_leader_stack_jepa_8xh100/storage_pass_allint6.log)
- [tmp_control/train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-31_jepa_iso_8xh100_ema/records/tmp_control/train.log)
- [tmp_jepa01/train.log](/Users/simon/Code/parameter-golf/runpod_artifacts/2026-03-31_jepa_iso_8xh100_ema/records/tmp_jepa01/train.log)

## Submission Checklist

- [ ] finish seeds `42` and `2026`
- [ ] compute significance from the three seeds
- [ ] flatten wrapper + base + storage logic into one submission-local `train_gpt.py`
- [ ] emit the under-cap metric directly from the final script
- [ ] prepare `submission.json`
- [ ] copy logs for all three seeds into the final record folder

## Current Bottom Line

This work already succeeds as a genuine JEPA contribution:
- it is careful about attribution
- it preserves negative results instead of hiding them
- it demonstrates both JEPA strengths and limits

And if the remaining seeds hold up, it is also a serious competition submission.
