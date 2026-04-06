# Byte-Level JEPA Attack Plan

## Objective
Beat the March 17 Naive Baseline `1.22436570` with a true byte-level JEPA-based submission:
- no tokenizer
- byte-level input only
- JEPA must beat a matched byte-level control

## Ground Truth

Best current byte-level JEPA result:
- `1.36169919` from the EMA byte-level isolation lane

Gap to baseline:
- about `0.1373` BPB

Conclusion:
- this is not a small tuning problem
- the next win probably does not come from another JEPA weight sweep

## Best Current Priors

### 1. Preserve what already worked
- `LeakyReLU(0.5)^2`
- model EMA
- JEPA auxiliary weight around `0.10`

These are the strongest low-coupling wins we have already observed in byte-level experiments.

### 2. The capacity question is answered
We ran the cheap `1xH100` capacity screen and got:
- `control` (`9x512`): `1.87138830`
- `jepa01` (`9x512`): `1.84092584`
- `depth12_control` (`12x512`): `2.33624034`
- `depth12_jepa01` (`12x512`): `2.29014684`
- `depth12wide_control` (`12x576`): `2.77315789`
- `depth12wide_jepa01` (`12x576`): `2.74212553`

Interpretation:
- the larger models were dramatically worse
- the current byte-level lane is throughput-bound, not parameter-starved
- we should stop scaling the old 1024-byte backbone and move directly to patch-first modeling

### 3. Make byte patches the real unit of modeling
The biggest weakness of the current byte-level lane is probably not JEPA itself. It is that the backbone is still too close to a token-level recipe transplanted onto bytes.

The plan should be:
- keep raw bytes as the only alphabet
- group bytes internally into fixed patches
- let the backbone operate mainly on patch summaries
- decode exact bytes with a small local head
- put JEPA pressure on future patch latents

This preserves the “no tokenizer” rule while giving the model a more semantically useful representation unit.

### 4. Add more effective depth per parameter
Byte-level modeling likely needs more effective depth or memory than the current baseline family provides.

Most promising forms:
- shared-depth recurrence
- universal-transformer style repeated blocks
- patch-level state-space or recurrent memory

These are better bets than simply making the byte transformer wider.

### 5. Only then use eval-time methods
If the training recipe gets close to the baseline:
- improve sliding-window byte eval
- possibly add legal score-first adaptation

This should be a late multiplier, not the first rescue path.

## Concrete Experiment Order

### Stage A: strengthen the existing byte-level isolation lane
Use [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py) as the truth lane.

Keep:
- byte-level data
- `LeakyReLU^2`
- model EMA
- JEPA `0.10`

Capacity result:
- tested and rejected as the next step

Do next:
- preserve this lane as the matched control/ablation truth lane
- keep `LeakyReLU^2`, model EMA, and JEPA `0.10`
- use it as the reference to beat with the patch-first candidate

Do not do first:
- more depth/width scaling of the current backbone
- longer `TRAIN_SEQ_LEN`
- large JEPA-weight sweeps
- optimizer rewrites
- eval-time tricks

Success criterion:
- materially better than `1.3617` on `8xH100`

### Stage B: build a patch-first byte JEPA lane
New candidate:
- internal patch embedding over raw bytes
- patch-level recurrent/shared transformer
- local byte decoder head
- JEPA on next-patch latent prediction

This is the first genuinely new architecture that could plausibly recover large BPB.

Current progress:
- first patch-first attempt (`PATCH_SIZE=4`, GRU-style local decoder): too weak overall, JEPA hurt
- second patch-first attempt (`PATCH_SIZE=2`, conditional local MLP decoder): JEPA helped again, but the whole patched family was still much worse than the old byte-level baseline lane
- next two probes are focused on the patch representation itself:
  - local byte mixer inside each patch
  - intra-patch causal attention before patch reduction

Success criterion:
- beat the strengthened Stage A byte-level control
- maintain a clean JEPA-on vs JEPA-off comparison inside the patched lane

### Stage C: only then import heavier leader-stack ideas
Possible later imports:
- XSA
- VE layers
- banked parameters / stronger optimizer partitioning

Do not open with these. They are powerful but high-coupling.

## Things To Avoid First
- more large JEPA-weight sweeps
- BigramHash-like features that blur the “true byte-level” spirit
- tokenizer-dependent tricks
- TTT before the byte-level training recipe is already competitive
- MTP heads before the backbone is clearly strong enough

## Claim Discipline
If we eventually win this challenge, the claim should remain precise:
- bytes only
- JEPA beats a matched byte-level control
- final score beats `1.2244`

That is the standard this folder is for.
