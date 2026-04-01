# JEPA Plan And Progress

## Goal
Learn JEPA through a real parameter-golf implementation without risking the main winning path, and only spend `8xH100` money after a cheaper evidence ladder is passed.

## Current Approach
- Branch: `codex/jepa-experiments`
- Main experiment file: [train_jepa.py](/Users/simon/Code/parameter-golf/train_jepa.py)
- Research brief: [program_jepa.md](/Users/simon/Code/parameter-golf/program_jepa.md)

The current model is a **byte-level JEPA hybrid**, not a pure JEPA:
- Input representation is `byte260`
- Backbone is still a causal transformer so we keep exact BPB evaluation
- Auxiliary JEPA loss predicts the latent embedding of the next byte patch
- Target latents come from an EMA target embedding
- The first purpose is to test whether JEPA-style representation pressure helps a byte-level control

This is deliberate. A pure JEPA does not naturally produce calibrated byte probabilities, while parameter golf is scored by BPB. The hybrid gives us a legitimate JEPA-inspired training signal without breaking evaluation.

## Why We Are Not Doing Full 8xH100 Yet
The right comparison is not against the tokenizer-optimized winning run. The right first comparison is:
- `train_jepa.py` with `JEPA_LOSS_WEIGHT=0.0` as the byte-level causal control
- versus the same script with JEPA enabled

If the JEPA hybrid cannot beat or at least credibly track its own byte-level control on cheap runs, it has not earned an `8xH100` promotion.

## Cheap Validation Ladder
### Stage 0: local logic checks
- `python3 -m py_compile train_jepa.py`
- CPU or local forward-pass smoke test
- Verify that JEPA metrics move when `JEPA_LOSS_WEIGHT > 0`
- Verify export and final roundtrip evaluation still run

Status:
- Completed for syntax and forward-pass smoke

### Stage 1: local or cheap GPU screens
Use [run_jepa_screen.sh](/Users/simon/Code/parameter-golf/run_jepa_screen.sh) with:
- `control`
- `jepa01`
- `jepa02`
- `patch16`
- `ema099`

What to look for:
- Does training remain stable?
- Does JEPA loss decrease?
- Does CE immediately worsen?
- Is step time still reasonable?

### Stage 2: 1xL4 or 1xH100 short screens
Use [launch_jepa_screen_runpod.sh](/Users/simon/Code/parameter-golf/launch_jepa_screen_runpod.sh) for the same variants.

Promotion criteria from this stage:
- JEPA variant is not obviously worse than control on early BPB
- Throughput penalty is acceptable
- Export still lands comfortably under `16,000,000` bytes

### Stage 3: 1xH100 longer confirmatory runs
Only the top 1-2 screen winners should be promoted.

Promotion criteria from this stage:
- Better or equal post-quantization BPB versus byte-level control
- Better trend over longer wallclock, not just noise
- No obvious artifact-size or quantization-gap failure

### Stage 4: 8xH100
Only after the above stages produce a credible winner.

## New Scripts
- [train_jepa.py](/Users/simon/Code/parameter-golf/train_jepa.py): byte-level JEPA hybrid trainer
- [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py): minimal-diff JEPA isolation lane built on the 2026-03-17 NaiveBaseline
- [program_jepa.md](/Users/simon/Code/parameter-golf/program_jepa.md): JEPA research brief
- [run_jepa_screen.sh](/Users/simon/Code/parameter-golf/run_jepa_screen.sh): local/manual control and A/B launcher
- [launch_jepa_screen_runpod.sh](/Users/simon/Code/parameter-golf/launch_jepa_screen_runpod.sh): RunPod screen launcher for cheap GPU promotion

## Isolation Lane
The exploratory hybrid in [train_jepa.py](/Users/simon/Code/parameter-golf/train_jepa.py) is useful for learning and discovery, but it does not cleanly answer whether JEPA itself helps.

The new isolation lane in [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py) is the cleaner scientific test:
- reuse the March 17 NaiveBaseline stack
- keep the same block structure, optimizer split, and int8+zlib export path
- switch to byte-level input by default
- add only a small JEPA auxiliary loss on top of baseline hidden states

Important limitation:
- this first isolation version uses a parameter-free JEPA predictor, not the richer predictor head from the exploratory hybrid
- that is intentional for now because it minimizes confounds
- if this lane shows promise, we can then add one more piece of complexity at a time

## Current Variants
- `control`: `JEPA_LOSS_WEIGHT=0.0`
- `jepa01`: `JEPA_LOSS_WEIGHT=0.1`
- `jepa02`: `JEPA_LOSS_WEIGHT=0.2`
- `patch16`: `PATCH_SIZE=16`, `JEPA_LOSS_WEIGHT=0.2`
- `ema099`: `TARGET_EMA=0.99`, `JEPA_LOSS_WEIGHT=0.2`

## Current Results
Cheap CUDA screens were run on `1x RTX A4500` with `byte260`, `TRAIN_BATCH_TOKENS=131072`, `TRAIN_SEQ_LEN=1024`, `USE_COMPILE=0`.

Short 120s screens:
- `control`: pre-quant `3.7549`, post-quant `3.8093`, `last_step=132`
- `jepa01`: pre-quant `3.7484`, post-quant `3.7998`, `last_step=131`
- `jepa02`: pre-quant `3.7481`, post-quant `3.7921`, `last_step=131`

Longer 600s confirmation:
- `jepa02`: pre-quant `1.7598`, post-quant `1.7647`, `last_step=648`, artifact `8,985,478` bytes
- `control`: pre-quant `1.7503`, post-quant `1.7556`, `last_step=661`, artifact `9,039,245` bytes

Current leader:
- `jepa02` is still the best JEPA setting so far
- It improved over the short 120s control
- It did **not** beat the matched 600s control
- The current evidence is therefore:
  - JEPA is alive and not obviously broken
  - JEPA may help early convergence in this byte-level regime
  - the present hybrid does not yet justify promotion as a stronger training recipe

## Competitive Assessment Prep
The right question is not "how do we copy tm0054?" The right question is "which tm0054 ideas are low-coupling enough to test inside the JEPA byte lane without destroying interpretability?"

Adopt soon:
- `LeakyReLU(0.5)^2` MLP activation. This is the cleanest transplant from the `2026-03-30_tm0054_AutoresearchTokenizer_LegalTTT` family. It is local to the feedforward path, already showed real gains there, and does not depend on tokenizer tricks or eval-time adaptation.
- Full-model weight EMA or EMA-plus-SWA. `train_jepa.py` already uses an EMA target embedding for JEPA targets, but it does not yet average the model weights for export/eval the way the stronger record-family runs did.

Possible later:
- Quantization/export refinements. The current JEPA path is only around `9MB`, so aggressive `int6+lzma` work is not yet necessary. It becomes relevant only if the JEPA family grows and starts to spend the artifact budget.
- More expressive local MLP activations beyond plain `relu^2`, but only after we finish the matched 600s control and keep the next A/Bs small.

Do not import yet:
- Legal TTT. That is an eval-time adaptation stack and would make it too easy to confuse JEPA training gains with eval-time gains.
- BigramHash and tokenizer-coupled representation tricks. Those belong to the tokenized record lane, not the byte-level JEPA lane.
- Parameter banking / Parallel Muon rewrites. Those are throughput engineering for the mature 8xH100 stack, not the next bottleneck on a cheap single-GPU JEPA investigation.

Recommended next order after the matched 600s control:
1. Run the new isolation lane in [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py):
   - byte-level control with `JEPA_LOSS_WEIGHT=0.0`
   - the same byte-level baseline with `JEPA_LOSS_WEIGHT=0.1` and `0.2`
2. If the isolation lane shows a positive JEPA delta, add `LeakyReLU(0.5)^2` to [train_jepa.py](/Users/simon/Code/parameter-golf/train_jepa.py) and rerun the exploratory 120s screen matrix.
3. Add model-weight EMA export/eval and repeat the `600s` leader-vs-control check.
4. Only if JEPA recovers a long-horizon advantage should we consider larger architectural or eval-time ideas.

## Recommended First Commands
Download minimal byte-level data locally:

```bash
python3 data/cached_challenge_fineweb.py --variant byte260 --train-shards 1
```

Run the local byte-level control:

```bash
./run_jepa_screen.sh control
```

Run the first JEPA A/B:

```bash
./run_jepa_screen.sh jepa01
./run_jepa_screen.sh jepa02
```

On a cheap pod:

```bash
USE_COMPILE=0 DATA_ROOT_MODE=tmp NPROC_PER_NODE=1 SCREEN_SECONDS=180 bash ./launch_jepa_screen_runpod.sh control
USE_COMPILE=0 DATA_ROOT_MODE=tmp NPROC_PER_NODE=1 SCREEN_SECONDS=180 bash ./launch_jepa_screen_runpod.sh jepa02
```

## Open Questions
- Does JEPA help byte-level convergence at all in this regime?
- Is patch prediction better at `PATCH_SIZE=8` or `16`?
- Is a simple EMA embedding target enough, or do we need a real target encoder?
- If the hybrid works, should the next step be a stronger JEPA target, or a smaller decoder head?

## Notes
- `autoresearch.py` is still aimed at `train_gpt.py` style mutations and is not yet wired to drive `train_jepa.py`.
- That is intentional for now. We should first earn the right to automate JEPA search by proving the family is alive on cheap runs.

## 8xH100 Promotion Status
- The byte-level isolation lane in [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py) is the current promotion candidate.
- A dedicated launcher now exists at [launch_jepa_baseline_runpod.sh](/Users/simon/Code/parameter-golf/launch_jepa_baseline_runpod.sh).
- Byte-level `/tmp` staging now has an explicit helper at [setup_byte260_data_runpod.sh](/Users/simon/Code/parameter-golf/setup_byte260_data_runpod.sh).

`US-GA-2` prep volume status:
- volume: `gs27hsi4q0`
- prep pod: `jr9c5i1c4pf06b`
- repo root: `/workspace/pg-data/parameter-golf`
- JEPA-only transfer bundle: `/workspace/pg-data/jepa_iso_bundle/parameter-golf-jepa-iso-bundle.tar`
- leader-stack code bundle: `/workspace/pg-data/parameter-golf/staging_bundles/parameter-golf-leader-stack-jepa-bundle.tar`
- leader-stack archive relay helper: `/workspace/pg-data/parameter-golf/relay_runpod_archive.sh`
- cheap regional staging helper: `/workspace/pg-data/parameter-golf/seed_leader_stack_staging_pod.sh`
- staged `sp1024` dataset: `/workspace/pg-data/parameter-golf/data/datasets/fineweb10B_sp1024`
- staged `byte260` dataset: `/workspace/pg-data/parameter-golf/data/datasets/fineweb10B_byte260`
- staged `sp1024` tokenizer: `/workspace/pg-data/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`
- staged `byte260` tokenizer: `/workspace/pg-data/parameter-golf/data/tokenizers/fineweb_pure_byte_260.json`
- staged `sp1024` archive: `/workspace/pg-data/parameter-golf/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst`

Validation note:
- The standard verifier on the CPU prep pod failed only because that pod does not have `numpy`.
- A plain Python stdlib header check passed on all staged `byte260` shard files.
- A minimal JEPA-isolation transfer bundle was created on the prep pod at about `861M`, so future `8xH100` pods can be seeded with a single transfer instead of a noisy full-repo sync.

Intended `8xH100` run sequence once capacity is available:

```bash
cd /workspace/parameter-golf
DATA_ROOT_MODE=tmp NPROC_PER_NODE=8 MAX_WALLCLOCK_SECONDS=600 bash ./launch_jepa_baseline_runpod.sh control
DATA_ROOT_MODE=tmp NPROC_PER_NODE=8 MAX_WALLCLOCK_SECONDS=600 bash ./launch_jepa_baseline_runpod.sh jepa01
```

Current blocker:
- RunPod REST pod creation for an `8x H100` pod in `US-GA-2` with `networkVolumeId=gs27hsi4q0` currently returns `There are no instances currently available`.
- Cross-region archive copies directly into a live `8xH100` pod are too slow and too expensive to use as the standard path.
- New operational rule: before any future leader-stack `8xH100` launch, first seed a cheap staging pod in the target region with the leader-stack code bundle and the local `sp1024` archive. Only then create/promote the `8xH100`.

## Durable Results Log

### 8xH100 matched isolation runs
Using [train_jepa_baseline.py](/Users/simon/Code/parameter-golf/train_jepa_baseline.py) on `8x H100`, `byte260`, `600s`, `DATA_ROOT_MODE=tmp`:

- `control` (`JEPA_LOSS_WEIGHT=0.0`): `final_int8_zlib_roundtrip_exact val_bpb = 1.36692884`
- `jepa01` (`JEPA_LOSS_WEIGHT=0.1`): `final_int8_zlib_roundtrip_exact val_bpb = 1.36182961`
- `jepa02` (`JEPA_LOSS_WEIGHT=0.2`): `final_int8_zlib_roundtrip_exact val_bpb = 1.36663146`

Current conclusion from the real promotion run:
- JEPA is helping in the isolation lane on `8xH100`
- `0.1` is the best weight proven so far at full scale
- the gain over the matched control is about `0.00510` BPB
- this is a credible JEPA-positive result, but still well above the March 17 target of `1.2244`

### Cheap single-GPU JEPA weight sweep
Using the isolation lane on a cheap single GPU, `byte260`, `180s`, `TRAIN_BATCH_TOKENS=131072`, `TRAIN_SEQ_LEN=1024`:

- `w0p65`: `final_int8_zlib_roundtrip_exact val_loss:1.29404186 val_bpb:1.86690778`
- `w0p15`: `final_int8_zlib_roundtrip_exact val_loss:1.29404957 val_bpb:1.86691890`
- `w0p55`: `final_int8_zlib_roundtrip_exact val_loss:1.29478845 val_bpb:1.86798488`
- `w0p05`: `final_int8_zlib_roundtrip_exact val_loss:1.29605058 val_bpb:1.86980574`
- `w1p00`: `final_int8_zlib_roundtrip_exact val_loss:1.29631343 val_bpb:1.87018496`
- `w0p45`: `final_int8_zlib_roundtrip_exact val_loss:1.29670615 val_bpb:1.87075153`
- `w0p95`: `final_int8_zlib_roundtrip_exact val_loss:1.29781500 val_bpb:1.87235127`
- `w0p90`: `final_int8_zlib_roundtrip_exact val_loss:1.29793175 val_bpb:1.87251969`
- `w0p50`: `final_int8_zlib_roundtrip_exact val_loss:1.29838871 val_bpb:1.87317895`
- `w0p10`: `final_int8_zlib_roundtrip_exact val_loss:1.29852931 val_bpb:1.87338179`
- `w0p80`: `final_int8_zlib_roundtrip_exact val_loss:1.29902096 val_bpb:1.87409110`
- `w0p25`: `final_int8_zlib_roundtrip_exact val_loss:1.29983709 val_bpb:1.87526852`
- `w0p40`: `final_int8_zlib_roundtrip_exact val_loss:1.29986623 val_bpb:1.87531057`
- `w0p30`: `final_int8_zlib_roundtrip_exact val_loss:1.30032384 val_bpb:1.87597075`
- `w0p70`: `final_int8_zlib_roundtrip_exact val_loss:1.30298896 val_bpb:1.87981571`
- `w0p75`: `final_int8_zlib_roundtrip_exact val_loss:1.30324671 val_bpb:1.88018757`
- `w0p85`: `final_int8_zlib_roundtrip_exact val_loss:1.30369632 val_bpb:1.88083622`
- `w0p20`: `final_int8_zlib_roundtrip_exact val_loss:1.30434502 val_bpb:1.88177209`
- `w0p35`: `final_int8_zlib_roundtrip_exact val_loss:1.30475990 val_bpb:1.88237064`
- `w0p60`: `final_int8_zlib_roundtrip_exact val_loss:1.30593691 val_bpb:1.88406871`
- `w0p00`: `final_int8_zlib_roundtrip_exact val_loss:1.31746854 val_bpb:1.90070533`

Interpretation:
- the cheap sweep shows the JEPA-weight response is not monotonic around `0.1`
- on the cheap screen, many nonzero JEPA weights beat `0.1`, with the best result at `0.65`
- this does not overrule the `8xH100` evidence, but it does justify testing at least one larger weight at full scale beyond `0.2`

### Leader-stack 1xH100 confirmatory runs
Using [train_gpt_leader_stack_jepa.py](/Users/simon/Code/parameter-golf/train_gpt_leader_stack_jepa.py) on `1x H100`, `sp1024`, `TTT_ENABLED=0`, `DATA_ROOT_MODE=tmp`:

Short matched sweep, `180s`:
- `control`: `4.53858963`
- `weight:0.05`: `4.52191317`
- `jepa01` / `0.10`: `4.52670811`
- `weight:0.15`: `4.51712679`

Longer confirmatory, `600s`:
- `control`: `2.22466503`
- `weight:0.15`: `2.29532844`

Follow-up longer JEPA-only check against the same `600s` control baseline:
- `weight:0.05`: `2.22707166`
- `jepa01` / `0.10`: `2.19728869`

Current leader-stack conclusion:
- short screens were misleading about `0.15`
- at longer horizon, `0.15` loses badly to control
- `0.05` is roughly neutral/slightly worse than control
- `0.10` is the strongest leader-stack JEPA setting tested so far, beating the `600s` control by about `0.02738` BPB

Operational note:
- the raw logs for the last `600s` `0.05` / `0.10` follow-up were not fully copied before pod shutdown because pull and stop were launched in parallel.
- the exact final metrics above were captured before shutdown and preserved here.

## 2026-04-01 CPU Staging Probe

Goal:
- measure whether direct ordinary AWS S3 -> Runpod is fast enough to replace slow pod-to-pod archive copies for future leader-stack promotions

Setup:
- archive under test:
  - `s3://parameter-golf-staging-094651608775/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst`
- method:
  - generate a presigned URL locally
  - on each cheap CPU pod, `curl -L -r 0-536870911` the first `512MB`
  - compare `speed_download`

Results:
- `US-GA-2` cheap CPU pod (`runpod/parameter-golf:latest`): about `35.5 MB/s`
- `US-GA-2` prep pod (`runpod/base:1.0.2-ubuntu2204`): about `57.3 MB/s`
- `US-IL-1` cheap CPU pod (`runpod/parameter-golf:latest`): about `46.1 MB/s`
- `AP-IN-1` cheap staging pod (`runpod/parameter-golf:latest`): about `5.0 MB/s`

Interpretation:
- direct AWS S3 -> Runpod is a viable staging path in at least some US regions
- `AP-IN-1` remains a bad fit for data-heavy startup unless the archive is already local
- Canada CPU probes could not be completed because `CA-MTL-1`, `CA-MTL-2`, and `CA-MTL-3` had no available CPU capacity at probe time
