# RunPod Workflow

This file is the canonical RunPod runbook for this repo. `CLAUDE.md` carries the same RunPod rules in shorter form, but this file is the version to follow when operating pods.

## Current Recommended Pod Path

Use the standard template-backed RunPod flow that is already wired into [`pod.sh`](/Users/simon/Code/parameter-golf/pod.sh).

Why:
- it has been materially faster to start in practice
- it has been more reliable about reaching SSH-ready state
- the custom image/template path is currently suspended as a default because we observed repeated cold-start stalls where the pod stayed `RUNNING` but never became SSH-ready

Current rule:
- do not treat `ghcr.io/simon-marcus/parameter-golf-runpod:cuda128-torch210` / `vtoarnccmw` as the preferred default until that startup regression is resolved
- if you intentionally test that custom image path again, treat it as experimental

## Storage Model

Use `/workspace` for persistent state:
- repo checkout
- dataset cache
- tokenizer cache
- logs
- pulled artifacts

Use `/tmp/parameter-golf-data` for fast run-time staging:
- `DATA_PATH`
- `TOKENIZER_PATH`
- timed training/eval runs

Rule:
- `/workspace` is durable
- `/tmp` is fast but ephemeral

## Datacenter Rule

Network volumes are datacenter-locked.

Efficient pattern:
1. Create the persistent volume in the target datacenter.
2. Attach it to a cheap prep pod in that same datacenter.
3. Download/cache repo data under `/workspace`.
4. Stop the cheap pod.
5. Attach the same volume to the expensive pod in that same datacenter.
6. Stage from `/workspace/...` to `/tmp/parameter-golf-data/...`.

Do not prepare data in one datacenter and expect to reuse that volume in another.

## Pod Creation

Even when using the template, pass disk and volume flags explicitly.

### 1xH100 Smoke Pod

```bash
runpodctl pod create \
  --name "pg-smoke" \
  --template-id "y5cejece4j" \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --cloud-type SECURE \
  --ssh \
  --container-disk-in-gb 50 \
  --volume-in-gb 50 \
  -o json
```

### 8xH100 Record Pod

```bash
runpodctl pod create \
  --name "pg-record" \
  --template-id "y5cejece4j" \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 8 \
  --cloud-type SECURE \
  --ssh \
  --container-disk-in-gb 50 \
  --volume-in-gb 50 \
  -o json
```

Then:
1. Get SSH info: `runpodctl pod get <pod_id> -o json`
2. Read `.ssh.ip` and `.ssh.port`
3. Use the direct TCP SSH endpoint, not the wrapper endpoint, for automation

## SSH And Sync

### Direct SSH

```bash
ssh root@HOST -p PORT -i ~/.ssh/id_ed25519
```

### Sync Repo

Use a narrow `rsync` and keep large local junk out of the pod:

```bash
rsync -avz -e "ssh -p PORT -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519" \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude '.venv*' \
  --exclude '.claude' \
  --exclude '.gstack' \
  --exclude 'data/datasets' \
  --exclude 'data/tokenizers' \
  --exclude 'runpod_artifacts' \
  --exclude 'modal_logs' \
  /Users/simon/Code/parameter-golf/ \
  root@HOST:/workspace/parameter-golf/
```

## Required Preflight

On every fresh pod:

```bash
ssh root@HOST -p PORT "bash /opt/parameter-golf/validate_image.sh"
```

After repo sync:

```bash
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ops/runpod_image/validate_image.sh"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && mkdir -p logs"
```

Verify workspace data:

```bash
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ./verify_runpod_data_ready.sh /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
```

Stage to `/tmp`:

```bash
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ./setup_local_parity_data_runpod.sh"
```

If `/workspace/parameter-golf/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst` exists, `setup_local_parity_data_runpod.sh` now prefers that archive automatically instead of copying the directory tree shard-by-shard. To build the reusable archive on a prep machine:

```bash
cd /workspace/parameter-golf
bash ./build_runpod_data_archive.sh ./data ./data/archives fineweb10B_sp1024 fineweb_1024_bpe
```

For expensive leader-stack promotions, do not rely on a live cross-region archive copy into the `8xH100` pod. The archive must already be present locally on the pod before launch. The leader-stack launcher now refuses multi-GPU `/tmp` runs if `/workspace/parameter-golf/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst` is missing.

Recommended leader-stack `8xH100` prep flow:
1. Keep the durable source archive and code bundle on the prep pod or prep volume.
2. When `8xH100` capacity appears in a region, first create a cheap staging pod in that same region.
3. Copy the leader-stack code bundle and the `sp1024` archive onto the cheap staging pod.
4. Verify the archive and, if possible, keep the cheap pod alive while waiting for the `8xH100`.
5. Only then create the `8xH100` pod in the same region and copy the already-local archive across.
6. Launch the leader-stack run only after the archive exists on the `8xH100` pod and preflight passes.

Reusable prep assets:
- archive: `/workspace/pg-data/parameter-golf/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst`
- code bundle: `/workspace/pg-data/parameter-golf/staging_bundles/parameter-golf-leader-stack-jepa-bundle.tar`
- relay helper: `/workspace/pg-data/parameter-golf/relay_runpod_archive.sh`

Verify `/tmp`:

```bash
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ./verify_runpod_data_ready.sh /tmp/parameter-golf-data/datasets/fineweb10B_sp1024 /tmp/parameter-golf-data/tokenizers/fineweb_1024_bpe.model"
```

Do not launch `torchrun` before all of the above pass.

## Standard Runs

### 1xH100 Smoke

```bash
ssh root@HOST -p PORT "cd /workspace/parameter-golf && USE_COMPILE=0 DATA_ROOT_MODE=tmp NPROC_PER_NODE=1 SCREEN_SECONDS=180 bash ./launch_leadercore_screen_runpod.sh base"
```

### 8xH100 Record-Style

```bash
ssh root@HOST -p PORT "cd /workspace/parameter-golf && DATA_ROOT_MODE=tmp bash ./launch_leadercore_ablation_runpod.sh base"
```

## Rules

- Prefer the standard template-backed pod flow over the suspended custom image path.
- Never download data on an expensive pod by default.
- Never trust `/tmp` as persistent storage.
- Never launch before preflight passes.
- For multi-GPU leader-stack runs, never launch until the local `sp1024` archive is already present on the pod.
- Always pull logs and artifacts before stopping a pod.
- If a new image/template repeatedly stays `RUNNING` without becoming SSH-ready, do not adopt it as the default path.
- Use `DATA_ROOT_MODE=tmp` for serious throughput or record-style runs.
- Use `USE_COMPILE=0` for cheap smoke/debug runs; keep compile on for real runs unless debugging compile itself.

## Known Failure Modes

- Missing `zstandard` causing `zlib` fallback and oversized artifacts
- Missing `/tmp/parameter-golf-data/tokenizers/...` on resumed pods
- Overlay-copy corruption when staging into `/tmp`
- Missing `logs/` directory on fresh pods
- Template-backed pod creation silently dropping disk/volume unless passed explicitly
- Pulling artifacts after stopping the pod
- Image starts but exits immediately because the container command does not stay alive
- Direct SSH/SCP/rsync broken because `sshd` is missing from the image
- Custom-image cold pulls that never progress to SSH-ready despite the pod showing `RUNNING`
