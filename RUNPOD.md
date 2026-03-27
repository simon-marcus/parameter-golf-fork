# RunPod Workflow

This file is the canonical RunPod runbook for this repo. `CLAUDE.md` carries the same RunPod rules in shorter form, but this file is the version to follow when operating pods.

## Current Preferred Image And Template

- Image: `ghcr.io/simon-marcus/parameter-golf-runpod:cuda128-torch210`
- Template: `vtoarnccmw`

Why this image:
- pinned CUDA/Torch stack
- `zstandard` preinstalled
- helper tools preinstalled
- direct TCP SSH, `scp`, and `rsync` supported
- `PUBLIC_KEY` injected on boot
- persistent container startup via `sshd` + keepalive

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
  --template-id "vtoarnccmw" \
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
  --template-id "vtoarnccmw" \
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

- Prefer the pinned image/template over rebuilding expensive pods from scratch.
- Never download data on an expensive pod by default.
- Never trust `/tmp` as persistent storage.
- Never launch before preflight passes.
- Always pull logs and artifacts before stopping a pod.
- Reused pods are expected; cold starts on this custom image can be slow.
- If direct TCP SSH or `rsync` stop working for a new image revision, do not adopt it.
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
