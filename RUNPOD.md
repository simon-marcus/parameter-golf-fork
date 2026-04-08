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

Use `/workspace` for durable pod-local state:
- repo checkout
- downloaded archives and bundles
- logs
- pulled artifacts

Use `/tmp` for the actual hot-path run bundle:
- `DATA_PATH`
- `TOKENIZER_PATH`
- `TOKENIZER_META_PATH`
- timed training/eval runs

Rule:
- `/workspace` is the pod-local cache
- `/tmp` is the fast execution path

Preferred pattern for serious runs:
1. download a dataset archive and a small code bundle into `/workspace`
2. extract the code bundle into `/workspace/parameter-golf`
3. extract or stage the dataset into `/tmp/...`
4. run preflight against the `/tmp` paths
5. only then launch `torchrun`

## Datacenter Rule

Network volumes are datacenter-locked, but they are no longer the default recommendation for recurring assets.

Current preferred durable source:
- ordinary AWS S3

Why:
- it avoids datacenter lock
- it avoids depending on a warm prep pod or volume handoff
- for US regions we have already measured acceptable download speed

Use a RunPod network volume only when:
- S3 is unavailable or too slow in the target region
- or you already have a same-datacenter prep workflow set up

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

Do not assume `/opt/parameter-golf/validate_image.sh` exists on every acceptable template. Some good template-backed pods do not provide it.

Safe rule:
- if `/opt/parameter-golf/validate_image.sh` exists, run it
- otherwise continue with the repo-local validation and actual data preflight

Optional image validation:

```bash
ssh root@HOST -p PORT 'test -x /opt/parameter-golf/validate_image.sh && bash /opt/parameter-golf/validate_image.sh || true'
```

Repo-local validation:

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

For expensive `8xH100` runs, the dataset archive and code bundle must exist locally on the pod before launch.

Current preferred `8xH100` prep flow:
1. Keep the durable dataset archive and a small experiment-specific code bundle in AWS S3.
2. When `8xH100` capacity appears, create the pod and wait for SSH.
3. Download both assets directly onto that pod with presigned URLs and `curl`.
4. Extract the code bundle into `/workspace/parameter-golf`.
5. Stage the dataset archive into `/tmp/...`.
6. Run `verify_runpod_data_ready.sh` against the staged `/tmp` paths.
7. Only then launch the run.

Use a cheap prep pod or a same-datacenter volume only as fallback, not as the default.

## Recent Pitfalls

Keep these in mind for short Scylla screens and similar pod fanouts:

- Verify that each pod has exactly one active `torchrun` and one worker process after launch.
  We hit a bad relaunch state where two overlapping jobs wrote into the same `train.log`, which doubled the observed `step_avg` and made the run look much slower than it really was.
  Useful check:

```bash
ssh root@HOST -p PORT "pgrep -af 'train_gpt_legal_ttt|torchrun' || true"
```

- For `1xH100` smoke/screen runs, prefer `USE_COMPILE=0` and confirm the training script actually honors it.
  If the script still calls `torch.compile` unconditionally, startup/compile time can dominate the run and distort timing comparisons.

- Do not assume the template image can decode every export codec.
  On these template-backed pods, system Python did not have `brotli` or `zstandard`, and PEP 668 blocked casual `pip install` into the base environment.
  For quick screens, prefer `COMPRESSOR=lzma BYTE_SHUFFLE=0` unless you have already provisioned those Python modules in a venv or the image.

- If you add byte-based TTT chunking, force token ids to integer dtype before LUT indexing.
  We hit a real crash in `_ttt_chunk_bounds()` until `prev_ids` and `tgt_ids` were cast to `torch.int64`.

## Runpod S3-Compatible API

The Runpod S3-compatible API is a strong candidate for reducing prep friction, but it only works in the datacenters listed in Runpod's S3 docs. `AP-IN-1` is not currently on that list, so do not assume the S3 path will help an `AP-IN-1` run directly.

What is ready in this repo:
- [runpod_s3_env.example](/Users/simon/Code/parameter-golf/runpod_s3_env.example)
- [runpod_s3_probe.sh](/Users/simon/Code/parameter-golf/runpod_s3_probe.sh)
- [runpod_s3_cp.sh](/Users/simon/Code/parameter-golf/runpod_s3_cp.sh)
- [runpod_s3_sync_leader_stack_assets.sh](/Users/simon/Code/parameter-golf/runpod_s3_sync_leader_stack_assets.sh)

Current state:
- existing supported volume: `gs27hsi4q0` in `US-GA-2`
- local `aws` CLI is installed
- current local AWS credentials are **not** a Runpod S3 API key, so Runpod S3 auth currently fails with `SignatureDoesNotMatch`

Next step before S3 testing:
1. Create a Runpod S3 API key in the Runpod console.
2. Export those credentials or place them in an AWS profile.
3. Run:

```bash
cp runpod_s3_env.example .env.runpod-s3
# fill in AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from Runpod S3 API Keys
source .env.runpod-s3
bash ./runpod_s3_probe.sh
```

If the probe works, then sync the regular leader-stack assets:

```bash
source .env.runpod-s3
bash ./prepare_leader_stack_jepa_bundle.sh
bash ./runpod_s3_sync_leader_stack_assets.sh
```

## AWS S3 Staging

Independent of Runpod's S3-compatible API, we can also use ordinary AWS S3 as a cross-region staging source for Runpod pods.

Current AWS staging bucket:
- `parameter-golf-staging-094651608775`

Current staged objects:
- leader-stack `sp1024` archive:
  - `s3://parameter-golf-staging-094651608775/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst`
- JEPA isolation `byte260` archive:
  - `s3://parameter-golf-staging-094651608775/data/archives/fineweb10B_byte260__fineweb_pure_byte_260.tar.zst`
- `byte260` archive manifest:
  - `s3://parameter-golf-staging-094651608775/data/archives/fineweb10B_byte260__fineweb_pure_byte_260.manifest.tsv`
- leader-stack code bundle:
  - `s3://parameter-golf-staging-094651608775/staging_bundles/parameter-golf-leader-stack-jepa-bundle.tar`
- leader-stack bundle manifest:
  - `s3://parameter-golf-staging-094651608775/staging_bundles/parameter-golf-leader-stack-jepa-bundle.MANIFEST.txt`
- JEPA isolation bundle:
  - `s3://parameter-golf-staging-094651608775/staging_bundles/parameter-golf-jepa-iso-bundle.tar`
- JEPA isolation bundle manifest:
  - `s3://parameter-golf-staging-094651608775/staging_bundles/parameter-golf-jepa-iso-bundle.MANIFEST.txt`
- Scylla-v2 exact dataset archive:
  - `s3://parameter-golf-staging-094651608775/data/archives/scylla_v2_cap0_fullbyte.tar.zst`
- Scylla-v2 exact dataset manifest:
  - `s3://parameter-golf-staging-094651608775/data/archives/scylla_v2_cap0_fullbyte.manifest.tsv`
- Scylla pre-quant TTT bundle:
  - `s3://parameter-golf-staging-094651608775/staging_bundles/parameter-golf-scylla-v2-prequant-bundle.tar`
- Scylla pre-quant TTT bundle manifest:
  - `s3://parameter-golf-staging-094651608775/staging_bundles/parameter-golf-scylla-v2-prequant-bundle.MANIFEST.txt`

Validated behavior:
- uploaded the leader-stack bundle successfully
- downloaded that bundle directly onto a Runpod pod in `AP-IN-1` using a presigned URL and `curl`
- uploaded the full `sp1024` archive to ordinary AWS S3
- uploaded the reusable `byte260` archive and JEPA isolation bundle to ordinary AWS S3

Operational implication:
- a Runpod pod does not need `aws` installed to fetch from ordinary AWS S3 if we generate presigned URLs locally
- this is now the preferred durable staging path for recurring assets
- it is also a viable fallback for regions like `AP-IN-1` that are not in Runpod's S3-compatible datacenter list

Measured `512MB` ranged download throughput from AWS S3 to cheap CPU pods using a presigned URL:
- `US-GA-2` (`runpod/parameter-golf:latest` CPU pod): `time_total=14.402087`, `speed_download=37277299` bytes/s, about `35.5 MB/s`
- `US-GA-2` prep pod (`runpod/base:1.0.2-ubuntu2204`): `time_total=8.935649`, `speed_download=60081915` bytes/s, about `57.3 MB/s`
- `US-IL-1` (`runpod/parameter-golf:latest` CPU pod): `time_total=11.098344`, `speed_download=48373965` bytes/s, about `46.1 MB/s`
- `AP-IN-1` (`runpod/parameter-golf:latest` cheap staging pod): `time_total=103.007029`, `speed_download=5211983` bytes/s, about `5.0 MB/s`

Current conclusion:
- `AP-IN-1` is high-risk for data-heavy staging. Avoid renting `8xH100` there unless the archive is already local.
- `US-GA-2` and `US-IL-1` are good enough to treat direct AWS S3 -> Runpod as a viable staging path.
- Canada CPU capacity for this test was unavailable in `CA-MTL-1`, `CA-MTL-2`, and `CA-MTL-3` at probe time, so no Canada throughput number is recorded yet.
- for future `8xH100` promotions, prefer `US-GA-2` or `US-IL-1` if capacity exists before considering weaker regions like `AP-IN-1`

Recommended restore flow from AWS S3:
1. Generate a presigned URL locally for the asset you want.
2. On the target cheap pod or run pod, download the archive or bundle with `curl -L --fail`.
3. For bundle tarballs, extract into `/workspace/parameter-golf`.
4. For dataset archives, stage into `/tmp` with [stage_runpod_data_archive.sh](/Users/simon/Code/parameter-golf/stage_runpod_data_archive.sh).
5. Run [verify_runpod_data_ready.sh](/Users/simon/Code/parameter-golf/verify_runpod_data_ready.sh) before any `torchrun`.

Important tar hygiene:
- our locally built tarballs may contain macOS `._*` sidecar files and ownership metadata
- extract with `tar --no-same-owner`
- when needed, exclude `._*` and `.DS_Store`
- warnings about unknown `LIBARCHIVE.xattr.*` keywords are expected and not fatal

Example: download the leader-stack `sp1024` archive onto a pod:

```bash
curl -L --fail "$PRESIGNED_URL" -o /workspace/parameter-golf/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst
```

Example: stage that archive into `/tmp`:

```bash
cd /workspace/parameter-golf
bash ./stage_runpod_data_archive.sh \
  /workspace/parameter-golf/data/archives/fineweb10B_sp1024__fineweb_1024_bpe.tar.zst \
  /tmp/parameter-golf-data
```

Example: download and unpack the JEPA isolation bundle:

```bash
curl -L --fail "$PRESIGNED_URL" -o /workspace/parameter-golf-jepa-iso-bundle.tar
tar -xf /workspace/parameter-golf-jepa-iso-bundle.tar -C /workspace/parameter-golf
```

Example: Scylla-style fast `8xH100` restore:

```bash
curl -L --fail "$SCYLLA_BUNDLE_URL" -o /workspace/parameter-golf-scylla-bundle.tar
curl -L --fail "$SCYLLA_ARCHIVE_URL" -o /workspace/parameter-golf/data/archives/scylla_v2_cap0_fullbyte.tar.zst

cd /workspace/parameter-golf
tar --no-same-owner --exclude='._*' -xf /workspace/parameter-golf-scylla-bundle.tar -C /workspace/parameter-golf

rm -rf /tmp/parameter-golf-scylla-v2
mkdir -p /tmp/parameter-golf-scylla-v2
tar --no-same-owner --exclude='._*' --exclude='.DS_Store' -I zstd \
  -xf /workspace/parameter-golf/data/archives/scylla_v2_cap0_fullbyte.tar.zst \
  -C /tmp/parameter-golf-scylla-v2

bash ./verify_runpod_data_ready.sh \
  /tmp/parameter-golf-scylla-v2/datasets/fineweb10B_scylla_v2_cap0_fullbyte \
  /tmp/parameter-golf-scylla-v2/tokenizers/scylla_v2_cap0_fullbyte.meta.npz \
  11 1
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

- Prefer the standard template-backed pod flow over the suspended custom image path.
- Never trust `/tmp` as persistent storage.
- Never launch before preflight passes.
- For multi-GPU runs, never launch until the dataset archive and code bundle are already present on the pod and the staged `/tmp` paths have passed preflight.
- Always pull logs and artifacts before stopping a pod.
- If a new image/template repeatedly stays `RUNNING` without becoming SSH-ready, do not adopt it as the default path.
- Use `DATA_ROOT_MODE=tmp` for serious throughput or record-style runs.
- Use `USE_COMPILE=0` for cheap smoke/debug runs; keep compile on for real runs unless debugging compile itself.

Practical note:
- direct S3 -> expensive pod is acceptable when the archive is already staged in S3, the region has decent throughput, and the archive is modest enough to restore quickly
- do not require a prep pod when it adds complexity without reducing launch risk

## Automated 8xH100 Pod Hunting

[`watch_runpod_8xh100_launch.sh`](/Users/simon/Code/parameter-golf/watch_runpod_8xh100_launch.sh) polls the RunPod REST API to grab an 8×H100 pod as soon as capacity appears. It uses an explicit datacenter allowlist that excludes `AP-IN-1` (and all other `AP-IN-*` regions) due to the poor S3 download throughput measured above (~5 MB/s vs ~35–57 MB/s in US regions).

Key behavior:
- Retries every `INTERVAL_SECONDS` (default 30s) until a pod is successfully created
- Uses `dataCenterPriority: "availability"` so it takes the first available slot in any allowed region
- Logs attempts and the success response to `runpod_artifacts/launch_watch/`

Customizable env vars:
- `POD_NAME` — pod name (default `pg-leader-jepa-8x`)
- `IMAGE_NAME` — container image (default `runpod/parameter-golf:latest`)
- `GPU_COUNT` — number of GPUs (default 8)
- `GPU_TYPE` — GPU type (default `NVIDIA H100 80GB HBM3`)
- `INTERVAL_SECONDS` — polling interval (default 30)

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
