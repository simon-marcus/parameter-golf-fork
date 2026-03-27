# RunPod Image Plan

This directory defines our pinned base image and the checks we want before any expensive RunPod launch.

## Goals

- Eliminate repeated pod bootstrap drift.
- Make missing deps like `zstandard` impossible to miss.
- Standardize `/workspace` vs `/tmp` usage.
- Make expensive pods nearly launch-ready.

## What Goes Where

Use `/workspace` for:
- the git checkout
- persistent dataset cache
- tokenizer cache
- pulled logs and artifacts

Use `/tmp/parameter-golf-data` for:
- `DATA_PATH`
- `TOKENIZER_PATH`
- throughput-sensitive training and eval runs

Why:
- `/workspace` survives pod stop/start and is the right place for durable state
- `/tmp` is the fast local disk path we want for actual timed runs

## Build

```bash
cd /workspace/parameter-golf
bash ops/runpod_image/build_image.sh ghcr.io/<org>/parameter-golf-runpod:cuda128-torch210
```

## Validate Locally Or On A Pod

```bash
bash ops/runpod_image/validate_image.sh
```

This checks:
- Python imports
- `torch` availability
- `zstandard` availability
- `openssh-server` availability
- helper tool presence (`rsync`, `tmux`, `jq`, `pigz`)
- writable `/workspace` and `/tmp`
- presence of the report script

## Recommended Run Flow

1. Start pod from our template/image.
2. Sync repo code only.
3. Run `bash /opt/parameter-golf/validate_image.sh` once on the fresh pod.
4. Run `bash ops/runpod_image/validate_image.sh` from the repo checkout.
5. Keep persistent repo/data/tokenizer state under `/workspace`.
6. Stage dataset/tokenizer from `/workspace` into `/tmp/parameter-golf-data`.
7. Run the repo preflight.
8. Launch the actual job.
9. Pull logs/artifacts.
10. Only then stop the pod.

## Cheap Pod To Expensive Pod Flow

Best practice for record-style runs:

1. Create the persistent volume in the target datacenter.
2. Attach that volume to a cheap prep pod in the same datacenter.
3. On the cheap pod, populate under `/workspace/parameter-golf/`:
   - repo checkout
   - dataset cache
   - tokenizer cache
4. Stop the cheap pod.
5. Attach the same volume to the expensive pod in the same datacenter.
6. On the expensive pod, copy from `/workspace/...` to `/tmp/parameter-golf-data/...`.
7. Run preflight and then the timed job.

This is the main way to avoid wasting `8xH100` time on downloads and setup.

## RunPod Notes

- This image has a slow first cold start because it is large.
- Reused pods are the intended operating model.
- Direct TCP SSH/SCP/rsync are expected to work because the image starts
  `sshd` and injects `PUBLIC_KEY` on boot.
- For `runpodctl pod create`, pass disk/volume flags explicitly even when using the template:

```bash
runpodctl pod create \
  --template-id <template-id> \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --cloud-type SECURE \
  --ssh \
  --container-disk-in-gb 50 \
  --volume-in-gb 50
```

## Template Defaults

See [TEMPLATE_CHECKLIST.md](/Users/simon/Code/parameter-golf/ops/runpod_image/TEMPLATE_CHECKLIST.md).
