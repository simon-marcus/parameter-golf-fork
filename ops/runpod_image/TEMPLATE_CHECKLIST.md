# RunPod Template Checklist

## Image

- Use our pinned image, not a mutable public base.
- Tag by CUDA + Torch version, for example `cuda128-torch210`.
- Rebuild only when we intentionally change the stack.

## Pod Shape

- `1xH100` template for smoke tests and reduced ablations.
- `8xH100 SXM` template for record-style runs.
- Container disk: at least `50GB`.
- Volume: at least `50GB`.
- On current `runpodctl`, still pass `--container-disk-in-gb 50 --volume-in-gb 50`
  at pod creation time even when using the template. We observed template-backed
  pod creation silently dropping these values otherwise.

## Preconfigured Env

- `HF_HOME=/workspace/.cache/huggingface`
- `PIP_DISABLE_PIP_VERSION_CHECK=1`
- `PYTHONUNBUFFERED=1`

Optional but useful:
- `RUNPOD_DOWNLOAD_DATA=0`
- `USE_COMPILE=1`

## Volume Layout

- Repo checkout under `/workspace/parameter-golf`
- Persistent dataset cache under `/workspace/parameter-golf/data/datasets`
- Persistent tokenizer cache under `/workspace/parameter-golf/data/tokenizers`
- Logs under `/workspace/parameter-golf/logs`

## Required Validation Before Expensive Launch

1. `bash ops/runpod_image/validate_image.sh`
2. `bash ./verify_runpod_data_ready.sh "$WORKSPACE_DATA_PATH" "$WORKSPACE_TOKENIZER_PATH"`
3. `bash ./setup_local_parity_data_runpod.sh`
4. `bash ./verify_runpod_data_ready.sh "$TMP_DATA_PATH" "$TMP_TOKENIZER_PATH"`

Do not launch `torchrun` before all four pass.

## Known Failure Modes To Prevent

- Missing `zstandard` causing `zlib` fallback and oversized artifacts
- Missing `/tmp/parameter-golf-data/tokenizers/...` on resumed pods
- Overlay-copy corruption when staging into `/tmp`
- Missing `logs/` directory on fresh pods
- Pulling artifacts after stopping the pod
- Container immediately exiting because the image default command does not stay
  alive under RunPod
- Direct TCP SSH not working because the image lacks `openssh-server` or does
  not start `sshd` on boot
- Spending expensive pod time on setup when the right pattern is cheap-pod prep
  onto a datacenter-local persistent volume

## Validation Matrix

### Cheap checks

- Image validation script passes
- `python3 -c 'import torch, zstandard, sentencepiece, huggingface_hub'`
- `nvidia-smi` works
- `/workspace` writable
- `/tmp` writable

### 1x smoke check

- repo preflight passes
- one reduced run starts and reaches `step:10`
- packaging path logs the expected compressor
- data path in logs is `/tmp/parameter-golf-data/...`

### 8x record check

- repo preflight passes
- all ranks launch
- first train step appears
- final packaging size is legal
- logs and artifact pull succeed before stop
