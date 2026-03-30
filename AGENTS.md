# Parameter Golf — Project Guide

## What This Is

This is an entry for the [OpenAI Parameter Golf challenge](https://github.com/openai/parameter-golf): train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8×H100s. Scored by validation bits-per-byte (BPB) on FineWeb — lower is better.

We use an **autoresearch loop** that runs on RunPod GPU pods: Codex (`Codex -p`) proposes surgical edits to `train_gpt.py`, training runs, results are evaluated, and improvements are kept.

## Key Files

| File | Purpose |
|------|---------|
| `train_gpt.py` | The training script (from OpenAI). This is what gets modified by experiments. |
| `autoresearch.py` | The autoresearch loop — orchestrates propose→train→evaluate→keep/revert. |
| `program.md` | High-level research directives that guide Codex's proposals. |
| `run_lane.sh` | Launches a specific research lane (core, storage, eval_time, etc). |
| `pod.sh` | RunPod pod management: create, ssh, sync, pull, stop, destroy. |
| `EXPERIMENTS.md` | Human-readable experiment tracker and findings log. |
| `.env.local` | Contains `RUNPOD_API_KEY` and `ANTHROPIC_API_KEY` (not in git). |
| `.pod_id` | ID of the original RunPod pod (not in git). |
| `.lane_pods` | `host:port:label` entries for lane pods (not in git). |

## Architecture

### Autoresearch Loop (`autoresearch.py`)

Each experiment cycle:
1. **Propose**: `Codex -p` reads `train_gpt.py`, makes a surgical edit, returns a description
2. **Train**: `torchrun --standalone --nproc_per_node=$GPUS train_gpt.py`
3. **Evaluate**: Parse `val_bpb` (post-int8-quantization) and artifact size from output
4. **Keep/Revert**: If BPB improved and under 16MB, keep. Otherwise revert to previous code.
5. **Snapshot**: Save code + rationale to `autoresearch/experiments/NNNN/`

**Pipelining**: While training experiment N, speculatively proposes experiment N+1. If N is reverted (most common), the speculative proposal is ready immediately. If N is kept, the stale proposal is discarded and re-proposed.

### Research Lanes

Multiple lanes can run in parallel on separate pods, each exploring a different axis:

| Lane | Focus | Key env vars |
|------|-------|-------------|
| `core_discovery` | Architecture, hyperparameters, training tricks | Default settings |
| `storage_discovery` | Compression, quantization, artifact size reduction | `MAX_QUANTIZATION_GAP`, `STORAGE_MAX_REGRESSION` |
| `eval_time_discovery` | Eval-time compute tricks (using the 10-min eval window) | `MAX_EVAL_TIME_MS` |
| `core_promotion` | Validate discoveries at full scale (8×GPU, 10 min) | `EXPERIMENT_SECONDS=600` |

Each lane gets its own namespace directory: `autoresearch/<namespace>/` with independent history, experiments, and best code.

Launch a lane: `BACKGROUND=1 bash ./run_lane.sh <lane_key>`

### Key Metrics

- **val_bpb**: Post-int8-quantization bits-per-byte (THE competition metric)
- **prequant_bpb**: Pre-quantization BPB (shows raw model quality)
- **quantization_gap**: val_bpb - prequant_bpb (how much compression hurts)
- **artifact_bytes**: Total submission size (must be < 16,000,000)
- **model_params**: Parameter count
- **last_step**: Training steps completed in the time budget

## RunPod Infrastructure

### Pod Management (`pod.sh`)

```bash
./pod.sh create [gpus]   # Spin up H100 pod (original pod, saved to .pod_id)
./pod.sh status          # Show pod info
./pod.sh ssh             # Connect to original pod
./pod.sh sync            # Push local files to original pod
./pod.sh pull            # Pull experiment results from ALL pods (original + lanes)
./pod.sh stop            # Pause original pod (keeps volume)
./pod.sh destroy         # Terminate original pod
```

### Current Pods

Pod details are in `.pod_id` (original) and `.lane_pods` (lanes). To check current pods:

```bash
export RUNPOD_API_KEY=$(grep RUNPOD_API_KEY .env.local | cut -d= -f2)
runpodctl pod list -o json
```

### SSH to any pod

```bash
ssh -o StrictHostKeyChecking=no root@<HOST> -p <PORT>
```

Pods land in `/workspace/`. The repo is at `/workspace/parameter-golf/`.

### Setting Up a New Pod

Recommended pod path:
- Use the standard template-backed RunPod flow reflected in [`pod.sh`](/Users/simon/Code/parameter-golf/pod.sh)
- Do not treat the custom image/template path as preferred right now; it is suspended as a default due to cold-start pods that stayed `RUNNING` without becoming SSH-ready

Pod creation rule:
- Even when using the template, pass `--container-disk-in-gb 50 --volume-in-gb 50` explicitly.
- Use the direct TCP SSH endpoint from `.ssh.ip` / `.ssh.port` for automation.
- Follow `RUNPOD.md` as the detailed runbook.

Example `1xH100` smoke pod:
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

Example `8xH100` record pod:
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
3. Use the direct TCP SSH endpoint from `.ssh.ip` and `.ssh.port`
4. Validate the image on the fresh pod:
   ```bash
   ssh root@HOST -p PORT "bash /opt/parameter-golf/validate_image.sh"
   ```
5. Sync the repo with a narrow exclude set:
   ```bash
   rsync -avz -e "ssh -p PORT -o StrictHostKeyChecking=no" \
     --exclude .git \
     --exclude __pycache__ \
     --exclude .DS_Store \
     --exclude .venv* \
     --exclude .Codex \
     --exclude .gstack \
     --exclude data/datasets \
     --exclude data/tokenizers \
     --exclude runpod_artifacts \
     --exclude modal_logs \
     /path/to/parameter-golf/ root@HOST:/workspace/parameter-golf/
   ```
6. On the pod:
   ```bash
   cd /workspace/parameter-golf
   mkdir -p logs
   bash ops/runpod_image/validate_image.sh
   ```
7. Run repo preflight, then the actual job

### RunPod Rules For Agents

- Prefer the standard template-backed pod flow over the suspended custom image path.
  If the custom image/template is being tested again, treat it as experimental until SSH-ready startup is proven reliable.
- Never launch an expensive run before preflight passes.
  Use `./verify_runpod_data_ready.sh "$DATA_PATH" "$TOKENIZER_PATH"` or a launcher that already calls it.
- Do not download dataset shards on an `8xH100` pod by default.
  Preferred flow: prepare data on a cheap pod or persistent volume, then stage to `/tmp/parameter-golf-data` on the expensive pod.
- Network volumes are datacenter-locked.
  Cheap prep pod and expensive run pod must be in the same datacenter if they share a persistent volume.
- Use `DATA_ROOT_MODE=tmp` for serious timing or record-style runs.
  `/workspace` is acceptable for bootstrap and inspection, not for throughput-sensitive comparisons.
- Use `USE_COMPILE=0` for `1xH100` smoke tests and debugging.
  Full runs should normally keep `USE_COMPILE=1`.
- If preflight fails, fix staging or data integrity first.
  Do not blindly rerun `torchrun`.
- Always pull logs and artifacts before stopping a pod.
  If you stop first, assume artifact recovery may fail.
- Direct TCP SSH/SCP/rsync are part of the contract now.
  If a new image/template breaks direct SSH or `rsync`, do not adopt it.
- If a pod sits in `RUNNING` / `initializing` without becoming SSH-ready, treat that as an image/template regression and stop using it as the default path.

### Recommended Run Sequences

Cheap `1xH100` smoke test:

```bash
ssh root@HOST -p PORT "bash /opt/parameter-golf/validate_image.sh"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ops/runpod_image/validate_image.sh"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ./verify_runpod_data_ready.sh /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ./setup_local_parity_data_runpod.sh"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ./verify_runpod_data_ready.sh /tmp/parameter-golf-data/datasets/fineweb10B_sp1024 /tmp/parameter-golf-data/tokenizers/fineweb_1024_bpe.model"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && USE_COMPILE=0 DATA_ROOT_MODE=tmp NPROC_PER_NODE=1 SCREEN_SECONDS=180 bash ./launch_leadercore_screen_runpod.sh base"
```

Full `8xH100` timing or record-style run:

```bash
ssh root@HOST -p PORT "bash /opt/parameter-golf/validate_image.sh"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ops/runpod_image/validate_image.sh"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ./verify_runpod_data_ready.sh /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ./setup_local_parity_data_runpod.sh"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && bash ./verify_runpod_data_ready.sh /tmp/parameter-golf-data/datasets/fineweb10B_sp1024 /tmp/parameter-golf-data/tokenizers/fineweb_1024_bpe.model"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && DATA_ROOT_MODE=tmp bash ./launch_leadercore_ablation_runpod.sh base"
```

### Modal 8xH100 Runs

Use the image-root dataset path for serious timing runs on Modal. In our March 29, 2026 A/B smoke test with the official baseline, `image` mode ran at about `45.5ms/step` while `tmp` mode ran at about `46.9ms/step`, so Modal does not show the same `/workspace` vs `/tmp` penalty we saw on Runpod. The launcher still supports `/tmp` staging for explicit comparison, and it verifies shard/tokenizer integrity with the same preflight used on Runpod.

Prereqs:
- Modal CLI and Python package installed in `.venv-modal`
- Active Modal profile (`modal profile current`)
- `HF_TOKEN` available via env or `.env.local` so the image build can fetch the FineWeb export

Recommended sequence:

```bash
source .venv-modal/bin/activate
modal profile current

# 8xH100 baseline smoke on Modal
modal run modal_record_candidate_run.py \
  --source records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
  --run-name baseline_modal_image_smoke_8x \
  --output-dir records/track_10min_16mb/2026-03-29_BaselineModalImageSmoke \
  --max-wallclock-seconds 60 \
  --val-loss-every 200 \
  --train-log-every 50 \
  --data-root-mode image

# 8xH100 full 10-minute baseline parity check on Modal
modal run modal_record_candidate_run.py \
  --source records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
  --run-name baseline_modal_image_8x \
  --output-dir records/track_10min_16mb/2026-03-29_BaselineModalImage \
  --max-wallclock-seconds 600 \
  --val-loss-every 200 \
  --train-log-every 50 \
  --data-root-mode image

# 8xH100 candidate run on Modal
modal run modal_record_candidate_run.py \
  --source /absolute/path/to/train_gpt.py \
  --run-name candidate_modal_image_8x \
  --output-dir records/track_10min_16mb/<run_name> \
  --max-wallclock-seconds 600 \
  --val-loss-every 0 \
  --train-log-every 50 \
  --data-root-mode image
```

Notes:
- Keep `--data-root-mode image` for normal Modal record runs. Use `tmp` only for explicit A/B checks.
- `modal_record_candidate_run.py` always runs `verify_runpod_data_ready.sh` before `torchrun`; in `tmp` mode it also copies `/root/parameter-golf/data` into `/tmp/parameter-golf-data` first.
- Logs and a `submission.json` snapshot are written to the local `--output-dir` after the remote run completes.

Optional on-pod dataset download when you intentionally want it:

```bash
RUNPOD_DOWNLOAD_DATA=1 bash /workspace/parameter-golf/setup_runpod.sh
```

### Syncing Changes to Running Pods

When you modify `autoresearch.py`, `program.md`, or other files locally:

```bash
# Sync to a specific pod
rsync -avz -e "ssh -p PORT -o StrictHostKeyChecking=no" \
  --exclude .git \
  --exclude __pycache__ \
  --exclude .DS_Store \
  --exclude .venv* \
  --exclude .Codex \
  --exclude .gstack \
  --exclude data/datasets \
  --exclude data/tokenizers \
  --exclude runpod_artifacts \
  --exclude modal_logs \
  autoresearch.py program.md \
  root@HOST:/workspace/parameter-golf/

# Kill and restart the lane
ssh root@HOST -p PORT "kill \$(ps aux | grep 'python3 autoresearch' | grep -v grep | awk '{print \$2}') 2>/dev/null"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && PYTHONUNBUFFERED=1 BACKGROUND=1 bash ./run_lane.sh <lane_key>"
```

### Monitoring

```bash
# Tail a lane's output
ssh root@HOST -p PORT "tail -f /workspace/parameter-golf/autoresearch/<namespace>/autoresearch.out"

# Check all lanes at once
for HP in "HOST1:PORT1:lane1" "HOST2:PORT2:lane2"; do
  HOST=$(echo $HP | cut -d: -f1); PORT=$(echo $HP | cut -d: -f2); LANE=$(echo $HP | cut -d: -f3)
  echo "=== $LANE ==="; ssh -o StrictHostKeyChecking=no root@$HOST -p $PORT "tail -10 /workspace/parameter-golf/autoresearch/$LANE/autoresearch.out" 2>&1; echo ""
done
```

### Pulling Results

`./pod.sh pull` syncs from all pods (original + `.lane_pods`). It pulls:
- `autoresearch/<namespace>/experiments/NNNN/README.md` — rationale + results
- `autoresearch/<namespace>/experiments/NNNN/train_gpt.py` — code snapshot
- `autoresearch/<namespace>/history.jsonl` — machine-readable history
- `autoresearch/<namespace>/train_gpt.best.py` — current best code

Training logs (`*.log`) are excluded to keep git clean.

### Cost

- 1×H100 SXM: ~$2.69/hr (secure cloud)
- Stopped pods keep their volume for a tiny hourly cost; use this for warmed reusable setups.
- Always `./pod.sh stop` or destroy pods when not in use
- Check balance: `runpodctl user`

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EXPERIMENT_SECONDS` | 180 | Training time budget per experiment |
| `GPUS` | 1 | Number of GPUs for training |
| `AUTORESEARCH_MODEL` | opus | Codex model for proposals |
| `CLAUDE_EFFORT` | high | Thinking effort level |
| `MAX_EXPERIMENTS` | 100 | Max experiments before stopping |
| `BASELINE_BPB` | 0 | Seed BPB for first comparison |
| `AUTORESEARCH_NAMESPACE` | "" | Subdirectory for lane isolation |
| `AUTORESEARCH_LANE` | core | Lane type (core/storage/eval_time) |
| `AUTORESEARCH_STAGE` | discovery | Stage (discovery/promotion) |
| `PYTHONUNBUFFERED` | — | Set to 1 for real-time log output |
