"""
Flexible Modal launcher for 8xH100 Parameter Golf candidate runs.

Examples:
  python modal_record_candidate_run.py \
    --source autoresearch/core_record_discovery/train_gpt.best.py \
    --run-name core_record_candidate \
    --output-dir records/track_10min_16mb/2026-03-19_CoreRecordCandidate \
    --max-wallclock-seconds 600 \
    --data-root-mode image

  python modal_record_candidate_run.py \
    --source autoresearch/core_record_discovery/train_gpt.best.py \
    --run-name core_record_candidate_20k \
    --output-dir records/non_record/2026-03-19_CoreRecordCandidate_full \
    --max-wallclock-seconds 0 \
    --val-loss-every 200 \
    --data-root-mode image
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import argparse

import modal


ROOT = Path(__file__).resolve().parent
ENV_LOCAL = ROOT / ".env.local"
LOCAL_TM0054_ROOT = Path("/Users/simon/Code/parameter-golf-local/tm0054_full_export")


def load_env_local() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_LOCAL.exists():
        return values
    for line in ENV_LOCAL.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip()
    return values


ENV_LOCAL_VALUES = load_env_local()
HF_TOKEN = os.environ.get("HF_TOKEN") or ENV_LOCAL_VALUES.get("HF_TOKEN")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5",
        "numpy",
        "sentencepiece",
        "huggingface_hub",
    )
    .env({"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {})
    .add_local_dir(
        str(ROOT / "data"),
        remote_path="/root/parameter-golf/data",
        copy=True,
    )
    .add_local_file(
        str(ROOT / "setup_local_parity_data_runpod.sh"),
        remote_path="/root/parameter-golf/setup_local_parity_data_runpod.sh",
        copy=True,
    )
    .add_local_file(
        str(ROOT / "verify_runpod_data_ready.sh"),
        remote_path="/root/parameter-golf/verify_runpod_data_ready.sh",
        copy=True,
    )
    .add_local_dir(
        str(LOCAL_TM0054_ROOT),
        remote_path="/root/parameter-golf/tm0054_bundle",
        copy=True,
    )
    .run_commands(
        "mkdir -p /root/parameter-golf",
        "chmod +x /root/parameter-golf/setup_local_parity_data_runpod.sh /root/parameter-golf/verify_runpod_data_ready.sh",
        "cd /root/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024",
    )
)

app = modal.App("parameter-golf-record-candidate", image=image)


@app.function(
    gpu="H100:8",
    timeout=2 * 60 * 60,
    memory=65536,
)
def run_candidate(
    train_script_text: str,
    run_name: str,
    max_wallclock_seconds: int,
    val_loss_every: int,
    train_log_every: int,
    data_root_mode: str,
    candidate_bundle: str,
):
    work_dir = "/root/parameter-golf"
    train_script = os.path.join(work_dir, "train_gpt.py")
    image_data_root = os.path.join(work_dir, "data")
    local_data_root = "/tmp/parameter-golf-data"

    with open(train_script, "w") as f:
        f.write(train_script_text)

    if candidate_bundle == "baseline":
        if data_root_mode == "tmp":
            subprocess.run(
                [
                    "bash",
                    os.path.join(work_dir, "setup_local_parity_data_runpod.sh"),
                    image_data_root,
                    local_data_root,
                    "fineweb10B_sp1024",
                    "fineweb_1024_bpe",
                ],
                check=True,
                cwd=work_dir,
            )
            data_root = local_data_root
        elif data_root_mode == "image":
            data_root = image_data_root
        else:
            raise ValueError(f"Unsupported data_root_mode={data_root_mode!r}; expected 'tmp' or 'image'")

        data_path = os.path.join(data_root, "datasets", "fineweb10B_sp1024")
        tokenizer_path = os.path.join(data_root, "tokenizers", "fineweb_1024_bpe.model")
        tokenizer_meta_path = os.path.join(data_root, "tokenizers", "fineweb_1024_bpe.meta.npz")
        vocab_size = "1024"
        expected_train_shards = "80"
        expected_val_shards = "1"
    elif candidate_bundle == "tm0054":
        if data_root_mode != "image":
            raise ValueError("candidate_bundle='tm0054' currently requires --data-root-mode image")
        data_root = "/root/parameter-golf/tm0054_bundle"
        data_path = os.path.join(data_root, "datasets", "fineweb10B_tm0054")
        tokenizer_path = os.path.join(data_root, "tokenizers", "candidate.vocab")
        tokenizer_meta_path = os.path.join(data_root, "tokenizers", "candidate.meta.npz")
        vocab_size = "998"
        expected_train_shards = "10"
        expected_val_shards = "1"
    else:
        raise ValueError(f"Unsupported candidate_bundle={candidate_bundle!r}")

    if candidate_bundle == "baseline":
        verify_tokenizer_path = tokenizer_path
    else:
        verify_tokenizer_path = tokenizer_meta_path

    subprocess.run(
        [
            "bash",
            os.path.join(work_dir, "verify_runpod_data_ready.sh"),
            data_path,
            verify_tokenizer_path,
            expected_train_shards,
            expected_val_shards,
        ],
        check=True,
        cwd=work_dir,
    )

    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": run_name,
            "DATA_PATH": data_path,
            "TOKENIZER_PATH": tokenizer_path,
            "TOKENIZER_META_PATH": tokenizer_meta_path,
            "VOCAB_SIZE": vocab_size,
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
            "TRAIN_LOG_EVERY": str(train_log_every),
            "VAL_LOSS_EVERY": str(val_loss_every),
            "NCCL_IB_DISABLE": "1",
            "DATA_ROOT_MODE": data_root_mode,
        }
    )

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        train_script,
    ]

    print(f"STARTING CANDIDATE RUN: {run_name}")
    print(
        f"data_root_mode={data_root_mode} candidate_bundle={candidate_bundle} "
        f"data_path={data_path} tokenizer_path={tokenizer_path}"
    )
    print(f"Command: {' '.join(cmd)}")

    log_lines: list[str] = []
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=work_dir,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        print(line)
        log_lines.append(line)

    process.wait()
    elapsed = time.time() - start_time
    log_text = "\n".join(log_lines)

    metrics: dict[str, float | int] = {}
    for line in log_lines:
        if "final_int8_zlib_roundtrip_exact" in line and "val_bpb" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["val_bpb"] = float(part.split(":")[1])
                if part.startswith("val_loss:"):
                    metrics["val_loss"] = float(part.split(":")[1])
        if "final_val_loss:" in line:
            for part in line.split():
                if part.startswith("final_val_loss:"):
                    metrics["final_val_loss"] = float(part.split(":")[1])
                if part.startswith("final_val_bpb:"):
                    metrics["final_val_bpb"] = float(part.split(":")[1])
        if "int8+zlib total_submission_bytes:" in line:
            for part in line.split():
                if part.startswith("total_submission_bytes:"):
                    metrics["bytes_total"] = int(part.split(":")[1])
        if "code_bytes:" in line and "int8+zlib" in line:
            for part in line.split():
                if part.startswith("code_bytes:"):
                    metrics["bytes_code"] = int(part.split(":")[1])
        if "step_avg:" in line:
            for part in line.split():
                if part.startswith("step_avg:"):
                    raw = part.split(":")[1].rstrip("ms")
                    try:
                        metrics["step_avg_ms"] = float(raw)
                    except ValueError:
                        pass

    return {
        "log": log_text,
        "metrics": metrics,
        "exit_code": process.returncode,
        "elapsed_seconds": elapsed,
    }


@app.local_entrypoint()
def main(
    source: str,
    run_name: str = "record_candidate",
    output_dir: str = "records/track_10min_16mb/record_candidate",
    max_wallclock_seconds: int = 600,
    val_loss_every: int = 0,
    train_log_every: int = 50,
    data_root_mode: str = "image",
    candidate_bundle: str = "baseline",
):
    source_path = (ROOT / source).resolve() if not os.path.isabs(source) else Path(source)
    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    train_script_text = source_path.read_text()
    result = run_candidate.remote(
        train_script_text=train_script_text,
        run_name=run_name,
        max_wallclock_seconds=max_wallclock_seconds,
        val_loss_every=val_loss_every,
        train_log_every=train_log_every,
        data_root_mode=data_root_mode,
        candidate_bundle=candidate_bundle,
    )

    out_dir = (ROOT / output_dir).resolve() if not os.path.isabs(output_dir) else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "train.log").write_text(result["log"])
    shutil.copy(source_path, out_dir / "train_gpt.py")

    submission = {
        "name": run_name,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(source_path),
        "exit_code": result["exit_code"],
        "elapsed_seconds": result["elapsed_seconds"],
        **result.get("metrics", {}),
    }
    with open(out_dir / "submission.json", "w") as f:
        json.dump(submission, f, indent=2)

    print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-name", default="record_candidate")
    parser.add_argument("--output-dir", default="records/track_10min_16mb/record_candidate")
    parser.add_argument("--max-wallclock-seconds", type=int, default=600)
    parser.add_argument("--val-loss-every", type=int, default=0)
    parser.add_argument("--train-log-every", type=int, default=50)
    parser.add_argument("--data-root-mode", choices=("tmp", "image"), default="image")
    parser.add_argument("--candidate-bundle", choices=("baseline", "tm0054"), default="baseline")
    args = parser.parse_args()
    main(
        source=args.source,
        run_name=args.run_name,
        output_dir=args.output_dir,
        max_wallclock_seconds=args.max_wallclock_seconds,
        val_loss_every=args.val_loss_every,
        train_log_every=args.train_log_every,
        data_root_mode=args.data_root_mode,
        candidate_bundle=args.candidate_bundle,
    )
