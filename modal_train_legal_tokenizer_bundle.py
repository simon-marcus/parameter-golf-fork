from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
ENV_LOCAL = ROOT / ".env.local"
LOCAL_BASELINE_ROOT = Path("/Users/simon/Code/parameter-golf-local/data")
LOCAL_TM0054_ROOT = Path("/Users/simon/Code/parameter-golf-local/tm0054_full_export")
LEGAL_TRAIN_SCRIPT = ROOT / "records" / "track_10min_16mb" / "2026-03-29_PPM_LOO_TokenizerBundle" / "train_gpt.py"


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
    .pip_install("torch>=2.5", "numpy", "sentencepiece", "tokenmonster", "huggingface_hub", "zstandard")
    .env({"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {})
    .add_local_file(LEGAL_TRAIN_SCRIPT, remote_path="/root/parameter-golf/train_gpt.py", copy=True)
    .add_local_dir(ROOT / "data", remote_path="/root/parameter-golf/data", copy=True)
    .add_local_dir(LOCAL_BASELINE_ROOT, remote_path="/root/parameter-golf/baseline_bundle", copy=True)
    .add_local_dir(LOCAL_TM0054_ROOT, remote_path="/root/parameter-golf/tm0054_bundle", copy=True)
    .run_commands("mkdir -p /root/parameter-golf/runs")
)

app = modal.App("parameter-golf-train-legal-tokenizer-bundle", image=image)


def parse_train_log(log_lines: list[str]) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    exact_prefixes = {
        "final_int6_roundtrip_exact": "roundtrip_postquant_val_bpb",
        "final_int6_sliding_window_exact": "sliding_postquant_val_bpb",
        "legal_ttt_exact": "legal_ttt_postquant_val_bpb",
        "ngram_two_pass_exact": "ngram_two_pass_postquant_val_bpb",
    }
    for line in log_lines:
        for prefix, key in exact_prefixes.items():
            if line.startswith(prefix):
                for part in line.split():
                    if part.startswith("val_bpb:"):
                        try:
                            metrics[key] = float(part.split(":")[1])
                        except ValueError:
                            pass
                        break
        if "model_params:" in line:
            try:
                metrics["model_params"] = int(line.split("model_params:")[1].strip())
            except Exception:
                pass
        if "Total submission size int8+zlib:" in line:
            try:
                metrics["artifact_bytes"] = int(line.split("Total submission size int8+zlib:")[1].split()[0])
            except Exception:
                pass
        if "step_avg:" in line:
            for part in line.split():
                if part.startswith("step_avg:"):
                    raw = part.split(":")[1].rstrip("ms")
                    try:
                        metrics["step_avg_ms"] = float(raw)
                    except ValueError:
                        pass
        if "step:" in line and "/" in line and "train_loss:" in line:
            for part in line.split():
                if part.startswith("step:"):
                    metrics["last_step"] = part.split(":")[1]
                    break
    return metrics


@app.function(gpu="H100", timeout=90 * 60, memory=32768)
def run_training(
    *,
    bundle_root: str,
    dataset_name: str,
    tokenizer_relpath: str,
    tokenizer_meta_relpath: str,
    vocab_size: int,
    run_name: str,
    max_wallclock_seconds: int,
    iterations: int,
    train_log_every: int,
    val_loss_every: int,
    seed: int,
) -> dict[str, object]:
    work_dir = "/root/parameter-golf"
    bundle_path = Path(bundle_root)
    data_path = bundle_path / "datasets" / dataset_name
    tokenizer_path = bundle_path / tokenizer_relpath
    tokenizer_meta_path = bundle_path / tokenizer_meta_relpath
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": run_name,
            "DATA_PATH": str(data_path),
            "TOKENIZER_PATH": str(tokenizer_path),
            "TOKENIZER_META_PATH": str(tokenizer_meta_path),
            "VOCAB_SIZE": str(vocab_size),
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
            "ITERATIONS": str(iterations),
            "TRAIN_LOG_EVERY": str(train_log_every),
            "VAL_LOSS_EVERY": str(val_loss_every),
            "SEED": str(seed),
            "NCCL_IB_DISABLE": "1",
            "NGRAM_ENABLED": "0",
            "TTT_ENABLED": "1",
            "TRAIN_SEQ_LEN": "2048",
            "EVAL_SEQ_LEN": "2048",
            "TRAIN_BATCH_TOKENS": "786432",
            "NUM_LAYERS": "11",
            "MLP_MULT": "3",
            "BIGRAM_VOCAB_SIZE": "1536",
            "XSA_LAST_N": "4",
            "SWA_ENABLED": "1",
            "SWA_EVERY": "50",
            "ROPE_DIMS": "16",
            "LN_SCALE": "1",
            "LATE_QAT_THRESHOLD": "0.15",
            "VE_ENABLED": "1",
            "VE_DIM": "128",
            "VE_LAYERS": "9,10",
            "MUON_WD": "0.04",
            "ADAM_WD": "0.04",
            "MATRIX_LR": "0.025",
            "SCALAR_LR": "0.025",
            "EMBED_LR": "0.6",
            "HEAD_LR": "0.008",
            "TIED_EMBED_LR": "0.035",
            "MUON_MOMENTUM": "0.99",
            "MUON_MOMENTUM_WARMUP_START": "0.92",
            "MUON_MOMENTUM_WARMUP_STEPS": "1500",
            "WARMDOWN_ITERS": "3500",
            "EVAL_STRIDE": "64",
            "TTT_LR": "0.002",
            "TTT_EPOCHS": "3",
            "TTT_CHUNK_TOKENS": "32768",
            "TTT_MOMENTUM": "0.9",
            "TTT_GRAD_CLIP": "1.0",
        }
    )
    cmd = ["torchrun", "--standalone", "--nproc_per_node=1", f"{work_dir}/train_gpt.py"]
    print("RUNNING:", " ".join(cmd))
    print("DATA_PATH:", data_path)
    print("TOKENIZER_PATH:", tokenizer_path)
    print("TOKENIZER_META_PATH:", tokenizer_meta_path)
    start = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=work_dir,
        env=env,
    )
    assert process.stdout is not None
    log_lines: list[str] = []
    for line in process.stdout:
        line = line.rstrip()
        print(line)
        log_lines.append(line)
    process.wait()
    elapsed = time.time() - start
    metrics = parse_train_log(log_lines)
    return {
        "exit_code": process.returncode,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "log": "\n".join(log_lines),
    }


@app.local_entrypoint()
def main(
    output_dir: str = "modal_logs/train_tm0054_legal_ladder",
    max_wallclock_seconds: int = 600,
    iterations: int = 9000,
    train_log_every: int = 500,
    val_loss_every: int = 4000,
    seed: int = 1337,
):
    configs = [
        {
            "label": "baseline_sp1024",
            "bundle_root": "/root/parameter-golf/baseline_bundle",
            "dataset_name": "fineweb10B_sp1024",
            "tokenizer_relpath": "tokenizers/fineweb_1024_bpe.model",
            "tokenizer_meta_relpath": "tokenizers/fineweb_1024_bpe.meta.npz",
            "vocab_size": 1024,
            "run_name": f"baseline_sp1024_legal_s{seed}",
        },
        {
            "label": "tm0054_candidate",
            "bundle_root": "/root/parameter-golf/tm0054_bundle",
            "dataset_name": "fineweb10B_tm0054",
            "tokenizer_relpath": "tokenizers/candidate.vocab",
            "tokenizer_meta_relpath": "tokenizers/candidate.meta.npz",
            "vocab_size": 998,
            "run_name": f"tm0054_candidate_legal_s{seed}",
        },
    ]
    out_dir = (ROOT / output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for cfg in configs:
        result = run_training.remote(
            bundle_root=cfg["bundle_root"],
            dataset_name=cfg["dataset_name"],
            tokenizer_relpath=cfg["tokenizer_relpath"],
            tokenizer_meta_relpath=cfg["tokenizer_meta_relpath"],
            vocab_size=cfg["vocab_size"],
            run_name=cfg["run_name"],
            max_wallclock_seconds=max_wallclock_seconds,
            iterations=iterations,
            train_log_every=train_log_every,
            val_loss_every=val_loss_every,
            seed=seed,
        )
        (out_dir / f"{cfg['label']}.log").write_text(result["log"], encoding="utf-8")
        rows.append(
            {
                "label": cfg["label"],
                "seed": seed,
                "bundle_root": cfg["bundle_root"],
                "dataset_name": cfg["dataset_name"],
                "tokenizer_relpath": cfg["tokenizer_relpath"],
                "tokenizer_meta_relpath": cfg["tokenizer_meta_relpath"],
                "vocab_size": cfg["vocab_size"],
                **result["metrics"],
                "exit_code": result["exit_code"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="modal_logs/train_tm0054_legal_ladder")
    parser.add_argument("--max-wallclock-seconds", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=9000)
    parser.add_argument("--train-log-every", type=int, default=500)
    parser.add_argument("--val-loss-every", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        max_wallclock_seconds=args.max_wallclock_seconds,
        iterations=args.iterations,
        train_log_every=args.train_log_every,
        val_loss_every=args.val_loss_every,
        seed=args.seed,
    )
