"""
Modal launcher for tokenizer proxy validation on a cheap single GPU.

This runs `proxy_validate_tokenizers.py` remotely with:
- baseline sp1024 tokenizer
- current best tokenizer-discovery candidate

Default target is an L4 GPU because this is a relative proxy comparison, not a
record-timing run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
ENV_LOCAL = ROOT / ".env.local"
LOCAL_DATA_ROOT = Path("/Users/simon/Code/parameter-golf-local/data")
LOCAL_DATASET_DIR = LOCAL_DATA_ROOT / "datasets" / "fineweb10B_sp1024"
LOCAL_TOKENIZERS_DIR = LOCAL_DATA_ROOT / "tokenizers"
LOCAL_TOKENIZER_EXPERIMENTS_DIR = ROOT / "autoresearch" / "tokenizer_discovery" / "experiments"
LOCAL_TOKENIZER_PROXY_EXPERIMENTS_DIR = ROOT / "autoresearch" / "tokenizer_proxy_discovery" / "experiments"
LOCAL_TOKENMONSTER_EXPERIMENTS_DIR = ROOT / "autoresearch" / "tokenmonster_discovery" / "experiments"

REMOTE_ROOT = "/root/parameter-golf"
REMOTE_DATASET_DIR = f"{REMOTE_ROOT}/mounted_data/datasets/fineweb10B_sp1024"
REMOTE_TOKENIZERS_DIR = f"{REMOTE_ROOT}/mounted_data/tokenizers"
REMOTE_TOKENIZER_EXPERIMENTS_DIR = f"{REMOTE_ROOT}/tokenizer_candidates"
REMOTE_TOKENIZER_PROXY_EXPERIMENTS_DIR = f"{REMOTE_ROOT}/tokenizer_proxy_candidates"
REMOTE_TOKENMONSTER_EXPERIMENTS_DIR = f"{REMOTE_ROOT}/tokenmonster_candidates"


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
    .add_local_file(ROOT / "train_gpt.py", remote_path=f"{REMOTE_ROOT}/train_gpt.py", copy=True)
    .add_local_file(ROOT / "proxy_validate_tokenizers.py", remote_path=f"{REMOTE_ROOT}/proxy_validate_tokenizers.py", copy=True)
    .add_local_dir(ROOT / "data", remote_path=f"{REMOTE_ROOT}/data", copy=True)
    .add_local_dir(LOCAL_DATASET_DIR, remote_path=REMOTE_DATASET_DIR, copy=True)
    .add_local_dir(LOCAL_TOKENIZERS_DIR, remote_path=REMOTE_TOKENIZERS_DIR, copy=True)
    .add_local_dir(LOCAL_TOKENIZER_EXPERIMENTS_DIR, remote_path=REMOTE_TOKENIZER_EXPERIMENTS_DIR, copy=True)
    .add_local_dir(
        LOCAL_TOKENIZER_PROXY_EXPERIMENTS_DIR,
        remote_path=REMOTE_TOKENIZER_PROXY_EXPERIMENTS_DIR,
        copy=True,
    )
    .add_local_dir(LOCAL_TOKENMONSTER_EXPERIMENTS_DIR, remote_path=REMOTE_TOKENMONSTER_EXPERIMENTS_DIR, copy=True)
    .run_commands(f"mkdir -p {REMOTE_ROOT}/modal_logs {REMOTE_ROOT}/proxy_validation")
)

app = modal.App("parameter-golf-proxy-validate", image=image)


@app.function(gpu="L4", timeout=60 * 60, memory=32768)
def run_proxy_validation(
    *,
    candidate_models: list[str],
    tokenmonster_vocabs: list[str],
    tokenmonster_models: list[str],
    top_k: int,
    train_sample_tokens: int,
    val_sample_tokens: int,
    train_max_chunks: int,
    val_max_chunks: int,
    max_wallclock_seconds: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    import subprocess

    work_dir = REMOTE_ROOT
    output_json = f"{REMOTE_ROOT}/proxy_validation/modal_summary.json"
    cmd = [
        "python3",
        f"{REMOTE_ROOT}/proxy_validate_tokenizers.py",
        "--dataset-dir",
        REMOTE_DATASET_DIR,
        "--baseline-tokenizer",
        f"{REMOTE_TOKENIZERS_DIR}/fineweb_1024_bpe.model",
        "--top-k",
        str(top_k),
        "--train-sample-tokens",
        str(train_sample_tokens),
        "--val-sample-tokens",
        str(val_sample_tokens),
        "--train-max-chunks",
        str(train_max_chunks),
        "--val-max-chunks",
        str(val_max_chunks),
        "--max-wallclock-seconds",
        str(max_wallclock_seconds),
        "--iterations",
        str(iterations),
        "--seed",
        str(seed),
        "--output-json",
        output_json,
    ]
    for candidate_model in candidate_models:
        cmd.extend(["--candidate-model", candidate_model])
    for vocab_ref in tokenmonster_vocabs:
        cmd.extend(["--tokenmonster-vocab", vocab_ref])
    for model_path in tokenmonster_models:
        cmd.extend(["--tokenmonster-model", model_path])
    env = os.environ.copy()
    env["PROXY_TRAIN_PYTHON"] = "python3"

    print("RUNNING:", " ".join(cmd))
    process = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True)
    log_text = process.stdout + process.stderr
    print(log_text)

    summary = None
    try:
        summary = json.loads(Path(output_json).read_text(encoding="utf-8"))
    except Exception:
        summary = None

    return {
        "exit_code": process.returncode,
        "log": log_text,
        "summary": summary,
    }


@app.local_entrypoint()
def main(
    output_dir: str = "modal_logs/proxy_validate_tokenizers_l4",
    top_k: int = 0,
    candidate_models: str = "",
    tokenmonster_vocabs: str = "",
    tokenmonster_models: str = "",
    train_sample_tokens: int = 2_000_000,
    val_sample_tokens: int = 500_000,
    train_max_chunks: int = 400,
    val_max_chunks: int = 120,
    max_wallclock_seconds: int = 20,
    iterations: int = 200,
    seed: int = 1337,
):
    candidate_model_list = [item for item in candidate_models.split(",") if item]
    tokenmonster_vocab_list = [item for item in tokenmonster_vocabs.split(",") if item]
    tokenmonster_model_list = [item for item in tokenmonster_models.split(",") if item]
    result = run_proxy_validation.remote(
        candidate_models=candidate_model_list,
        tokenmonster_vocabs=tokenmonster_vocab_list,
        tokenmonster_models=tokenmonster_model_list,
        top_k=top_k,
        train_sample_tokens=train_sample_tokens,
        val_sample_tokens=val_sample_tokens,
        train_max_chunks=train_max_chunks,
        val_max_chunks=val_max_chunks,
        max_wallclock_seconds=max_wallclock_seconds,
        iterations=iterations,
        seed=seed,
    )

    out_dir = (ROOT / output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "modal_proxy_validate.log").write_text(result["log"], encoding="utf-8")
    if result.get("summary") is not None:
        (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"exit_code": result["exit_code"], "output_dir": str(out_dir)}, indent=2))
    if result.get("summary") is not None:
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="modal_logs/proxy_validate_tokenizers_l4")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--candidate-model", action="append", default=[])
    parser.add_argument("--tokenmonster-vocab", action="append", default=[])
    parser.add_argument("--tokenmonster-model", action="append", default=[])
    parser.add_argument("--train-sample-tokens", type=int, default=2_000_000)
    parser.add_argument("--val-sample-tokens", type=int, default=500_000)
    parser.add_argument("--train-max-chunks", type=int, default=400)
    parser.add_argument("--val-max-chunks", type=int, default=120)
    parser.add_argument("--max-wallclock-seconds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        top_k=args.top_k,
        candidate_models=",".join(args.candidate_model),
        tokenmonster_vocabs=",".join(args.tokenmonster_vocab),
        tokenmonster_models=",".join(args.tokenmonster_model),
        train_sample_tokens=args.train_sample_tokens,
        val_sample_tokens=args.val_sample_tokens,
        train_max_chunks=args.train_max_chunks,
        val_max_chunks=args.val_max_chunks,
        max_wallclock_seconds=args.max_wallclock_seconds,
        iterations=args.iterations,
        seed=args.seed,
    )
