"""
Autoresearch helper for dataset/representation discovery.

This lane does not patch `train_gpt.py` directly. Instead it:
1. profiles the current FineWeb/tokenizer setup
2. computes vocab-budget tradeoffs
3. asks Claude to synthesize the evidence into ranked next experiments

Usage:
    python3 autoresearch_representation.py

Useful environment variables:
    AUTORESEARCH_MODEL=opus
    CLAUDE_EFFORT=medium
    PROFILE_VARIANT=sp1024
    PROFILE_TRAIN_TOKENS=1000000
    PROFILE_VAL_TOKENS=1000000
    TRADEOFF_VOCABS=256,384,512,768,1024,1536,2048,4096,8192,28416
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CLAUDE_MODEL = os.environ.get("AUTORESEARCH_MODEL", "opus")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "medium")

PROFILE_VARIANT = os.environ.get("PROFILE_VARIANT", "sp1024")
PROFILE_DATASET_DIR = os.environ.get("PROFILE_DATASET_DIR", "")
PROFILE_TOKENIZER_PATH = os.environ.get("PROFILE_TOKENIZER_PATH", "")
PROFILE_TRAIN_TOKENS = int(os.environ.get("PROFILE_TRAIN_TOKENS", "1000000"))
PROFILE_VAL_TOKENS = int(os.environ.get("PROFILE_VAL_TOKENS", "1000000"))
PROFILE_REPETITION_TOKENS = int(os.environ.get("PROFILE_REPETITION_TOKENS", "200000"))
TRADEOFF_VOCABS = os.environ.get("TRADEOFF_VOCABS", "256,384,512,768,1024,1536,2048,4096,8192,28416")
TRADEOFF_MODEL_DIM = int(os.environ.get("TRADEOFF_MODEL_DIM", "512"))
TRADEOFF_BASELINE_VOCAB = int(os.environ.get("TRADEOFF_BASELINE_VOCAB", "1024"))
TRADEOFF_TIED = int(os.environ.get("TRADEOFF_TIED", "1"))

ROOT = Path(__file__).resolve().parent
PROGRAM_FILE = ROOT / "program_representation.md"
OUT_DIR = ROOT / "autoresearch" / "representation_discovery"
PROFILE_JSON = OUT_DIR / "latest_profile.json"
PROFILE_MD = OUT_DIR / "latest_profile.md"
TRADEOFF_JSON = OUT_DIR / "latest_tradeoff.json"
TRADEOFF_MD = OUT_DIR / "latest_tradeoff.md"
REPORT_MD = OUT_DIR / "latest_report.md"
HISTORY_FILE = OUT_DIR / "history.jsonl"
RUN_LOG = OUT_DIR / "run.log"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def log_line(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_cmd(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def run_profile() -> dict:
    log_line(
        f"profile:start variant={PROFILE_VARIANT} train_tokens={PROFILE_TRAIN_TOKENS} "
        f"val_tokens={PROFILE_VAL_TOKENS} repetition_tokens={PROFILE_REPETITION_TOKENS}"
    )
    cmd = [
        sys.executable,
        str(ROOT / "data" / "profile_fineweb.py"),
        "--variant",
        PROFILE_VARIANT,
        "--train-tokens",
        str(PROFILE_TRAIN_TOKENS),
        "--val-tokens",
        str(PROFILE_VAL_TOKENS),
        "--repetition-tokens",
        str(PROFILE_REPETITION_TOKENS),
        "--output-json",
        str(PROFILE_JSON),
        "--output-md",
        str(PROFILE_MD),
    ]
    if PROFILE_DATASET_DIR:
        cmd.extend(["--dataset-dir", PROFILE_DATASET_DIR])
    if PROFILE_TOKENIZER_PATH:
        cmd.extend(["--tokenizer-path", PROFILE_TOKENIZER_PATH])
    run_cmd(cmd)
    profile = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    log_line(
        f"profile:done val_bytes_per_token={profile['val']['avg_bytes_per_token']:.4f} "
        f"train_bytes_per_token={profile['train']['avg_bytes_per_token']:.4f}"
    )
    return profile


def run_tradeoff() -> dict:
    log_line(
        f"tradeoff:start baseline_vocab={TRADEOFF_BASELINE_VOCAB} model_dim={TRADEOFF_MODEL_DIM} "
        f"tied={TRADEOFF_TIED} vocabs={TRADEOFF_VOCABS}"
    )
    cmd = [
        sys.executable,
        str(ROOT / "data" / "tokenizer_tradeoff_report.py"),
        "--vocab-sizes",
        TRADEOFF_VOCABS,
        "--baseline-vocab",
        str(TRADEOFF_BASELINE_VOCAB),
        "--model-dim",
        str(TRADEOFF_MODEL_DIM),
        "--tie-embeddings",
        str(TRADEOFF_TIED),
        "--output-json",
        str(TRADEOFF_JSON),
        "--output-md",
        str(TRADEOFF_MD),
    ]
    run_cmd(cmd)
    tradeoff = json.loads(TRADEOFF_JSON.read_text(encoding="utf-8"))
    log_line(f"tradeoff:done rows={len(tradeoff['rows'])}")
    return tradeoff


def load_history_tail(limit: int = 8) -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    rows = [json.loads(line) for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-limit:]


def build_prompt(profile: dict, tradeoff: dict) -> str:
    program = PROGRAM_FILE.read_text(encoding="utf-8")
    history = load_history_tail()
    history_text = "\n".join(
        f"- {row['timestamp']}: {row['headline']}" for row in history
    ) or "- no previous reports"

    # Keep the prompt compact by summarizing only the most relevant parts.
    val_classes = list(profile["val"]["class_stats"].items())[:6]
    rep_rows = profile["val"]["repetition_profile"]
    tradeoff_rows = tradeoff["rows"]
    top_vocab_rows = [row for row in tradeoff_rows if row["vocab_size"] in (512, 768, 1024, 1536, 2048, 4096, 28416)]

    prompt = f"""You are helping with the representation-discovery lane for the OpenAI parameter-golf challenge.

You are NOT patching train_gpt.py. Your job is to read the current profiler outputs and produce a high-signal research memo.

Research program:
{program}

Recent history:
{history_text}

Current FineWeb profile (validation split focus):
- val bytes/token: {profile['val']['avg_bytes_per_token']:.4f}
- train bytes/token: {profile['train']['avg_bytes_per_token']:.4f}
- val unique tokens observed: {profile['val']['unique_tokens_observed']}
- top validation classes by byte mass:
{json.dumps(val_classes, indent=2)}

Validation repetition profile:
{json.dumps(rep_rows, indent=2)}

Tokenizer vocab-budget tradeoff rows:
{json.dumps(top_vocab_rows, indent=2)}

Write a concise markdown memo with these sections:
1. Verdict
2. What the current profile implies about FineWeb structure
3. What it implies about tokenizer strategy
4. Ranked next experiments (5 max)
5. What not to do yet

Requirements:
- Be concrete and competition-focused.
- Prefer decisions over brainstorming.
- Say plainly whether a tokenizer experiment looks justified now.
- If you infer something from the data rather than observing it directly, label it as an inference.
"""
    return prompt


def run_claude(prompt: str) -> str:
    log_line(f"claude:start model={CLAUDE_MODEL} effort={CLAUDE_EFFORT}")
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        CLAUDE_MODEL,
        "--output-format",
        "json",
    ]
    if CLAUDE_EFFORT:
        cmd.extend(["--effort", CLAUDE_EFFORT])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Claude failed:\n{result.stdout}\n{result.stderr}")
    payload = json.loads(result.stdout.strip())
    if payload.get("is_error"):
        raise RuntimeError(f"Claude returned error payload:\n{result.stdout}\n{result.stderr}")
    report = str(payload.get("result") or "").strip()
    log_line(f"claude:done chars={len(report)}")
    return report


def extract_headline(report: str) -> str:
    for line in report.splitlines():
        clean = line.strip().lstrip("#").strip()
        if clean:
            return clean[:160]
    return "representation discovery update"


def append_history(report: str) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "headline": extract_headline(report),
        "profile_json": str(PROFILE_JSON),
        "tradeoff_json": str(TRADEOFF_JSON),
    }
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def main() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(__doc__.strip())
        return
    ensure_dirs()
    log_line("run:start")
    profile = run_profile()
    tradeoff = run_tradeoff()
    report = run_claude(build_prompt(profile, tradeoff))
    REPORT_MD.write_text(report + "\n", encoding="utf-8")
    append_history(report)
    log_line(f"run:done report={REPORT_MD}")
    print(report)


if __name__ == "__main__":
    main()
