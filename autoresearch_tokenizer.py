"""
Autoresearch loop for tokenizer discovery.

This loop:
1. reconstructs a text sample from tokenized FineWeb shards using the baseline tokenizer
2. evaluates the baseline tokenizer on that sample
3. asks Claude to propose one tokenizer experiment at a time
4. trains/evaluates the candidate tokenizer
5. logs history and repeats

Usage:
  DATASET_DIR=/path/to/fineweb10B_sp1024 \
  TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
  python3 autoresearch_tokenizer.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path


CLAUDE_MODEL = os.environ.get("AUTORESEARCH_MODEL", "opus")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "medium")
MAX_EXPERIMENTS = int(os.environ.get("MAX_EXPERIMENTS", "8"))
PROPOSAL_TIMEOUT = int(os.environ.get("PROPOSAL_TIMEOUT", "180"))
REVIEW_TIMEOUT = int(os.environ.get("REVIEW_TIMEOUT", "180"))

DATASET_DIR = os.environ.get("DATASET_DIR", "")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "")
SPLIT = os.environ.get("TEXT_SAMPLE_SPLIT", "train")
TEXT_SAMPLE_TOKENS = int(os.environ.get("TEXT_SAMPLE_TOKENS", "2000000"))
TEXT_SAMPLE_MAX_SHARDS = int(os.environ.get("TEXT_SAMPLE_MAX_SHARDS", "10"))
TEXT_SAMPLE_TOKENS_PER_SHARD = int(os.environ.get("TEXT_SAMPLE_TOKENS_PER_SHARD", "2000000"))
TEXT_SAMPLE_OFFSET_MODE = os.environ.get("TEXT_SAMPLE_OFFSET_MODE", "staggered")
TEXT_SAMPLE_CHUNK_TOKENS = int(os.environ.get("TEXT_SAMPLE_CHUNK_TOKENS", "2048"))
EVAL_MAX_CHUNKS = int(os.environ.get("EVAL_MAX_CHUNKS", "0"))

ROOT = Path(__file__).resolve().parent
PROGRAM_FILE = ROOT / "program_tokenizer.md"
OUT_DIR = ROOT / "autoresearch" / "tokenizer_discovery"
RUN_LOG = OUT_DIR / "run.log"
HISTORY_FILE = OUT_DIR / "history.jsonl"
TEXT_SAMPLE_PATH = OUT_DIR / "text_sample.jsonl"
BASELINE_JSON = OUT_DIR / "baseline_eval.json"
FRONTIER_JSON = OUT_DIR / "frontier.json"
LATEST_REVIEW_MD = OUT_DIR / "latest_review.md"
EXPERIMENTS_DIR = OUT_DIR / "experiments"
RUN_MANIFEST = OUT_DIR / "run_manifest.json"
ARCHIVE_DIR = OUT_DIR / "_archives"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)


def log_line(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_cmd(cmd: list[str], *, timeout: int | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def append_history(entry: dict) -> None:
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    return [json.loads(line) for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenizer_score(metrics: dict) -> float:
    tokens_per_byte = float(metrics["tokens_per_byte"])
    vocab_size = int(metrics["vocab_size"])
    dead_vocab_frac = float(metrics["dead_vocab_frac"])
    vocab_penalty = max(vocab_size - 1024, 0) / 1024.0 * 0.015
    dead_penalty = dead_vocab_frac * 0.02
    return tokens_per_byte + vocab_penalty + dead_penalty


def canonical_params(params: dict) -> dict[str, object]:
    return {
        "vocab_size": int(params["vocab_size"]),
        "model_type": str(params.get("model_type", "bpe")),
        "character_coverage": round(float(params.get("character_coverage", 0.9995)), 8),
        "input_sentence_size": int(params.get("input_sentence_size", 200000)),
        "max_chunks": int(params.get("max_chunks", 0)),
    }


def params_key(params: dict) -> str:
    return json.dumps(canonical_params(params), sort_keys=True, separators=(",", ":"))


def tried_param_keys() -> set[str]:
    keys: set[str] = set()
    for row in load_history():
        params = row.get("params")
        if isinstance(params, dict) and "vocab_size" in params:
            keys.add(params_key(params))
    return keys


def current_run_config() -> dict[str, object]:
    config = {
        "dataset_dir": str(Path(DATASET_DIR).expanduser().resolve()) if DATASET_DIR else "",
        "tokenizer_path": str(Path(TOKENIZER_PATH).expanduser().resolve()) if TOKENIZER_PATH else "",
        "split": SPLIT,
        "text_sample_tokens": TEXT_SAMPLE_TOKENS,
        "text_sample_max_shards": TEXT_SAMPLE_MAX_SHARDS,
        "text_sample_tokens_per_shard": TEXT_SAMPLE_TOKENS_PER_SHARD,
        "text_sample_offset_mode": TEXT_SAMPLE_OFFSET_MODE,
        "text_sample_chunk_tokens": TEXT_SAMPLE_CHUNK_TOKENS,
        "eval_max_chunks": EVAL_MAX_CHUNKS,
        "program_file": str(PROGRAM_FILE.resolve()),
    }
    config_json = json.dumps(config, sort_keys=True)
    config["config_hash"] = sha256(config_json.encode("utf-8")).hexdigest()[:12]
    return config


def archive_incompatible_state(previous: dict[str, object], current: dict[str, object]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = ARCHIVE_DIR / f"{timestamp}_{previous.get('config_hash', 'unknown')}"
    archive_root.mkdir(parents=True, exist_ok=True)
    for path in (TEXT_SAMPLE_PATH, BASELINE_JSON, FRONTIER_JSON, HISTORY_FILE, LATEST_REVIEW_MD, EXPERIMENTS_DIR):
        if path.exists():
            shutil.move(str(path), str(archive_root / path.name))
    (archive_root / "previous_run_manifest.json").write_text(
        json.dumps(previous, indent=2) + "\n",
        encoding="utf-8",
    )
    (archive_root / "new_run_manifest.json").write_text(
        json.dumps(current, indent=2) + "\n",
        encoding="utf-8",
    )
    if EXPERIMENTS_DIR.exists():
        shutil.rmtree(EXPERIMENTS_DIR)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    log_line(
        "state:archived_incompatible_run "
        f"archive={archive_root} old_hash={previous.get('config_hash')} new_hash={current.get('config_hash')}"
    )


def ensure_run_manifest() -> dict[str, object]:
    current = current_run_config()
    if RUN_MANIFEST.exists():
        previous = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
        if previous != current:
            archive_incompatible_state(previous, current)
    RUN_MANIFEST.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return current


def save_frontier(frontier: dict) -> None:
    FRONTIER_JSON.write_text(json.dumps(frontier, indent=2) + "\n", encoding="utf-8")


def load_frontier(baseline: dict) -> dict:
    if FRONTIER_JSON.exists():
        return json.loads(FRONTIER_JSON.read_text(encoding="utf-8"))
    frontier = {
        "best_label": "baseline_sp1024",
        "best_score": tokenizer_score(baseline),
        "best_metrics": baseline,
        "best_experiment_id": None,
        "headline": "baseline control",
        "frontier_update": "starting point",
        "next_direction": "test nearby vocabulary sizes first",
        "strategy_experiment_id": None,
        "strategy_score": tokenizer_score(baseline),
    }
    save_frontier(frontier)
    return frontier


def ensure_text_sample() -> None:
    if TEXT_SAMPLE_PATH.exists():
        return
    if not DATASET_DIR or not TOKENIZER_PATH:
        raise ValueError("DATASET_DIR and TOKENIZER_PATH are required")
    log_line(
        "text_sample:start "
        f"split={SPLIT} max_tokens={TEXT_SAMPLE_TOKENS} "
        f"max_shards={TEXT_SAMPLE_MAX_SHARDS} "
        f"tokens_per_shard={TEXT_SAMPLE_TOKENS_PER_SHARD} "
        f"offset_mode={TEXT_SAMPLE_OFFSET_MODE} "
        f"chunk_tokens={TEXT_SAMPLE_CHUNK_TOKENS}"
    )
    cmd = [
        sys.executable,
        str(ROOT / "data" / "extract_text_sample_from_shards.py"),
        "--dataset-dir",
        DATASET_DIR,
        "--tokenizer-path",
        TOKENIZER_PATH,
        "--split",
        SPLIT,
        "--max-tokens",
        str(TEXT_SAMPLE_TOKENS),
        "--max-shards",
        str(TEXT_SAMPLE_MAX_SHARDS),
        "--tokens-per-shard",
        str(TEXT_SAMPLE_TOKENS_PER_SHARD),
        "--offset-mode",
        TEXT_SAMPLE_OFFSET_MODE,
        "--chunk-tokens",
        str(TEXT_SAMPLE_CHUNK_TOKENS),
        "--output",
        str(TEXT_SAMPLE_PATH),
    ]
    run_cmd(cmd)
    log_line(f"text_sample:done path={TEXT_SAMPLE_PATH}")


def eval_tokenizer(model_path: Path, output_json: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "data" / "evaluate_tokenizer_on_text_sample.py"),
        "--family",
        "sentencepiece",
        "--model-path",
        str(model_path),
        "--sample-path",
        str(TEXT_SAMPLE_PATH),
        "--output-json",
        str(output_json),
    ]
    if EVAL_MAX_CHUNKS > 0:
        cmd.extend(["--max-chunks", str(EVAL_MAX_CHUNKS)])
    run_cmd(cmd)
    return json.loads(output_json.read_text(encoding="utf-8"))


def ensure_baseline() -> dict:
    if BASELINE_JSON.exists():
        return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    log_line("baseline:start")
    baseline = eval_tokenizer(Path(TOKENIZER_PATH).expanduser().resolve(), BASELINE_JSON)
    log_line(
        f"baseline:done tokens_per_byte={baseline['tokens_per_byte']:.6f} "
        f"dead_vocab_frac={baseline['dead_vocab_frac']:.4f}"
    )
    return baseline


def recent_history_summary(limit: int = 10) -> str:
    rows = load_history()[-limit:]
    if not rows:
        return "- no experiments yet"
    lines = []
    for row in rows:
        metrics = row.get("metrics", {})
        lines.append(
            f"- exp {row['experiment_id']}: type={row['type']} "
            f"tpb={metrics.get('tokens_per_byte')} dead={metrics.get('dead_vocab_frac')} "
            f"score={row.get('score')} keep={row.get('decision', {}).get('keep')} "
            f"params={canonical_params(row.get('params', {})) if row.get('params') else row.get('params')}"
        )
    return "\n".join(lines)


def build_prompt(baseline: dict, frontier: dict) -> str:
    program = PROGRAM_FILE.read_text(encoding="utf-8")
    tried_keys = sorted(tried_param_keys())
    recent_tried = "\n".join(f"- {key}" for key in tried_keys[-15:]) if tried_keys else "- none"
    return f"""You are running tokenizer autoresearch for the OpenAI parameter-golf challenge.

Research program:
{program}

Current baseline metrics:
{json.dumps(baseline, indent=2)}

Current frontier:
{json.dumps(frontier, indent=2)}

Recent experiment history:
{recent_history_summary()}

Recently tried canonical params:
{recent_tried}

You may propose exactly one experiment of this form:
{{
  "description": "...",
  "type": "sentencepiece_train",
  "params": {{
    "vocab_size": 1024,
    "model_type": "bpe",
    "character_coverage": 0.9995,
    "input_sentence_size": 200000,
    "max_chunks": 0
  }}
}}

Rules:
- Keep experiments cheap and local.
- Search nearby vocab sizes first unless the history strongly suggests otherwise.
- Prefer changing one primary variable at a time.
- Do not propose any experiment you already tried.
- Optimize the frontier score, not raw tokens_per_byte alone.
- Larger vocabularies must earn their cost.
- Output JSON only.
"""


def propose_experiment(baseline: dict, frontier: dict) -> dict:
    cmd = [
        "claude",
        "-p",
        build_prompt(baseline, frontier),
        "--model",
        CLAUDE_MODEL,
        "--output-format",
        "text",
    ]
    if CLAUDE_EFFORT:
        cmd.extend(["--effort", CLAUDE_EFFORT])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=PROPOSAL_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Claude proposal failed:\n{result.stdout}\n{result.stderr}")
    text = result.stdout.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not parse proposal JSON:\n{text}")
    proposal = json.loads(match.group(0))
    if proposal.get("type") != "sentencepiece_train":
        raise ValueError(f"Unsupported experiment type: {proposal.get('type')}")
    params = proposal.get("params")
    if not isinstance(params, dict):
        raise ValueError("Proposal must include a params object")
    key = params_key(params)
    if key in tried_param_keys():
        raise ValueError(f"Claude proposed an already-tried experiment: {canonical_params(params)}")
    return proposal


def run_experiment(experiment_id: int, proposal: dict) -> dict:
    exp_dir = EXPERIMENTS_DIR / f"{experiment_id:04d}"
    exp_dir.mkdir(exist_ok=True)
    params = proposal["params"]
    prefix = exp_dir / "tokenizer"
    log_line(f"experiment:{experiment_id}:start type=sentencepiece_train params={params}")
    train_cmd = [
        sys.executable,
        str(ROOT / "data" / "train_sentencepiece_tokenizer.py"),
        "--sample-path",
        str(TEXT_SAMPLE_PATH),
        "--output-prefix",
        str(prefix),
        "--vocab-size",
        str(int(params["vocab_size"])),
        "--model-type",
        str(params.get("model_type", "bpe")),
        "--character-coverage",
        str(float(params.get("character_coverage", 0.9995))),
        "--input-sentence-size",
        str(int(params.get("input_sentence_size", 200000))),
        "--max-chunks",
        str(int(params.get("max_chunks", 0))),
    ]
    run_cmd(train_cmd)
    metrics = eval_tokenizer(prefix.with_suffix(".model"), exp_dir / "eval.json")
    score = tokenizer_score(metrics)
    readme = exp_dir / "README.md"
    readme.write_text(
        json.dumps(
            {
                "description": proposal.get("description", ""),
                "type": proposal["type"],
                "params": params,
                "metrics": metrics,
                "score": score,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "description": proposal.get("description", ""),
        "type": proposal["type"],
        "params": params,
        "metrics": metrics,
        "score": score,
    }
    log_line(
        f"experiment:{experiment_id}:done tokens_per_byte={metrics['tokens_per_byte']:.6f} "
        f"dead_vocab_frac={metrics['dead_vocab_frac']:.4f} score={score:.6f}"
    )
    return entry


def build_review_prompt(baseline: dict, frontier: dict, result: dict) -> str:
    return f"""You are reviewing a tokenizer autoresearch experiment for the OpenAI parameter-golf challenge.

Baseline metrics:
{json.dumps(baseline, indent=2)}

Current frontier:
{json.dumps(frontier, indent=2)}

New experiment result:
{json.dumps(result, indent=2)}

The scalar score is lower-is-better and already includes penalties for larger vocabularies and dead vocab.

Return JSON only:
{{
  "keep": true,
  "headline": "short judgment",
  "reasoning": "2-4 sentence explanation",
  "frontier_update": "what this changes about the search strategy",
  "next_direction": "one short sentence about what to try next"
}}

Rules:
- keep=true only if this result meaningfully improves the frontier or reveals an important strategic shift
- do not keep tiny cosmetic changes unless they matter strategically
- use the actual score and metrics, not vibes
"""


def review_experiment(baseline: dict, frontier: dict, result: dict) -> dict:
    log_line(f"review:start experiment={result['experiment_id']}")
    cmd = [
        "claude",
        "-p",
        build_review_prompt(baseline, frontier, result),
        "--model",
        CLAUDE_MODEL,
        "--output-format",
        "text",
    ]
    if CLAUDE_EFFORT:
        cmd.extend(["--effort", CLAUDE_EFFORT])
    review = subprocess.run(cmd, capture_output=True, text=True, timeout=REVIEW_TIMEOUT)
    if review.returncode != 0:
        raise RuntimeError(f"Claude review failed:\n{review.stdout}\n{review.stderr}")
    text = review.stdout.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not parse review JSON:\n{text}")
    decision = json.loads(match.group(0))
    log_line(
        f"review:done experiment={result['experiment_id']} keep={decision.get('keep')} "
        f"headline={decision.get('headline', '')}"
    )
    return decision


def apply_review(frontier: dict, result: dict, decision: dict) -> dict:
    result["decision"] = decision
    result["kept"] = bool(decision.get("keep"))
    current_best = float(frontier["best_score"])
    result_score = float(result["score"])
    if result["kept"]:
        frontier = {
            **frontier,
            "headline": decision.get("headline", ""),
            "frontier_update": decision.get("frontier_update", ""),
            "next_direction": decision.get("next_direction", ""),
            "strategy_experiment_id": result["experiment_id"],
            "strategy_score": result_score,
        }
    if result["kept"] and result_score < current_best:
        frontier.update(
            {
                "best_label": f"experiment_{result['experiment_id']:04d}",
                "best_score": result_score,
                "best_metrics": result["metrics"],
                "best_experiment_id": result["experiment_id"],
            }
        )
        save_frontier(frontier)
        log_line(
            f"frontier:update best=experiment_{result['experiment_id']:04d} "
            f"score={result_score:.6f}"
        )
    elif result["kept"]:
        save_frontier(frontier)
        log_line(
            f"frontier:strategy_update experiment={result['experiment_id']:04d} "
            f"best_score={current_best:.6f} strategy_score={result_score:.6f}"
        )
    append_history(result)
    return frontier


def main() -> None:
    ensure_dirs()
    run_config = ensure_run_manifest()
    log_line("run:start")
    log_line(f"run:config hash={run_config['config_hash']} split={run_config['split']}")
    ensure_text_sample()
    baseline = ensure_baseline()
    frontier = load_frontier(baseline)
    existing = load_history()
    start_id = 1 + max((row.get("experiment_id", 0) for row in existing), default=0)
    for experiment_id in range(start_id, start_id + MAX_EXPERIMENTS):
        proposal = propose_experiment(baseline, frontier)
        result = run_experiment(experiment_id, proposal)
        decision = review_experiment(baseline, frontier, result)
        frontier = apply_review(frontier, result, decision)
        LATEST_REVIEW_MD.write_text(
            "\n".join(
                [
                    f"# Review {experiment_id:04d}",
                    "",
                    f"- keep: `{bool(decision.get('keep'))}`",
                    f"- headline: {decision.get('headline', '')}",
                    f"- frontier_update: {decision.get('frontier_update', '')}",
                    f"- next_direction: {decision.get('next_direction', '')}",
                    "",
                    decision.get("reasoning", ""),
                    "",
                ]
            ),
            encoding="utf-8",
        )
    log_line("run:done")


if __name__ == "__main__":
    main()
