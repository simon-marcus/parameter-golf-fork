"""
Focused TokenMonster candidate-generation loop.

This lane no longer searches across tokenizer families.
It explores only small-vocab TokenMonster candidates around the current 1024-vocab frontier.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path


CLAUDE_MODEL = os.environ.get("AUTORESEARCH_MODEL", "opus")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "medium")
MAX_EXPERIMENTS = int(os.environ.get("MAX_EXPERIMENTS", "12"))
PROPOSAL_TIMEOUT = int(os.environ.get("PROPOSAL_TIMEOUT", "180"))
REVIEW_TIMEOUT = int(os.environ.get("REVIEW_TIMEOUT", "180"))
CLAUDE_RETRIES = int(os.environ.get("CLAUDE_RETRIES", "3"))

DATASET_DIR = os.environ.get("DATASET_DIR", "")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "")
TEXT_SAMPLE_TOKENS = int(os.environ.get("TEXT_SAMPLE_TOKENS", "2_000_000"))
TEXT_SAMPLE_MAX_SHARDS = int(os.environ.get("TEXT_SAMPLE_MAX_SHARDS", "10"))
TEXT_SAMPLE_TOKENS_PER_SHARD = int(os.environ.get("TEXT_SAMPLE_TOKENS_PER_SHARD", "2_000_000"))
TEXT_SAMPLE_OFFSET_MODE = os.environ.get("TEXT_SAMPLE_OFFSET_MODE", "staggered")
TEXT_SAMPLE_CHUNK_TOKENS = int(os.environ.get("TEXT_SAMPLE_CHUNK_TOKENS", "2048"))
PROXY_AGGREGATE_PATH = os.environ.get(
    "TOKENIZER_PROXY_AGGREGATE_PATH",
    "/Users/simon/Code/parameter-golf/modal_logs/proxy_confirm_long_aggregate.json",
)
TOKENMONSTER_VOCABS = [
    item
    for item in os.environ.get(
        "TOKENMONSTER_CANDIDATES",
        "english-1024-balanced-v1,english-1024-clean-v1,english-1024-consistent-v1,english-2048-clean-v1",
    ).split(",")
    if item
]

ROOT = Path(__file__).resolve().parent
PROGRAM_FILE = ROOT / "program_tokenmonster.md"
OUT_DIR = ROOT / "autoresearch" / "tokenmonster_discovery"
RUN_LOG = OUT_DIR / "run.log"
HISTORY_FILE = OUT_DIR / "history.jsonl"
TEXT_SAMPLE_PATH = OUT_DIR / "text_sample.jsonl"
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


def run_claude_json(prompt: str, *, timeout: int, phase: str) -> dict:
    cmd = ["claude", "-p", prompt, "--model", CLAUDE_MODEL, "--output-format", "text"]
    if CLAUDE_EFFORT:
        cmd.extend(["--effort", CLAUDE_EFFORT])
    last_error: Exception | None = None
    for attempt in range(1, CLAUDE_RETRIES + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                raise RuntimeError(f"Claude {phase} failed:\n{result.stdout}\n{result.stderr}")
            text = result.stdout.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise RuntimeError(f"Could not parse {phase} JSON:\n{text}")
            return json.loads(match.group(0))
        except Exception as exc:
            last_error = exc
            log_line(f"{phase}:error attempt={attempt}/{CLAUDE_RETRIES} error={exc}")
    assert last_error is not None
    raise RuntimeError(f"Claude {phase} failed after {CLAUDE_RETRIES} attempts") from last_error


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    return [json.loads(line) for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_history(entry: dict) -> None:
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def current_run_config() -> dict[str, object]:
    config = {
        "dataset_dir": DATASET_DIR,
        "tokenizer_path": TOKENIZER_PATH,
        "text_sample_tokens": TEXT_SAMPLE_TOKENS,
        "text_sample_max_shards": TEXT_SAMPLE_MAX_SHARDS,
        "text_sample_tokens_per_shard": TEXT_SAMPLE_TOKENS_PER_SHARD,
        "text_sample_offset_mode": TEXT_SAMPLE_OFFSET_MODE,
        "text_sample_chunk_tokens": TEXT_SAMPLE_CHUNK_TOKENS,
        "proxy_aggregate_path": PROXY_AGGREGATE_PATH,
        "tokenmonster_vocabs": TOKENMONSTER_VOCABS,
        "program_file": str(PROGRAM_FILE.resolve()),
    }
    config["config_hash"] = sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return config


def archive_incompatible_state(previous: dict[str, object], current: dict[str, object]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = ARCHIVE_DIR / f"{timestamp}_{previous.get('config_hash', 'unknown')}"
    archive_root.mkdir(parents=True, exist_ok=True)
    for path in (TEXT_SAMPLE_PATH, FRONTIER_JSON, HISTORY_FILE, LATEST_REVIEW_MD, EXPERIMENTS_DIR):
        if path.exists():
            shutil.move(str(path), str(archive_root / path.name))
    (archive_root / "previous_run_manifest.json").write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    (archive_root / "new_run_manifest.json").write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
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


def ensure_text_sample() -> None:
    if TEXT_SAMPLE_PATH.exists():
        return
    cmd = [
        sys.executable,
        str(ROOT / "data" / "extract_text_sample_from_shards.py"),
        "--dataset-dir",
        DATASET_DIR,
        "--tokenizer-path",
        TOKENIZER_PATH,
        "--split",
        "train",
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
    log_line("text_sample:start")
    run_cmd(cmd)
    log_line(f"text_sample:done path={TEXT_SAMPLE_PATH}")


def evaluate_tokenizer(model_ref: str, output_json: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "data" / "evaluate_tokenizer_on_text_sample.py"),
        "--family",
        "tokenmonster",
        "--model-path",
        model_ref,
        "--sample-path",
        str(TEXT_SAMPLE_PATH),
        "--output-json",
        str(output_json),
    ]
    run_cmd(cmd)
    return json.loads(output_json.read_text(encoding="utf-8"))


def load_proxy_aggregate() -> dict[str, object]:
    return json.loads(Path(PROXY_AGGREGATE_PATH).expanduser().resolve().read_text(encoding="utf-8"))


def load_frontier() -> dict[str, object]:
    if FRONTIER_JSON.exists():
        return json.loads(FRONTIER_JSON.read_text(encoding="utf-8"))
    aggregate = load_proxy_aggregate()
    best = aggregate["rows"][0]
    frontier = {
        "best_label": best["name"],
        "best_score": float(best["postquant_val_bpb_mean"]),
        "best_std": float(best["postquant_val_bpb_std"]),
        "best_metrics": best,
        "headline": "aggregate proxy control",
        "next_direction": "start from the best 1024 TokenMonster family member",
    }
    FRONTIER_JSON.write_text(json.dumps(frontier, indent=2) + "\n", encoding="utf-8")
    return frontier


def tried_keys() -> set[str]:
    out = set()
    for row in load_history():
        out.add(json.dumps({"type": row["type"], "params": row["params"]}, sort_keys=True))
    return out


def recent_history(limit: int = 10) -> str:
    rows = load_history()[-limit:]
    if not rows:
        return "- none"
    return "\n".join(
        f"- exp {row['experiment_id']}: type={row['type']} score={row['score']} params={row['params']}"
        for row in rows
    )


def recent_rows(limit: int = 12) -> list[dict]:
    rows = load_history()
    if limit <= 0:
        return rows
    return rows[-limit:]


def is_local_resize_probe(row: dict) -> bool:
    if row.get("type") != "tokenmonster_build":
        return False
    params = row.get("params", {})
    if str(params.get("base_vocab", "")).startswith("english-1024-") is False:
        return False
    resize = int(params.get("resize", 0) or 0)
    return 1010 <= resize <= 1022


def stagnation_summary(frontier: dict) -> dict[str, object]:
    rows = recent_rows(12)
    if not rows:
        return {
            "recent_non_improving": 0,
            "recent_local_resize_probes": 0,
            "force_broadening": False,
            "blocked_resizes": [],
        }
    recent_non_improving = 0
    for row in reversed(rows):
        improved = bool(row.get("kept")) and float(row.get("score", 1e18)) < float(frontier["best_score"])
        if improved:
            break
        recent_non_improving += 1
    local_resize_rows = [row for row in rows if is_local_resize_probe(row)]
    blocked_resizes = sorted(
        {
            int(row.get("params", {}).get("resize", 0))
            for row in local_resize_rows
            if int(row.get("params", {}).get("resize", 0)) > 0
        }
    )
    force_broadening = recent_non_improving >= 6 and len(local_resize_rows) >= 5
    return {
        "recent_non_improving": recent_non_improving,
        "recent_local_resize_probes": len(local_resize_rows),
        "force_broadening": force_broadening,
        "blocked_resizes": blocked_resizes,
    }


def build_prompt(frontier: dict) -> str:
    aggregate = load_proxy_aggregate()
    program = PROGRAM_FILE.read_text(encoding="utf-8")
    stagnation = stagnation_summary(frontier)
    broadening_rules = ""
    if stagnation["force_broadening"]:
        blocked = ", ".join(str(x) for x in stagnation["blocked_resizes"]) or "none"
        broadening_rules = f"""

Stagnation signal:
{json.dumps(stagnation, indent=2)}

Mandatory broadening mode:
- do NOT propose another tiny local resize probe in the already-explored band
- avoid resize values in this exhausted band: {blocked}
- prefer one of:
  - switch base vocab among `english-1024-balanced-v1`, `english-1024-clean-v1`, `english-1024-consistent-v1`
  - keep resize at 1024 and change deletion heuristics materially
  - try punctuation/space-composite pruning or multiword pruning regimes not just byte-cap tweaks
  - use `tokenmonster_eval` for a different 1024 control if needed
"""
    return f"""You are running a TokenMonster-only candidate generation loop for Parameter Golf.

Program:
{program}

Current aggregate proxy results:
{json.dumps(aggregate, indent=2)}

Current frontier:
{json.dumps(frontier, indent=2)}

Recent history:
{recent_history()}
{broadening_rules}

Allowed experiment JSON:
{{
  "description": "...",
  "type": "tokenmonster_eval",
  "params": {{
    "vocab_ref": "english-1024-balanced-v1"
  }}
}}

or

{{
  "description": "...",
  "type": "tokenmonster_build",
  "params": {{
    "base_vocab": "english-1024-balanced-v1",
    "resize": 960,
    "max_decoded_bytes": 24,
    "delete_multiword": true,
    "delete_space_punct": false,
    "delete_regex": ["  {{3,}}"]
  }}
}}

Rules:
- stay near the 1024 TokenMonster family
- prefer pruning/editing over larger vocab growth
- do not propose SentencePiece
- after stagnation, change search mode instead of continuing micro-resize search
- output JSON only
"""


def propose_experiment(frontier: dict) -> dict:
    for attempt in range(1, CLAUDE_RETRIES + 2):
        proposal = run_claude_json(build_prompt(frontier), timeout=PROPOSAL_TIMEOUT, phase="proposal")
        key = json.dumps({"type": proposal["type"], "params": proposal["params"]}, sort_keys=True)
        if key not in tried_keys():
            return proposal
        log_line(f"proposal:duplicate attempt={attempt} type={proposal['type']} params={proposal['params']}")
    raise RuntimeError("Could not obtain a non-duplicate TokenMonster proposal")


def score_candidate(metrics: dict, frontier: dict, proposal: dict) -> float:
    aggregate = load_proxy_aggregate()
    if proposal["type"] == "tokenmonster_eval":
        exact = next(row for row in aggregate["rows"] if row["name"] == proposal["params"]["vocab_ref"])
        return float(exact["postquant_val_bpb_mean"]) + 0.25 * float(exact["postquant_val_bpb_std"])
    # Custom local candidates are triaged by a conservative cheap score anchored to current best.
    return (
        float(frontier["best_score"])
        + 6.0 * (float(metrics["tokens_per_byte"]) - float(frontier["best_metrics"].get("tokens_per_byte", 0.408)))
        + 0.5 * (float(metrics["dead_vocab_frac"]) - float(frontier["best_metrics"].get("dead_vocab_frac", 0.046)))
        + 0.05 * max(int(metrics["vocab_size"]) - 1024, 0) / 1024.0
    )


def run_experiment(experiment_id: int, proposal: dict) -> dict:
    exp_dir = EXPERIMENTS_DIR / f"{experiment_id:04d}"
    exp_dir.mkdir(exist_ok=True)
    params = proposal["params"]
    proposal_type = proposal["type"]
    log_line(f"experiment:{experiment_id}:start type={proposal_type} params={params}")
    if proposal_type == "tokenmonster_eval":
        model_ref = str(params["vocab_ref"])
    else:
        model_path = exp_dir / "candidate.vocab"
        cmd = [
            sys.executable,
            str(ROOT / "data" / "build_tokenmonster_candidate.py"),
            "--base-vocab",
            str(params["base_vocab"]),
            "--output-path",
            str(model_path),
            "--max-decoded-bytes",
            str(int(params.get("max_decoded_bytes", 24))),
        ]
        if int(params.get("resize", 0)) > 0:
            cmd.extend(["--resize", str(int(params["resize"]))])
        for pattern in params.get("delete_regex", []):
            cmd.extend(["--delete-regex", str(pattern)])
        if params.get("delete_multiword"):
            cmd.append("--delete-multiword")
        if params.get("delete_space_punct"):
            cmd.append("--delete-space-punct")
        run_cmd(cmd)
        model_ref = str(model_path)
    metrics = evaluate_tokenizer(model_ref, exp_dir / "eval.json")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "description": proposal.get("description", ""),
        "type": proposal_type,
        "params": params,
        "metrics": metrics,
    }
    log_line(
        f"experiment:{experiment_id}:done vocab={metrics['vocab_size']} "
        f"tpb={metrics['tokens_per_byte']:.6f} dead={metrics['dead_vocab_frac']:.4f}"
    )
    return entry


def build_review_prompt(frontier: dict, result: dict) -> str:
    return f"""You are reviewing a TokenMonster-only candidate generation result.

Frontier:
{json.dumps(frontier, indent=2)}

Result:
{json.dumps(result, indent=2)}

Return JSON only:
{{
  "keep": true,
  "headline": "short judgment",
  "reasoning": "2-4 sentences",
  "next_direction": "one sentence"
}}

Rules:
- be conservative
- this is a generation lane, not final proxy proof
- prefer candidates that look simpler/more atomic, and usually stay in the 1024 family unless stagnation evidence justifies a broader TokenMonster move
"""


def main() -> None:
    ensure_dirs()
    run_config = ensure_run_manifest()
    log_line("run:start")
    log_line(f"run:config hash={run_config['config_hash']}")
    ensure_text_sample()
    frontier = load_frontier()
    existing = load_history()
    start_id = 1 + max((row.get("experiment_id", 0) for row in existing), default=0)
    for experiment_id in range(start_id, start_id + MAX_EXPERIMENTS):
        proposal = propose_experiment(frontier)
        result = run_experiment(experiment_id, proposal)
        result["score"] = score_candidate(result["metrics"], frontier, proposal)
        decision = run_claude_json(build_review_prompt(frontier, result), timeout=REVIEW_TIMEOUT, phase="review")
        result["decision"] = decision
        result["kept"] = bool(decision.get("keep"))
        append_history(result)
        if result["kept"] and float(result["score"]) < float(frontier["best_score"]):
            frontier = {
                "best_label": f"experiment_{experiment_id:04d}",
                "best_score": float(result["score"]),
                "best_metrics": result["metrics"],
                "headline": decision.get("headline", ""),
                "next_direction": decision.get("next_direction", ""),
            }
            FRONTIER_JSON.write_text(json.dumps(frontier, indent=2) + "\n", encoding="utf-8")
            log_line(f"frontier:update best=experiment_{experiment_id:04d} score={result['score']:.6f}")
        LATEST_REVIEW_MD.write_text(
            "\n".join(
                [
                    f"# Review {experiment_id:04d}",
                    "",
                    f"- keep: `{result['kept']}`",
                    f"- headline: {decision.get('headline', '')}",
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
