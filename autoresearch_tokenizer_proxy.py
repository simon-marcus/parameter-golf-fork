"""
Autoresearch loop for proxy-calibrated tokenizer discovery.

This lane differs from `autoresearch_tokenizer.py` in two ways:
1. it calibrates the cheap tokenizer screen against observed proxy-training outcomes
2. it searches a mixed space of small SentencePiece and small TokenMonster candidates

Usage:
  DATASET_DIR=/path/to/fineweb10B_sp1024 \
  TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
  python3 autoresearch_tokenizer_proxy.py
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
MAX_EXPERIMENTS = int(os.environ.get("MAX_EXPERIMENTS", "8"))
PROPOSAL_TIMEOUT = int(os.environ.get("PROPOSAL_TIMEOUT", "180"))
REVIEW_TIMEOUT = int(os.environ.get("REVIEW_TIMEOUT", "180"))
CLAUDE_RETRIES = int(os.environ.get("CLAUDE_RETRIES", "3"))

DATASET_DIR = os.environ.get("DATASET_DIR", "")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "")
SPLIT = os.environ.get("TEXT_SAMPLE_SPLIT", "train")
TEXT_SAMPLE_TOKENS = int(os.environ.get("TEXT_SAMPLE_TOKENS", "2_000_000"))
TEXT_SAMPLE_MAX_SHARDS = int(os.environ.get("TEXT_SAMPLE_MAX_SHARDS", "10"))
TEXT_SAMPLE_TOKENS_PER_SHARD = int(os.environ.get("TEXT_SAMPLE_TOKENS_PER_SHARD", "2_000_000"))
TEXT_SAMPLE_OFFSET_MODE = os.environ.get("TEXT_SAMPLE_OFFSET_MODE", "staggered")
TEXT_SAMPLE_CHUNK_TOKENS = int(os.environ.get("TEXT_SAMPLE_CHUNK_TOKENS", "2048"))
EVAL_MAX_CHUNKS = int(os.environ.get("EVAL_MAX_CHUNKS", "0"))

SCORE_MODEL_DIM = int(os.environ.get("TOKENIZER_SCORE_MODEL_DIM", os.environ.get("MODEL_DIM", "512")))
SCORE_TIE_EMBEDDINGS = bool(int(os.environ.get("TOKENIZER_SCORE_TIE_EMBEDDINGS", os.environ.get("TIE_EMBEDDINGS", "1"))))
SCORE_DEAD_VOCAB_WEIGHT = float(os.environ.get("TOKENIZER_SCORE_DEAD_VOCAB_WEIGHT", "0.02"))
SCORE_ARTIFACT_BUDGET_BYTES = int(os.environ.get("TOKENIZER_SCORE_ARTIFACT_BUDGET_BYTES", "16000000"))
SCORE_VOCAB_ARTIFACT_BYTES_PER_PARAM = float(os.environ.get("TOKENIZER_SCORE_VOCAB_ARTIFACT_BYTES_PER_PARAM", "0.278"))
SCORE_ARTIFACT_WEIGHT = float(os.environ.get("TOKENIZER_SCORE_ARTIFACT_WEIGHT", "1.0"))

PROXY_SUMMARY_PATH = os.environ.get(
    "TOKENIZER_PROXY_SUMMARY_PATH",
    "/Users/simon/Code/parameter-golf/modal_logs/proxy_validate_tokenizers_mix_20260328/summary.json",
)
PROXY_AGGREGATE_PATH = os.environ.get(
    "TOKENIZER_PROXY_AGGREGATE_PATH",
    "/Users/simon/Code/parameter-golf/modal_logs/proxy_calib_aggregate_v2.json",
)
TOKENMONSTER_VOCABS = [
    item
    for item in os.environ.get(
        "TOKENMONSTER_CANDIDATES",
        "english-1024-clean-v1,english-1024-balanced-v1,english-1024-consistent-v1,"
        "english-2048-clean-v1,english-2048-balanced-v1,english-2048-consistent-v1",
    ).split(",")
    if item
]

ROOT = Path(__file__).resolve().parent
PROGRAM_FILE = ROOT / "program_tokenizer_proxy.md"
OUT_DIR = ROOT / "autoresearch" / "tokenizer_proxy_discovery"
RUN_LOG = OUT_DIR / "run.log"
HISTORY_FILE = OUT_DIR / "history.jsonl"
TEXT_SAMPLE_PATH = OUT_DIR / "text_sample.jsonl"
BASELINE_JSON = OUT_DIR / "baseline_eval.json"
FRONTIER_JSON = OUT_DIR / "frontier.json"
LATEST_REVIEW_MD = OUT_DIR / "latest_review.md"
EXPERIMENTS_DIR = OUT_DIR / "experiments"
RUN_MANIFEST = OUT_DIR / "run_manifest.json"
ARCHIVE_DIR = OUT_DIR / "_archives"
CALIBRATION_JSON = OUT_DIR / "proxy_calibration.json"


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
    cmd = ["claude", "-p", prompt, "--model", CLAUDE_MODEL, "--output-format", "json"]
    if CLAUDE_EFFORT:
        cmd.extend(["--effort", CLAUDE_EFFORT])
    last_error: Exception | None = None
    for attempt in range(1, CLAUDE_RETRIES + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                raise RuntimeError(f"Claude {phase} failed:\n{result.stdout}\n{result.stderr}")
            text = result.stdout.strip()
            payload = json.loads(text)
            if payload.get("is_error"):
                raise RuntimeError(f"Claude {phase} returned error payload:\n{text}")
            return payload
        except Exception as exc:
            last_error = exc
            log_line(f"{phase}:error attempt={attempt}/{CLAUDE_RETRIES} error={exc}")
    assert last_error is not None
    raise RuntimeError(f"Claude {phase} failed after {CLAUDE_RETRIES} attempts") from last_error


def append_history(entry: dict) -> None:
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    return [json.loads(line) for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def estimate_vocab_linked_params(vocab_size: int) -> int:
    multiplier = 1 if SCORE_TIE_EMBEDDINGS else 2
    return int(vocab_size) * SCORE_MODEL_DIM * multiplier


def enrich_metrics(metrics: dict, baseline: dict | None = None) -> dict[str, object]:
    enriched = dict(metrics)
    vocab_size = int(enriched["vocab_size"])
    vocab_linked_params = estimate_vocab_linked_params(vocab_size)
    enriched["estimated_vocab_linked_params"] = vocab_linked_params
    enriched["estimated_vocab_artifact_bytes"] = vocab_linked_params * SCORE_VOCAB_ARTIFACT_BYTES_PER_PARAM
    enriched["estimated_artifact_budget_frac"] = enriched["estimated_vocab_artifact_bytes"] / max(
        SCORE_ARTIFACT_BUDGET_BYTES, 1
    )
    enriched["is_tokenmonster"] = 1 if enriched.get("family") == "tokenmonster" else 0
    if baseline is not None:
        enriched["estimated_vocab_linked_params_delta"] = (
            vocab_linked_params - int(baseline["estimated_vocab_linked_params"])
        )
        enriched["estimated_vocab_artifact_bytes_delta"] = (
            float(enriched["estimated_vocab_artifact_bytes"]) - float(baseline["estimated_vocab_artifact_bytes"])
        )
        enriched["tokens_per_byte_delta"] = float(enriched["tokens_per_byte"]) - float(baseline["tokens_per_byte"])
        enriched["dead_vocab_frac_delta"] = float(enriched["dead_vocab_frac"]) - float(baseline["dead_vocab_frac"])
        enriched["artifact_budget_frac_delta"] = (
            float(enriched["estimated_artifact_budget_frac"]) - float(baseline["estimated_artifact_budget_frac"])
        )
    else:
        enriched["estimated_vocab_linked_params_delta"] = 0
        enriched["estimated_vocab_artifact_bytes_delta"] = 0.0
        enriched["tokens_per_byte_delta"] = 0.0
        enriched["dead_vocab_frac_delta"] = 0.0
        enriched["artifact_budget_frac_delta"] = 0.0
    enriched["budget_aware_score"] = (
        float(enriched["tokens_per_byte"])
        + float(enriched["dead_vocab_frac"]) * SCORE_DEAD_VOCAB_WEIGHT
        + float(enriched["estimated_artifact_budget_frac"]) * SCORE_ARTIFACT_WEIGHT
    )
    return enriched


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
        "proxy_summary_path": PROXY_SUMMARY_PATH,
        "proxy_aggregate_path": PROXY_AGGREGATE_PATH,
        "tokenmonster_vocabs": TOKENMONSTER_VOCABS,
        "score_model_dim": SCORE_MODEL_DIM,
        "score_tie_embeddings": SCORE_TIE_EMBEDDINGS,
        "score_dead_vocab_weight": SCORE_DEAD_VOCAB_WEIGHT,
        "score_artifact_budget_bytes": SCORE_ARTIFACT_BUDGET_BYTES,
        "score_vocab_artifact_bytes_per_param": SCORE_VOCAB_ARTIFACT_BYTES_PER_PARAM,
        "score_artifact_weight": SCORE_ARTIFACT_WEIGHT,
        "program_file": str(PROGRAM_FILE.resolve()),
    }
    config_json = json.dumps(config, sort_keys=True)
    config["config_hash"] = sha256(config_json.encode("utf-8")).hexdigest()[:12]
    return config


def archive_incompatible_state(previous: dict[str, object], current: dict[str, object]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = ARCHIVE_DIR / f"{timestamp}_{previous.get('config_hash', 'unknown')}"
    archive_root.mkdir(parents=True, exist_ok=True)
    for path in (
        TEXT_SAMPLE_PATH,
        BASELINE_JSON,
        FRONTIER_JSON,
        HISTORY_FILE,
        LATEST_REVIEW_MD,
        EXPERIMENTS_DIR,
        CALIBRATION_JSON,
    ):
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


def save_frontier(frontier: dict) -> None:
    FRONTIER_JSON.write_text(json.dumps(frontier, indent=2) + "\n", encoding="utf-8")


def resolve_local_candidate_ref(proxy_row: dict[str, object]) -> tuple[str, str]:
    family = str(proxy_row.get("family", "sentencepiece"))
    tokenizer_path = str(proxy_row["tokenizer_path"])
    if family == "tokenmonster":
        return family, tokenizer_path
    path = Path(tokenizer_path)
    if str(path).startswith("/root/parameter-golf/tokenizer_candidates/"):
        experiment_dir = path.parent.name
        return family, str((ROOT / "autoresearch" / "tokenizer_discovery" / "experiments" / experiment_dir / path.name).resolve())
    if path.name == "fineweb_1024_bpe.model":
        return family, str(Path(TOKENIZER_PATH).expanduser().resolve())
    return family, str(path)


def evaluate_tokenizer(family: str, model_ref: str, output_json: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "data" / "evaluate_tokenizer_on_text_sample.py"),
        "--family",
        family,
        "--model-path",
        model_ref,
        "--sample-path",
        str(TEXT_SAMPLE_PATH),
        "--output-json",
        str(output_json),
    ]
    if EVAL_MAX_CHUNKS > 0:
        cmd.extend(["--max-chunks", str(EVAL_MAX_CHUNKS)])
    run_cmd(cmd)
    return json.loads(output_json.read_text(encoding="utf-8"))


def ensure_text_sample() -> None:
    if TEXT_SAMPLE_PATH.exists():
        return
    log_line(
        "text_sample:start "
        f"split={SPLIT} max_tokens={TEXT_SAMPLE_TOKENS} max_shards={TEXT_SAMPLE_MAX_SHARDS} "
        f"tokens_per_shard={TEXT_SAMPLE_TOKENS_PER_SHARD} offset_mode={TEXT_SAMPLE_OFFSET_MODE} "
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


def ensure_baseline() -> dict:
    if BASELINE_JSON.exists():
        return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    baseline = enrich_metrics(
        evaluate_tokenizer("sentencepiece", str(Path(TOKENIZER_PATH).expanduser().resolve()), BASELINE_JSON)
    )
    BASELINE_JSON.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    log_line(
        f"baseline:done tokens_per_byte={baseline['tokens_per_byte']:.6f} "
        f"dead_vocab_frac={baseline['dead_vocab_frac']:.4f} "
        f"artifact_frac={baseline['estimated_artifact_budget_frac']:.4f}"
    )
    return baseline


def fit_proxy_calibration(baseline: dict) -> dict[str, object]:
    aggregate_path = Path(PROXY_AGGREGATE_PATH).expanduser().resolve()
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        family_vocab: dict[str, dict[str, float]] = {}
        exact: dict[str, dict[str, float]] = {}
        for row in aggregate["rows"]:
            key = f"{row['family']}:{int(row['vocab_size'])}"
            family_vocab[key] = {
                "mean": float(row["postquant_val_bpb_mean"]),
                "std": float(row["postquant_val_bpb_std"]),
                "n": int(row["n"]),
            }
            exact[str(row["name"])] = {
                "mean": float(row["postquant_val_bpb_mean"]),
                "std": float(row["postquant_val_bpb_std"]),
                "n": int(row["n"]),
                "family": str(row["family"]),
                "vocab_size": int(row["vocab_size"]),
            }
        calibration = {
            "type": "aggregate",
            "source": str(aggregate_path),
            "num_runs": int(aggregate["num_runs"]),
            "family_vocab": family_vocab,
            "exact": exact,
            "variance_penalty": 0.25,
        }
        CALIBRATION_JSON.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        return calibration

    proxy_path = Path(PROXY_SUMMARY_PATH).expanduser().resolve()
    if not proxy_path.exists():
        calibration = {
            "type": "fallback",
            "source": "fallback",
            "weights": [0.0, 1.0, SCORE_DEAD_VOCAB_WEIGHT, SCORE_ARTIFACT_WEIGHT * 4.0, 0.0],
            "baseline_postquant_val_bpb": None,
            "rows_used": 0,
        }
        CALIBRATION_JSON.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        return calibration

    summary_rows = json.loads(proxy_path.read_text(encoding="utf-8"))
    baseline_row = next((row for row in summary_rows if row["name"] == "baseline_sp1024"), None)
    if baseline_row is None:
        raise ValueError(f"Baseline row missing from proxy summary: {proxy_path}")
    baseline_proxy = float(baseline_row["postquant_val_bpb"])
    X: list[list[float]] = []
    y: list[float] = []
    details: list[dict[str, object]] = []
    for row in summary_rows:
        family, local_ref = resolve_local_candidate_ref(row)
        eval_json = OUT_DIR / f"calib_{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(row['name']))}.json"
        metrics = enrich_metrics(evaluate_tokenizer(family, local_ref, eval_json), baseline)
        target = float(row["postquant_val_bpb"]) - baseline_proxy
        X.append(
            [
                1.0,
                float(metrics["tokens_per_byte_delta"]),
                float(metrics["dead_vocab_frac_delta"]),
                float(metrics["artifact_budget_frac_delta"]),
                float(metrics["is_tokenmonster"]),
            ]
        )
        y.append(target)
        details.append(
            {
                "name": row["name"],
                "family": family,
                "local_ref": local_ref,
                "target_proxy_delta": target,
                "tokens_per_byte_delta": metrics["tokens_per_byte_delta"],
                "artifact_budget_frac_delta": metrics["artifact_budget_frac_delta"],
            }
        )

    import numpy as np

    beta, *_ = np.linalg.lstsq(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), rcond=None)
    calibration = {
        "type": "regression",
        "source": str(proxy_path),
        "weights": [float(x) for x in beta.tolist()],
        "baseline_postquant_val_bpb": baseline_proxy,
        "rows_used": len(X),
        "details": details,
    }
    CALIBRATION_JSON.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    return calibration


def _interpolate_family_vocab_score(family: str, vocab_size: int, calibration: dict[str, object]) -> float:
    rows = []
    for key, payload in calibration["family_vocab"].items():
        fam, vocab = key.split(":", 1)
        if fam != family:
            continue
        rows.append((int(vocab), float(payload["mean"]), float(payload["std"])))
    if not rows:
        raise KeyError(f"No aggregate proxy rows for family={family}")
    rows.sort()
    variance_penalty = float(calibration.get("variance_penalty", 0.25))
    if vocab_size <= rows[0][0]:
        return rows[0][1] + variance_penalty * rows[0][2]
    if vocab_size >= rows[-1][0]:
        return rows[-1][1] + variance_penalty * rows[-1][2]
    for (v0, m0, s0), (v1, m1, s1) in zip(rows, rows[1:], strict=False):
        if v0 <= vocab_size <= v1:
            ratio = (vocab_size - v0) / max(v1 - v0, 1)
            mean = m0 + ratio * (m1 - m0)
            std = s0 + ratio * (s1 - s0)
            return mean + variance_penalty * std
    return rows[-1][1] + variance_penalty * rows[-1][2]


def calibrated_score(metrics: dict, calibration: dict[str, object], proposal: dict | None = None) -> float:
    if calibration.get("type") == "aggregate":
        if proposal and proposal.get("type") == "tokenmonster_eval":
            vocab_ref = str(proposal["params"]["vocab_ref"])
            exact = calibration["exact"].get(vocab_ref)
            if exact is not None:
                return float(exact["mean"]) + float(calibration.get("variance_penalty", 0.25)) * float(exact["std"])
        return _interpolate_family_vocab_score(str(metrics["family"]), int(metrics["vocab_size"]), calibration)
    weights = [float(x) for x in calibration["weights"]]
    features = [
        1.0,
        float(metrics["tokens_per_byte_delta"]),
        float(metrics["dead_vocab_frac_delta"]),
        float(metrics["artifact_budget_frac_delta"]),
        float(metrics["is_tokenmonster"]),
    ]
    return sum(w * x for w, x in zip(weights, features, strict=True))


def load_frontier(baseline: dict, calibration: dict[str, object]) -> dict:
    if FRONTIER_JSON.exists():
        return json.loads(FRONTIER_JSON.read_text(encoding="utf-8"))
    score = calibrated_score(baseline, calibration)
    frontier = {
        "best_label": "baseline_sp1024",
        "best_score": score,
        "best_metrics": baseline,
        "best_experiment_id": None,
        "headline": "baseline control",
        "frontier_update": "starting point",
        "next_direction": "test smaller, simpler tokenizers first",
        "strategy_experiment_id": None,
        "strategy_score": score,
    }
    save_frontier(frontier)
    return frontier


def canonical_params(proposal_type: str, params: dict) -> dict[str, object]:
    if proposal_type == "sentencepiece_train":
        return {
            "type": proposal_type,
            "vocab_size": int(params["vocab_size"]),
            "model_type": str(params.get("model_type", "bpe")),
            "character_coverage": round(float(params.get("character_coverage", 0.9995)), 8),
            "input_sentence_size": int(params.get("input_sentence_size", 200000)),
            "max_chunks": int(params.get("max_chunks", 0)),
        }
    if proposal_type == "tokenmonster_eval":
        return {
            "type": proposal_type,
            "vocab_ref": str(params["vocab_ref"]),
        }
    raise ValueError(f"Unsupported proposal type: {proposal_type}")


def params_key(proposal_type: str, params: dict) -> str:
    return json.dumps(canonical_params(proposal_type, params), sort_keys=True, separators=(",", ":"))


def tried_param_keys() -> set[str]:
    keys: set[str] = set()
    for row in load_history():
        proposal_type = row.get("type")
        params = row.get("params")
        if isinstance(proposal_type, str) and isinstance(params, dict):
            keys.add(params_key(proposal_type, params))
    return keys


def recent_history_summary(limit: int = 12) -> str:
    rows = load_history()[-limit:]
    if not rows:
        return "- no experiments yet"
    lines = []
    for row in rows:
        metrics = row.get("metrics", {})
        lines.append(
            f"- exp {row['experiment_id']}: type={row['type']} family={metrics.get('family')} "
            f"vocab={metrics.get('vocab_size')} tpb={metrics.get('tokens_per_byte')} "
            f"artifact_frac={metrics.get('estimated_artifact_budget_frac')} "
            f"calibrated_score={row.get('score')} keep={row.get('decision', {}).get('keep')} "
            f"params={canonical_params(str(row['type']), row.get('params', {}))}"
        )
    return "\n".join(lines)


def build_prompt(baseline: dict, frontier: dict, calibration: dict[str, object]) -> str:
    program = PROGRAM_FILE.read_text(encoding="utf-8")
    recent_tried = "\n".join(f"- {key}" for key in sorted(tried_param_keys())[-20:]) or "- none"
    allowed_tokenmonster = "\n".join(f"- {v}" for v in TOKENMONSTER_VOCABS)
    return f"""You are running proxy-calibrated tokenizer autoresearch for the OpenAI parameter-golf challenge.

Research program:
{program}

Current baseline metrics:
{json.dumps(baseline, indent=2)}

Proxy calibration:
{json.dumps(calibration, indent=2)}

Current frontier:
{json.dumps(frontier, indent=2)}

Recent experiment history:
{recent_history_summary()}

Recently tried canonical params:
{recent_tried}

Allowed TokenMonster vocab refs:
{allowed_tokenmonster}

You may propose exactly one experiment of one of these forms:
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

or

{{
  "description": "...",
  "type": "tokenmonster_eval",
  "params": {{
    "vocab_ref": "english-2048-clean-v1"
  }}
}}

Rules:
- optimize the empirical proxy score, not tokens_per_byte alone
- prefer smaller and simpler tokenizers first
- avoid larger vocabularies unless the calibrated case is strong
- search near baseline scale for SentencePiece
- TokenMonster candidates should usually be 1024 or 2048 before 4096
- output JSON only
"""


def propose_experiment(baseline: dict, frontier: dict, calibration: dict[str, object]) -> dict:
    proposal = run_claude_json(build_prompt(baseline, frontier, calibration), timeout=PROPOSAL_TIMEOUT, phase="proposal")
    proposal_type = proposal.get("type")
    params = proposal.get("params")
    if proposal_type not in {"sentencepiece_train", "tokenmonster_eval"}:
        raise ValueError(f"Unsupported proposal type: {proposal_type}")
    if not isinstance(params, dict):
        raise ValueError("Proposal must include params")
    key = params_key(str(proposal_type), params)
    if key in tried_param_keys():
        raise ValueError(f"Claude proposed an already-tried experiment: {canonical_params(str(proposal_type), params)}")
    if proposal_type == "tokenmonster_eval" and str(params.get("vocab_ref")) not in TOKENMONSTER_VOCABS:
        raise ValueError(f"Unsupported TokenMonster vocab_ref: {params.get('vocab_ref')}")
    return proposal


def run_experiment(experiment_id: int, proposal: dict, baseline: dict, calibration: dict[str, object]) -> dict:
    exp_dir = EXPERIMENTS_DIR / f"{experiment_id:04d}"
    exp_dir.mkdir(exist_ok=True)
    params = proposal["params"]
    proposal_type = str(proposal["type"])
    log_line(f"experiment:{experiment_id}:start type={proposal_type} params={params}")
    if proposal_type == "sentencepiece_train":
        prefix = exp_dir / "tokenizer"
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
        raw_metrics = evaluate_tokenizer("sentencepiece", str(prefix.with_suffix(".model")), exp_dir / "eval.json")
    else:
        raw_metrics = evaluate_tokenizer("tokenmonster", str(params["vocab_ref"]), exp_dir / "eval.json")
    metrics = enrich_metrics(raw_metrics, baseline)
    score = calibrated_score(metrics, calibration, proposal)
    (exp_dir / "eval.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "description": proposal.get("description", ""),
        "type": proposal_type,
        "params": params,
        "metrics": metrics,
        "score": score,
    }
    log_line(
        f"experiment:{experiment_id}:done family={metrics['family']} vocab={metrics['vocab_size']} "
        f"tpb={metrics['tokens_per_byte']:.6f} artifact_frac={metrics['estimated_artifact_budget_frac']:.4f} "
        f"calibrated_score={score:.6f}"
    )
    return entry


def build_review_prompt(baseline: dict, frontier: dict, result: dict, calibration: dict[str, object]) -> str:
    return f"""You are reviewing a proxy-calibrated tokenizer experiment for the OpenAI parameter-golf challenge.

Baseline metrics:
{json.dumps(baseline, indent=2)}

Proxy calibration:
{json.dumps(calibration, indent=2)}

Current frontier:
{json.dumps(frontier, indent=2)}

New experiment result:
{json.dumps(result, indent=2)}

Lower calibrated score is better.

Return JSON only:
{{
  "keep": true,
  "headline": "short judgment",
  "reasoning": "2-4 sentence explanation",
  "frontier_update": "what this changes about the search strategy",
  "next_direction": "one short sentence about what to try next"
}}

Rules:
- keep=true only if the calibrated score meaningfully improves the frontier or reveals an important strategic shift
- prefer smaller/simple tokenizers when scores are close
- use the proxy-calibrated score as primary evidence
"""


def review_experiment(baseline: dict, frontier: dict, result: dict, calibration: dict[str, object]) -> dict:
    log_line(f"review:start experiment={result['experiment_id']}")
    decision = run_claude_json(
        build_review_prompt(baseline, frontier, result, calibration),
        timeout=REVIEW_TIMEOUT,
        phase="review",
    )
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
    if result["kept"] and result_score < current_best:
        frontier = {
            **frontier,
            "headline": decision.get("headline", ""),
            "frontier_update": decision.get("frontier_update", ""),
            "next_direction": decision.get("next_direction", ""),
            "strategy_experiment_id": result["experiment_id"],
            "strategy_score": result_score,
        }
        frontier.update(
            {
                "best_label": f"experiment_{result['experiment_id']:04d}",
                "best_score": result_score,
                "best_metrics": result["metrics"],
                "best_experiment_id": result["experiment_id"],
            }
        )
        save_frontier(frontier)
        log_line(f"frontier:update best=experiment_{result['experiment_id']:04d} score={result_score:.6f}")
    elif result["kept"]:
        frontier = {
            **frontier,
            "strategy_experiment_id": result["experiment_id"],
            "strategy_score": result_score,
            "strategy_headline": decision.get("headline", ""),
            "strategy_frontier_update": decision.get("frontier_update", ""),
            "strategy_next_direction": decision.get("next_direction", ""),
        }
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
    calibration = fit_proxy_calibration(baseline)
    if calibration.get("type") == "aggregate":
        log_line(
            f"calibration:done type=aggregate num_runs={calibration['num_runs']} "
            f"family_vocab_rows={len(calibration['family_vocab'])}"
        )
    else:
        log_line(
            f"calibration:done rows={calibration['rows_used']} "
            f"weights={','.join(f'{w:.4f}' for w in calibration['weights'])}"
        )
    frontier = load_frontier(baseline, calibration)
    existing = load_history()
    start_id = 1 + max((row.get("experiment_id", 0) for row in existing), default=0)
    for experiment_id in range(start_id, start_id + MAX_EXPERIMENTS):
        proposal = propose_experiment(baseline, frontier, calibration)
        result = run_experiment(experiment_id, proposal, baseline, calibration)
        decision = review_experiment(baseline, frontier, result, calibration)
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
