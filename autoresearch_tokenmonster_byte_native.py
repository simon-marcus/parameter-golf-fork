"""
Exactness-first TokenMonster byte-native discovery loop.
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

DATASET_DIR = os.environ.get("DATASET_DIR", "/Users/simon/Code/parameter-golf-local/data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/Users/simon/Code/parameter-golf-local/data/tokenizers/fineweb_1024_bpe.model")
TEXT_SAMPLE_TOKENS = int(os.environ.get("TEXT_SAMPLE_TOKENS", "2_000_000"))
TEXT_SAMPLE_MAX_SHARDS = int(os.environ.get("TEXT_SAMPLE_MAX_SHARDS", "10"))
TEXT_SAMPLE_TOKENS_PER_SHARD = int(os.environ.get("TEXT_SAMPLE_TOKENS_PER_SHARD", "2_000_000"))
TEXT_SAMPLE_OFFSET_MODE = os.environ.get("TEXT_SAMPLE_OFFSET_MODE", "staggered")
TEXT_SAMPLE_CHUNK_TOKENS = int(os.environ.get("TEXT_SAMPLE_CHUNK_TOKENS", "2048"))
VAL_SAMPLE_DOCS = int(os.environ.get("TOKENMONSTER_BYTE_NATIVE_VAL_DOCS", "200"))
ROOT = Path(__file__).resolve().parent
TOKENMONSTER_PYTHON = os.environ.get(
    "TOKENMONSTER_PYTHON",
    str(ROOT / ".venv-tokenmonster" / "bin" / "python"),
)
PROGRAM_FILE = ROOT / "program_tokenmonster_byte_native.md"
OUT_DIR = ROOT / "autoresearch" / "tokenmonster_byte_native_discovery"
RUN_LOG = OUT_DIR / "run.log"
HISTORY_FILE = OUT_DIR / "history.jsonl"
TEXT_SAMPLE_PATH = OUT_DIR / "text_sample.jsonl"
FRONTIER_JSON = OUT_DIR / "frontier.json"
LATEST_REVIEW_MD = OUT_DIR / "latest_review.md"
EXPERIMENTS_DIR = OUT_DIR / "experiments"
RUN_MANIFEST = OUT_DIR / "run_manifest.json"
ARCHIVE_DIR = OUT_DIR / "_archives"
BASE_CANDIDATES = [
    str(ROOT / "autoresearch" / "tokenmonster_discovery" / "experiments" / "0054" / "candidate.vocab"),
    str(ROOT / "autoresearch" / "tokenmonster_discovery" / "experiments" / "0045" / "candidate.vocab"),
    "english-1024-clean-v1",
    "english-1024-balanced-v1",
    "english-1024-consistent-v1",
]


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
        "val_sample_docs": VAL_SAMPLE_DOCS,
        "program_file": str(PROGRAM_FILE.resolve()),
        "base_candidates": BASE_CANDIDATES,
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
        TOKENMONSTER_PYTHON,
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


def audit_candidate(model_ref: str, output_json: Path) -> dict:
    cmd = [
        TOKENMONSTER_PYTHON,
        str(ROOT / "data" / "sample_audit_tokenmonster_vocab.py"),
        "--source-root",
        str(Path(DATASET_DIR).resolve().parents[1]),
        "--vocab",
        model_ref,
        "--docs",
        str(VAL_SAMPLE_DOCS),
        "--bytes-mode",
        "latin-1",
    ]
    text = run_cmd(cmd)
    output_json.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return json.loads(text)


def analyze_bad_docs(model_ref: str, output_json: Path) -> dict:
    cmd = [
        TOKENMONSTER_PYTHON,
        str(ROOT / "data" / "analyze_tokenmonster_bad_docs.py"),
        "--source-root",
        str(Path(DATASET_DIR).resolve().parents[1]),
        "--vocab",
        model_ref,
        "--docs",
        str(VAL_SAMPLE_DOCS),
    ]
    text = run_cmd(cmd)
    output_json.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return json.loads(text)


def load_frontier() -> dict[str, object]:
    default_frontier = {
        "best_valid_label": "",
        "best_valid_score": 1e18,
        "best_valid_metrics": {},
        "best_valid_audit": {},
        "best_invalid_label": "",
        "best_invalid_score": 1e18,
        "best_invalid_metrics": {},
        "best_invalid_audit": {},
        "headline": "no exact byte-native frontier yet",
        "next_direction": "start from the strongest 1024 TokenMonster family members",
    }
    if FRONTIER_JSON.exists():
        frontier = json.loads(FRONTIER_JSON.read_text(encoding="utf-8"))
        migrated = dict(default_frontier)
        migrated.update(frontier)
        if "best_score" in frontier:
            migrated["best_invalid_score"] = frontier.get("best_score", migrated["best_invalid_score"])
        if "best_label" in frontier:
            migrated["best_invalid_label"] = frontier.get("best_label", migrated["best_invalid_label"])
        if "best_metrics" in frontier:
            migrated["best_invalid_metrics"] = frontier.get("best_metrics", migrated["best_invalid_metrics"])
        if "best_audit" in frontier:
            migrated["best_invalid_audit"] = frontier.get("best_audit", migrated["best_invalid_audit"])
        if migrated != frontier:
            FRONTIER_JSON.write_text(json.dumps(migrated, indent=2) + "\n", encoding="utf-8")
        return migrated
    FRONTIER_JSON.write_text(json.dumps(default_frontier, indent=2) + "\n", encoding="utf-8")
    return default_frontier


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
        f"- exp {row['experiment_id']}: type={row['type']} score={row.get('score')} params={row['params']}"
        for row in rows
    )


def recent_failure_summary(limit: int = 8) -> dict[str, object]:
    rows = load_history()[-limit:]
    token_counter: dict[str, int] = {}
    raw_counter: dict[str, int] = {}
    for row in rows:
        audit = row.get("audit") or {}
        for token_id, count in audit.get("bad_token_ids_top", []):
            key = str(token_id)
            token_counter[key] = token_counter.get(key, 0) + int(count)
        for raw, count in audit.get("bad_raw_tokens_top", []):
            key = str(raw)
            raw_counter[key] = raw_counter.get(key, 0) + int(count)
    top_ids = sorted(token_counter.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_raw = sorted(raw_counter.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "top_bad_token_ids": [[int(k), int(v)] for k, v in top_ids],
        "top_bad_raw_tokens": [[k, int(v)] for k, v in top_raw],
    }


def frontier_summary(frontier: dict[str, object]) -> dict[str, object]:
    audit = frontier.get("best_invalid_audit") or {}
    metrics = frontier.get("best_invalid_metrics") or {}
    return {
        "best_invalid_label": frontier.get("best_invalid_label", ""),
        "bad_docs": audit.get("bad_docs"),
        "ok_docs": audit.get("ok_docs"),
        "byte_drift": audit.get("byte_drift"),
        "tokens_per_byte": metrics.get("tokens_per_byte"),
        "top_bad_raw_tokens": audit.get("bad_raw_tokens_top", [])[:8],
        "top_bad_token_ids": audit.get("bad_token_ids_top", [])[:8],
        "cluster_counts": audit.get("cluster_counts", {}),
        "nonascii_failure_tokens": audit.get("bad_fail_nonascii_raw_top", [])[:8],
    }


def safe_target_tokens(frontier: dict[str, object]) -> list[str]:
    audit = frontier.get("best_invalid_audit") or {}
    out: list[str] = []
    banned = {"C", "D", "DC", "DW", "W", " ", ""}
    for raw, _count in audit.get("bad_raw_tokens_top", []):
        raw = str(raw)
        if raw in banned:
            continue
        if len(raw) < 2:
            continue
        if re.fullmatch(r"[A-Z]+", raw):
            continue
        if raw not in out:
            out.append(raw)
    return out[:8]


def high_byte_target_tokens(frontier: dict[str, object], rows: list[dict] | None = None) -> list[str]:
    audit = frontier.get("best_invalid_audit") or {}
    if not audit.get("bad_fail_nonascii_raw_top") and rows:
        frontier_bad_docs = int(audit.get("bad_docs", 10**9))
        frontier_drift = int(audit.get("byte_drift", 10**9))
        for row in reversed(rows):
            row_audit = row.get("audit") or {}
            if not row_audit.get("bad_fail_nonascii_raw_top"):
                continue
            if int(row_audit.get("bad_docs", 10**9)) == frontier_bad_docs and int(row_audit.get("byte_drift", 10**9)) == frontier_drift:
                audit = row_audit
                break
    if not audit.get("bad_fail_nonascii_raw_top"):
        model_path = str((frontier.get("best_invalid_metrics") or {}).get("model_path") or "")
        if model_path:
            cache_path = OUT_DIR / "_frontier_bad_docs_analysis.json"
            try:
                analysis = analyze_bad_docs(model_path, cache_path)
            except Exception:
                analysis = {}
            raw_counter: dict[str, int] = {}
            for failure in analysis.get("failures", []):
                for raw in failure.get("raw_tokens_head", []):
                    raw = str(raw)
                    if any(ord(ch) >= 128 or ord(ch) < 32 for ch in raw):
                        raw_counter[raw] = raw_counter.get(raw, 0) + 1
            if raw_counter:
                audit = dict(audit)
                audit["bad_fail_nonascii_raw_top"] = sorted(
                    [[raw, count] for raw, count in raw_counter.items()],
                    key=lambda item: (-int(item[1]), item[0]),
                )[:8]
    out: list[str] = []
    lead_byte_literals = {"Â", "Ã", "Å", "Ì", "â"}
    deferred: list[str] = []
    for raw, _count in audit.get("bad_fail_nonascii_raw_top", []):
        raw = str(raw)
        if not any(ord(ch) >= 128 or ord(ch) < 32 for ch in raw):
            continue
        if raw in lead_byte_literals:
            if raw not in deferred:
                deferred.append(raw)
            continue
        if raw not in out:
            out.append(raw)
    for raw, _count in audit.get("bad_raw_tokens_top", []):
        raw = str(raw)
        if not any(ord(ch) >= 128 or ord(ch) < 32 for ch in raw):
            continue
        if raw in lead_byte_literals:
            if raw not in deferred:
                deferred.append(raw)
            continue
        if raw not in out:
            out.append(raw)
    out.extend(tok for tok in deferred if tok not in out)
    return out[:8]


def remaining_safe_targets(frontier: dict[str, object], rows: list[dict]) -> list[str]:
    safe = safe_target_tokens(frontier)
    tried_literals: set[str] = set()
    for row in rows:
        params = row.get("params", {})
        if params.get("capcode", 2) != 2:
            continue
        for pattern in params.get("delete_raw_regex", []):
            pattern = str(pattern)
            if pattern.startswith("^") and pattern.endswith("$"):
                tried_literals.add(re.sub(r"\\(.)", r"\1", pattern[1:-1]))
    return [tok for tok in safe if tok not in tried_literals]


def remaining_high_byte_targets(frontier: dict[str, object], rows: list[dict]) -> list[str]:
    safe = high_byte_target_tokens(frontier, rows)
    tried_literals: set[str] = set()
    for row in rows:
        params = row.get("params", {})
        if params.get("capcode", 2) != 2:
            continue
        for pattern in params.get("delete_raw_regex", []):
            pattern = str(pattern)
            if pattern.startswith("^") and pattern.endswith("$"):
                tried_literals.add(re.sub(r"\\(.)", r"\1", pattern[1:-1]))
    harmful = harmful_target_tokens(rows)
    return [tok for tok in safe if tok not in tried_literals and tok not in harmful]


def harmful_target_tokens(rows: list[dict]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        params = row.get("params", {})
        audit = row.get("audit", {})
        if params.get("capcode", 2) != 2:
            continue
        if int(audit.get("bad_docs", 10**9)) <= 20:
            continue
        for pattern in params.get("delete_raw_regex", []):
            pattern = str(pattern)
            if pattern.startswith("^") and pattern.endswith("$"):
                out.add(re.sub(r"\\(.)", r"\1", pattern[1:-1]))
    return out


def rejected_strategy_summary(rows: list[dict]) -> list[dict[str, object]]:
    rejected: list[dict[str, object]] = []
    for row in rows:
        params = row.get("params", {})
        audit = row.get("audit", {})
        bad_docs = int(audit.get("bad_docs", 10**9))
        byte_drift = int(audit.get("byte_drift", 10**9))
        if bad_docs < 150:
            continue
        reason = None
        if params.get("capcode") in (0, 1):
            reason = f"capcode={params['capcode']} caused {bad_docs} bad docs"
        elif any(p in {"^(C|D|DC|DW)$", "^(D|DC|DW)$", "^(C|D|DC)$", "^[A-Z]$"} for p in params.get("delete_raw_regex", [])):
            reason = f"broad deletion {params.get('delete_raw_regex')} caused {bad_docs} bad docs"
        elif bad_docs > 20 and params.get("delete_raw_regex"):
            reason = f"targeted deletion {params.get('delete_raw_regex')} still regressed to {bad_docs} bad docs"
        if reason:
            rejected.append(
                {
                    "experiment_id": row.get("experiment_id"),
                    "params": params,
                    "reason": reason,
                    "byte_drift": byte_drift,
                }
            )
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in rejected:
        key = json.dumps(item["params"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:6]


def working_region_summary(rows: list[dict]) -> list[dict[str, object]]:
    working: list[dict[str, object]] = []
    for row in rows:
        params = row.get("params", {})
        audit = row.get("audit", {})
        if int(audit.get("bad_docs", 10**9)) == 13 and int(audit.get("byte_drift", 10**9)) == -95:
            working.append(
                {
                    "experiment_id": row.get("experiment_id"),
                    "base_vocab": params.get("base_vocab"),
                    "capcode": params.get("capcode", 2),
                    "delete_raw_regex": params.get("delete_raw_regex", []),
                    "tokens_per_byte": row.get("metrics", {}).get("tokens_per_byte"),
                }
            )
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in working:
        key = json.dumps(
            {
                "base_vocab": item["base_vocab"],
                "capcode": item["capcode"],
                "delete_raw_regex": item["delete_raw_regex"],
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:6]


def validate_proposal(proposal: dict, frontier: dict[str, object]) -> str | None:
    if proposal.get("type") != "byte_native_build":
        return "unsupported type"
    params = proposal.get("params") or {}
    capcode = params.get("capcode", 2)
    delete_raw = [str(x) for x in params.get("delete_raw_regex", [])]
    if capcode in (0, 1):
        return f"global capcode={capcode} already shown catastrophic"
    banned_patterns = {"^(C|D|DC|DW)$", "^(D|DC|DW)$", "^(C|D|DC)$", "^[A-Z]$"}
    if any(p in banned_patterns for p in delete_raw):
        return "broad capcode/uppercase deletion already shown catastrophic"
    if len(delete_raw) > 2:
        return "too many deletion families at once"
    safe_targets = set(safe_target_tokens(frontier))
    history_rows = load_history()
    high_byte_targets = set(high_byte_target_tokens(frontier, history_rows))
    allowed_targets = safe_targets | high_byte_targets
    harmful_targets = harmful_target_tokens(history_rows)
    lead_byte_literals = {"Â", "Ã", "Å", "Ì", "â"}
    matched_literals: list[str] = []
    for pattern in delete_raw:
        if pattern.startswith("^") and pattern.endswith("$"):
            literal = re.sub(r"\\(.)", r"\1", pattern[1:-1])
            if literal in allowed_targets:
                matched_literals.append(literal)
                continue
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return f"invalid regex: {exc}"
        matched = [tok for tok in allowed_targets if compiled.fullmatch(tok)]
        if len(matched) != 1:
            display = [f"^{re.escape(tok)}$" for tok in sorted(allowed_targets)]
            return f"deletion target must uniquely match one shortlisted token: {display}"
        matched_literals.append(matched[0])
    if len(set(matched_literals)) != len(matched_literals):
        return "duplicate deletion targets are not allowed"
    harmful_matched = [tok for tok in matched_literals if tok in harmful_targets]
    if harmful_matched:
        return f"targets already shown harmful: {harmful_matched}"
    remaining_nonlead = [tok for tok in remaining_high_byte_targets(frontier, history_rows) if tok not in lead_byte_literals]
    if remaining_nonlead and any(tok in lead_byte_literals for tok in matched_literals):
        return "do not delete likely UTF-8 lead-byte coverage while non-lead high-byte suspects remain"
    return None


def build_prompt(frontier: dict) -> str:
    program = PROGRAM_FILE.read_text(encoding="utf-8")
    rows = load_history()
    failure_summary = recent_failure_summary()
    frontier_view = frontier_summary(frontier)
    rejected = rejected_strategy_summary(rows)
    working = working_region_summary(rows)
    safe_targets = safe_target_tokens(frontier)
    remaining_targets = remaining_safe_targets(frontier, rows)
    high_byte_targets = high_byte_target_tokens(frontier, rows)
    remaining_high_byte = remaining_high_byte_targets(frontier, rows)
    harmful_targets = sorted(harmful_target_tokens(rows))
    return f"""You are running an exactness-first TokenMonster byte-native discovery loop for Parameter Golf.

Program:
{program}

Current least-bad invalid frontier:
{json.dumps(frontier_view, indent=2)}

Working region so far:
{json.dumps(working, indent=2)}

Rejected strategy classes:
{json.dumps(rejected, indent=2)}

Recent failure clusters:
{json.dumps(failure_summary, indent=2)}

Safe exact-token deletion shortlist:
{json.dumps(safe_targets, indent=2)}

Remaining untried safe-token deletions:
{json.dumps(remaining_targets, indent=2)}

High-byte / non-ASCII suspect tokens:
{json.dumps(high_byte_targets, indent=2)}

Remaining untried high-byte deletions:
{json.dumps(remaining_high_byte, indent=2)}

Harmful deletion targets already shown to make exactness much worse:
{json.dumps(harmful_targets, indent=2)}

Base candidates:
{json.dumps(BASE_CANDIDATES, indent=2)}

Allowed experiment JSON:
{{
  "description": "...",
  "type": "byte_native_build",
  "params": {{
    "base_vocab": "{BASE_CANDIDATES[0]}",
    "capcode": 2,
    "delete_raw_regex": ["^Ã$", "^©$"]
  }}
}}

Rules:
- output JSON only
- prioritize exact byte preservation over tokenizer efficiency
- regexes must match the exported raw token strings from `get_dictionary()`
- act like a careful repair engineer, not a creative explorer
- use the current least-bad frontier as the anchor
- do not repeat rejected strategy classes
- current evidence says all remaining failures are `non_ascii_or_utf8`; prioritize high-byte suspects over ASCII cleanup
- prefer literal non-ASCII crumbs actually observed in failing docs (for example `©`, `°`, `¦`, `¥`, control-byte literals) before deleting likely UTF-8 lead-byte coverage like `Â`, `Ã`, or `â`
- use one of these mutation types:
  - keep `capcode=2` and delete one remaining high-byte suspect token by exact regex
  - keep `capcode=2` and delete two remaining high-byte suspect tokens by exact regex if they are tightly related
  - keep `capcode=2` and delete one remaining safe ASCII token only if you can justify it against the non-ASCII evidence
  - switch to a different base vocab
- if you switch base vocab, keep `capcode=2` and no deletions unless there is a very specific reason
- avoid broad speculative edits
- leave out ideas already disproven by the rejected list
- do not propose normalized vocabularies
"""


def propose_experiment(frontier: dict) -> dict:
    for attempt in range(1, CLAUDE_RETRIES + 2):
        proposal = run_claude_json(build_prompt(frontier), timeout=PROPOSAL_TIMEOUT, phase="proposal")
        rejection = validate_proposal(proposal, frontier)
        if rejection:
            log_line(f"proposal:rejected attempt={attempt} reason={rejection} params={proposal.get('params')}")
            continue
        key = json.dumps({"type": proposal["type"], "params": proposal["params"]}, sort_keys=True)
        if key not in tried_keys():
            return proposal
        log_line(f"proposal:duplicate attempt={attempt} type={proposal['type']} params={proposal['params']}")
    rows = load_history()
    remaining_high_byte = remaining_high_byte_targets(frontier, rows)
    remaining_targets = remaining_safe_targets(frontier, rows)
    base = str((frontier.get("best_invalid_metrics") or {}).get("model_path") or "english-1024-clean-v1")
    if remaining_high_byte:
        tok = remaining_high_byte[0]
        return {
            "description": f"Deterministic fallback: exact-token deletion for remaining high-byte suspect {tok!r} on current best invalid base",
            "type": "byte_native_build",
            "params": {
                "base_vocab": base,
                "capcode": 2,
                "delete_raw_regex": [f"^{re.escape(tok)}$"],
            },
        }
    if remaining_targets:
        tok = remaining_targets[0]
        return {
            "description": f"Deterministic fallback: exact-token deletion for remaining shortlist token {tok!r} on current best invalid base",
            "type": "byte_native_build",
            "params": {
                "base_vocab": base,
                "capcode": 2,
                "delete_raw_regex": [f"^{re.escape(tok)}$"],
            },
        }
    raise RuntimeError("Could not obtain a non-duplicate byte-native proposal")


def score_candidate(metrics: dict, audit: dict) -> float:
    # Exactness first: smaller bad-doc count and smaller absolute byte drift are overwhelmingly favored.
    bad_docs = int(audit.get("bad_docs", 10**9))
    byte_drift = abs(int(audit.get("byte_drift", 10**9)))
    return bad_docs * 1_000_000 + byte_drift * 1_000 + float(metrics.get("tokens_per_byte", 1.0))


def rebuild_frontier_from_history(rows: list[dict]) -> dict[str, object]:
    frontier = {
        "best_valid_label": "",
        "best_valid_score": 1e18,
        "best_valid_metrics": {},
        "best_valid_audit": {},
        "best_invalid_label": "",
        "best_invalid_score": 1e18,
        "best_invalid_metrics": {},
        "best_invalid_audit": {},
        "headline": "no exact byte-native frontier yet",
        "next_direction": "start from the strongest 1024 TokenMonster family members",
    }
    latest_decision: dict[str, object] | None = None
    for row in rows:
        if "score" not in row:
            row["score"] = score_candidate(row["metrics"], row["audit"])
        if not row.get("kept"):
            continue
        latest_decision = row.get("decision") or latest_decision
        bad_docs = int(row["audit"].get("bad_docs", 10**9))
        byte_drift = int(row["audit"].get("byte_drift", 10**9))
        label = f"experiment_{int(row['experiment_id']):04d}"
        if bad_docs == 0 and byte_drift == 0:
            if float(row["score"]) < float(frontier["best_valid_score"]):
                frontier["best_valid_label"] = label
                frontier["best_valid_score"] = float(row["score"])
                frontier["best_valid_metrics"] = row["metrics"]
                frontier["best_valid_audit"] = row["audit"]
        elif float(row["score"]) < float(frontier["best_invalid_score"]):
            frontier["best_invalid_label"] = label
            frontier["best_invalid_score"] = float(row["score"])
            frontier["best_invalid_metrics"] = row["metrics"]
            frontier["best_invalid_audit"] = row["audit"]
            latest_decision = row.get("decision") or latest_decision
    if latest_decision:
        frontier["headline"] = str(latest_decision.get("headline", frontier["headline"]))
        frontier["next_direction"] = str(latest_decision.get("next_direction", frontier["next_direction"]))
    return frontier


def run_experiment(experiment_id: int, proposal: dict) -> dict:
    exp_dir = EXPERIMENTS_DIR / f"{experiment_id:04d}"
    exp_dir.mkdir(exist_ok=True)
    params = proposal["params"]
    proposal_type = proposal["type"]
    log_line(f"experiment:{experiment_id}:start type={proposal_type} params={params}")
    if proposal_type != "byte_native_build":
        raise ValueError(f"unsupported proposal type: {proposal_type}")
    model_path = exp_dir / "candidate.yaml"
    cmd = [
        TOKENMONSTER_PYTHON,
        str(ROOT / "data" / "build_tokenmonster_byte_native_variant.py"),
        "--base-vocab",
        str(params["base_vocab"]),
        "--output-path",
        str(model_path),
        "--charset",
        "none",
        "--normalization",
        "none",
    ]
    if "capcode" in params:
        cmd.extend(["--capcode", str(params["capcode"])])
    for pattern in params.get("delete_raw_regex", []):
        cmd.extend(["--delete-raw-regex", str(pattern)])
    for pattern in params.get("delete_decoded_regex", []):
        cmd.extend(["--delete-decoded-regex", str(pattern)])
    run_cmd(cmd)
    metrics = evaluate_tokenizer(str(model_path), exp_dir / "eval.json")
    audit = audit_candidate(str(model_path), exp_dir / "audit.json")
    if int(audit.get("bad_docs", 0)) > 0:
        analysis = analyze_bad_docs(str(model_path), exp_dir / "bad_docs_analysis.json")
        audit["cluster_counts"] = analysis.get("clusters", {})
        raw_counter: dict[str, int] = {}
        for failure in analysis.get("failures", []):
            for raw in failure.get("raw_tokens_head", []):
                raw = str(raw)
                if any(ord(ch) >= 128 or ord(ch) < 32 for ch in raw):
                    raw_counter[raw] = raw_counter.get(raw, 0) + 1
        audit["bad_fail_nonascii_raw_top"] = sorted(
            [[raw, count] for raw, count in raw_counter.items()],
            key=lambda item: (-int(item[1]), item[0]),
        )[:8]
        audit["bad_failure_examples"] = [
            {
                "doc_index": failure.get("doc_index"),
                "cluster": failure.get("cluster"),
                "delta": failure.get("delta"),
                "nonascii_raw_tokens": [
                    str(raw)
                    for raw in failure.get("raw_tokens_head", [])
                    if any(ord(ch) >= 128 or ord(ch) < 32 for ch in str(raw))
                ][:8],
            }
            for failure in analysis.get("failures", [])[:6]
        ]
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "description": proposal.get("description", ""),
        "type": proposal_type,
        "params": params,
        "metrics": metrics,
        "audit": audit,
    }
    log_line(
        f"experiment:{experiment_id}:done ok_docs={audit.get('ok_docs')} bad_docs={audit.get('bad_docs')} "
        f"drift={audit.get('byte_drift')} tpb={metrics.get('tokens_per_byte'):.6f}"
    )
    return entry


def build_review_prompt(frontier: dict, result: dict) -> str:
    rejected = rejected_strategy_summary(load_history())
    safe_targets = safe_target_tokens(frontier)
    return f"""You are reviewing an exactness-first TokenMonster byte-native result.

Current least-bad invalid frontier:
{json.dumps(frontier_summary(frontier), indent=2)}

Rejected strategy classes:
{json.dumps(rejected, indent=2)}

Safe exact-token deletion shortlist:
{json.dumps(safe_targets, indent=2)}

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
- exactness is the main judge
- if bad_docs > 0, be very conservative
- valid frontier means: `bad_docs == 0` and `byte_drift == 0`
- otherwise only compare against the least-bad invalid frontier
- only keep as frontier if it improves the exactness-first score materially
- do not recommend broad capcode changes or broad token-class deletions if they are already in the rejected list
- prefer a narrow next direction tied to the safe shortlist
"""


def main() -> None:
    ensure_dirs()
    run_config = ensure_run_manifest()
    log_line("run:start")
    log_line(f"run:config hash={run_config['config_hash']}")
    ensure_text_sample()
    existing = load_history()
    frontier = load_frontier()
    rebuilt = rebuild_frontier_from_history(existing)
    if rebuilt != frontier:
        frontier = rebuilt
        FRONTIER_JSON.write_text(json.dumps(frontier, indent=2) + "\n", encoding="utf-8")
    start_id = 1 + max((row.get("experiment_id", 0) for row in existing), default=0)
    for experiment_id in range(start_id, start_id + MAX_EXPERIMENTS):
        proposal = propose_experiment(frontier)
        result = run_experiment(experiment_id, proposal)
        result["score"] = score_candidate(result["metrics"], result["audit"])
        decision = run_claude_json(build_review_prompt(frontier, result), timeout=REVIEW_TIMEOUT, phase="review")
        result["decision"] = decision
        result["kept"] = bool(decision.get("keep"))
        append_history(result)
        updated = False
        if result["kept"]:
            if int(result["audit"].get("bad_docs", 10**9)) == 0 and int(result["audit"].get("byte_drift", 10**9)) == 0:
                if float(result["score"]) < float(frontier["best_valid_score"]):
                    frontier["best_valid_label"] = f"experiment_{experiment_id:04d}"
                    frontier["best_valid_score"] = float(result["score"])
                    frontier["best_valid_metrics"] = result["metrics"]
                    frontier["best_valid_audit"] = result["audit"]
                    updated = True
            elif float(result["score"]) < float(frontier["best_invalid_score"]):
                frontier["best_invalid_label"] = f"experiment_{experiment_id:04d}"
                frontier["best_invalid_score"] = float(result["score"])
                frontier["best_invalid_metrics"] = result["metrics"]
                frontier["best_invalid_audit"] = result["audit"]
                updated = True
        if updated:
            frontier["headline"] = decision.get("headline", "")
            frontier["next_direction"] = decision.get("next_direction", "")
            FRONTIER_JSON.write_text(json.dumps(frontier, indent=2) + "\n", encoding="utf-8")
            log_line(f"frontier:update exp=experiment_{experiment_id:04d} score={result['score']:.3f}")
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
