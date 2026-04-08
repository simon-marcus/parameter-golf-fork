#!/usr/bin/env python3
"""
Simple JEPA target-embedding autoresearch loop.

Design: intentionally minimal, modeled after dummy_country_autoresearch.py
1) Build a concise prompt
2) Run `claude -p` once
3) Independently verify target file changed
4) Run probe script
5) Keep if val_bpb improved, else revert
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


TRAIN_SCRIPT = Path(os.environ.get("AUTORESEARCH_TRAIN_SCRIPT", "train_gpt.py"))
PROGRAM_FILE = Path(os.environ.get("AUTORESEARCH_PROGRAM_FILE", "program.md"))
SEED_SCRIPT_ENV = os.environ.get("AUTORESEARCH_SEED_SCRIPT", "").strip()
SEED_SCRIPT = Path(SEED_SCRIPT_ENV) if SEED_SCRIPT_ENV else None

NAMESPACE = os.environ.get("AUTORESEARCH_NAMESPACE", "").strip()
AUTORESEARCH_DIR = Path("autoresearch") / NAMESPACE if NAMESPACE else Path("autoresearch")
HISTORY_FILE = AUTORESEARCH_DIR / "history.jsonl"
LOGS_DIR = AUTORESEARCH_DIR / "logs"
TRAIN_SCRIPT_BEST = AUTORESEARCH_DIR / f"{TRAIN_SCRIPT.stem}.best.py"
KEPT_SNAPSHOTS_DIR = AUTORESEARCH_DIR / "kept_snapshots"
TOP_EXAMPLES_DIR = TRAIN_SCRIPT.parent / "top_examples"
TOP_EXAMPLES_README = TOP_EXAMPLES_DIR / "README.md"

MAX_EXPERIMENTS = int(os.environ.get("MAX_EXPERIMENTS", "20"))
EXPERIMENT_SECONDS = int(os.environ.get("EXPERIMENT_SECONDS", "180"))
TRAIN_TIMEOUT_PADDING_SECONDS = int(os.environ.get("TRAIN_TIMEOUT_PADDING_SECONDS", "300"))
PROPOSAL_TIMEOUT_SECONDS = int(os.environ.get("PROPOSAL_TIMEOUT_SECONDS", "300"))
GPUS = int(os.environ.get("GPUS", "1"))

CLAUDE_MODEL = os.environ.get("AUTORESEARCH_MODEL", "opus")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "medium")
CLAUDE_TOOLS = os.environ.get("AUTORESEARCH_TOOLS", "").strip()
CLAUDE_ALLOWED_TOOLS = os.environ.get("AUTORESEARCH_ALLOWED_TOOLS", "Bash,Read,Edit")
CLAUDE_PERMISSION_MODE = os.environ.get("AUTORESEARCH_PERMISSION_MODE", "bypassPermissions")
CLAUDE_SETTINGS = os.environ.get(
    "AUTORESEARCH_CLAUDE_SETTINGS",
    '{"alwaysThinkingEnabled":false}',
)
NO_CHANGE_RETRY_ATTEMPTS = int(os.environ.get("AUTORESEARCH_NO_CHANGE_RETRY_ATTEMPTS", "1"))
TOP_K = int(os.environ.get("AUTORESEARCH_TOP_K", "3"))
NEAR_MISS_DELTA = float(os.environ.get("AUTORESEARCH_NEAR_MISS_DELTA", "0.0015"))


def ensure_dirs() -> None:
    AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    KEPT_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    TOP_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    out: list[dict] = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def append_history(entry: dict) -> None:
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def best_bpb(history: list[dict]) -> float | None:
    vals = [
        float(h["val_bpb"])
        for h in history
        if h.get("kept") is True and h.get("val_bpb") is not None
    ]
    return min(vals) if vals else None


def top_kept_entries(history: list[dict], k: int) -> list[dict]:
    kept = [
        h
        for h in history
        if h.get("kept") is True and h.get("val_bpb") is not None
    ]
    kept.sort(key=lambda h: float(h["val_bpb"]))
    return kept[: max(0, k)]


def latest_baseline_entry(history: list[dict]) -> dict | None:
    baselines = [h for h in history if h.get("baseline") and h.get("val_bpb") is not None]
    return baselines[-1] if baselines else None


def near_miss_entries(history: list[dict], current_best: float | None, k: int = 4) -> list[dict]:
    if current_best is None:
        return []
    misses = [
        h
        for h in history
        if h.get("kept") is False
        and h.get("val_bpb") is not None
        and not h.get("baseline")
        and h.get("gate_reason", "").startswith("no_val_bpb_improvement")
        and float(h["val_bpb"]) <= current_best + NEAR_MISS_DELTA
    ]
    misses.sort(key=lambda h: float(h["val_bpb"]))
    return misses[: max(0, k)]


def sync_top_examples(history: list[dict]) -> None:
    TOP_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for old in TOP_EXAMPLES_DIR.glob("rank_*.py"):
        old.unlink(missing_ok=True)

    lines = ["# Top JEPA Examples", ""]
    top = top_kept_entries(history, TOP_K)
    if not top:
        lines.append("No kept examples yet.")
        TOP_EXAMPLES_README.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for rank, entry in enumerate(top, start=1):
        eid = int(entry["id"])
        val = float(entry["val_bpb"])
        desc = str(entry.get("description", "")).replace("\n", " ").strip()
        if len(desc) > 180:
            desc = desc[:177] + "..."

        snapshot = KEPT_SNAPSHOTS_DIR / f"experiment_{eid:04d}.py"
        source: Path | None = None
        if snapshot.exists():
            source = snapshot
        elif rank == 1 and TRAIN_SCRIPT_BEST.exists():
            source = TRAIN_SCRIPT_BEST

        if source is not None:
            out_name = f"rank_{rank:02d}_exp_{eid:04d}_val_{val:.6f}.py"
            out_path = TOP_EXAMPLES_DIR / out_name
            shutil.copy2(source, out_path)
            file_ref = f"./top_examples/{out_name}"
        else:
            file_ref = "(snapshot_missing)"

        lines.append(f"- rank {rank}: exp #{eid}, val_bpb={val:.6f}, file={file_ref}")
        lines.append(f"  desc: {desc}")

    TOP_EXAMPLES_README.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_top_examples_for_prompt(history: list[dict]) -> str:
    top = top_kept_entries(history, TOP_K)
    if not top:
        return "none yet"
    lines: list[str] = []
    for rank, entry in enumerate(top, start=1):
        eid = int(entry["id"])
        val = float(entry["val_bpb"])
        desc = str(entry.get("description", "")).replace("\n", " ").strip()
        if len(desc) > 140:
            desc = desc[:137] + "..."
        files = sorted(TOP_EXAMPLES_DIR.glob(f"rank_{rank:02d}_exp_{eid:04d}_val_*.py"))
        file_ref = f"./top_examples/{files[0].name}" if files else "(snapshot_pending)"
        lines.append(f"- rank {rank}: exp #{eid}, val_bpb={val:.6f}, file={file_ref} | {desc}")
    return "\n".join(lines)


def format_near_misses_for_prompt(history: list[dict], current_best: float | None) -> str:
    misses = near_miss_entries(history, current_best)
    if not misses:
        return "none yet"
    lines: list[str] = []
    for entry in misses:
        eid = int(entry["id"])
        val = float(entry["val_bpb"])
        desc = str(entry.get("description", "")).replace("\n", " ").strip()
        if len(desc) > 160:
            desc = desc[:157] + "..."
        lines.append(f"- exp #{eid}: val_bpb={val:.6f} | {desc}")
    return "\n".join(lines)


def format_recent(history: list[dict], n: int = 8) -> str:
    if not history:
        return "none yet"
    rows = []
    for h in history[-n:]:
        rid = h.get("id", "?")
        gate = h.get("gate_reason", "?")
        vb = h.get("val_bpb")
        vb_str = "FAILED" if vb is None else f"{float(vb):.6f}"
        kept = h.get("kept")
        desc = str(h.get("description", "")).replace("\n", " ").strip()
        if len(desc) > 180:
            desc = desc[:177] + "..."
        rows.append(f"#{rid} [{gate}] val_bpb={vb_str} kept={kept} | {desc}")
    return "\n".join(rows)


def sanitize_program_for_prompt(program: str, editable_path: str) -> str:
    program = re.sub(
        r"\[target_embedding_probe\.py\]\([^)]+\)",
        f"`{editable_path}`",
        program,
    )
    program = program.replace(
        "records/byte-level-jepa/target-embedding-autoresearch/target_embedding_probe.py",
        editable_path,
    )
    return program


def failure_family_label(description: str) -> str | None:
    d = description.lower()
    if "symmetric" in d and "infonce" in d:
        return "symmetric InfoNCE"
    if "cross-attention" in d:
        return "predictor/context cross-attention"
    if "contextaggregator" in d and "ffn" in d:
        return "ContextAggregator FFN expansion"
    if "deepen contextaggregator" in d or ("contextaggregator" in d and ("2-layer" in d or "3-layer" in d)):
        return "ContextAggregator depth increases"
    if "targetpatchencoder" in d and ("self-attention" in d or "relative position bias" in d):
        return "TargetPatchEncoder attention tweaks"
    if "dropout" in d:
        return "dropout regularization"
    if "projection head" in d:
        return "contrastive projection head"
    if "label smoothing" in d:
        return "InfoNCE label smoothing"
    if "weight_decay=0" in d or "weight decay=0" in d:
        return "zero weight decay on contrastive/predictor groups"
    return None


def repeated_failure_families(history: list[dict], min_count: int = 2) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for entry in history:
        if entry.get("kept") is True:
            continue
        if entry.get("val_bpb") is None:
            continue
        label = failure_family_label(str(entry.get("description", "")))
        if label is None:
            continue
        counts[label] = counts.get(label, 0) + 1
    rows = [(label, count) for label, count in counts.items() if count >= min_count]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows


def format_repeated_failures_for_prompt(history: list[dict]) -> str:
    rows = repeated_failure_families(history)
    if not rows:
        return "none yet"
    return "\n".join(f"- {label} failed {count} times" for label, count in rows)


def parse_probe_diagnostics(output: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "val_loss": None,
        "val_bpb": None,
        "val_cosine": None,
        "retrieval_at_1": None,
        "pred_std": None,
        "target_std": None,
        "shuffle_gap": None,
        "collapse_penalty": None,
    }

    diag = re.search(
        r"DIAGNOSTIC probe_summary "
        r"val_cosine:([0-9]*\.?[0-9]+) "
        r"retrieval_at_1:([0-9]*\.?[0-9]+) "
        r"pred_std:([0-9]*\.?[0-9]+) "
        r"target_std:([0-9]*\.?[0-9]+) "
        r"shuffle_gap:([0-9]*\.?[0-9]+) "
        r"collapse_penalty:([0-9]*\.?[0-9]+)",
        output,
    )
    if diag:
        metrics["val_cosine"] = float(diag.group(1))
        metrics["retrieval_at_1"] = float(diag.group(2))
        metrics["pred_std"] = float(diag.group(3))
        metrics["target_std"] = float(diag.group(4))
        metrics["shuffle_gap"] = float(diag.group(5))
        metrics["collapse_penalty"] = float(diag.group(6))

    final_exact = re.search(
        r"final_int8_zlib_roundtrip_exact val_loss:([0-9]*\.?[0-9]+) "
        r"val_bpb:([0-9]*\.?[0-9]+)",
        output,
    )
    if final_exact:
        metrics["val_loss"] = float(final_exact.group(1))
        metrics["val_bpb"] = float(final_exact.group(2))

    return metrics


def diagnostics_for_entry(entry: dict) -> dict[str, float | None]:
    keys = (
        "val_loss",
        "val_bpb",
        "val_cosine",
        "retrieval_at_1",
        "pred_std",
        "target_std",
        "shuffle_gap",
        "collapse_penalty",
    )
    cached_diag_keys = (
        "val_loss",
        "val_cosine",
        "retrieval_at_1",
        "pred_std",
        "target_std",
        "shuffle_gap",
        "collapse_penalty",
    )
    if any(entry.get(key) is not None for key in cached_diag_keys):
        return {key: entry.get(key) for key in keys}

    exp_id = entry.get("id")
    if exp_id is None:
        return {key: None for key in keys}
    log_file = LOGS_DIR / f"experiment_{int(exp_id):04d}.train.log"
    if not log_file.exists():
        return {key: None for key in keys}
    return parse_probe_diagnostics(log_file.read_text(encoding="utf-8"))


def current_best_summary(history: list[dict], current_best: float | None) -> str:
    if current_best is None:
        return "none yet"
    kept = top_kept_entries(history, 1)
    if not kept:
        return f"val_bpb={current_best:.6f}"
    best_entry = kept[0]
    diag = diagnostics_for_entry(best_entry)
    parts = [f"exp #{int(best_entry['id'])}", f"val_bpb={current_best:.6f}"]
    if diag.get("val_loss") is not None:
        parts.append(f"val_loss={diag['val_loss']:.6f}")
    if diag.get("retrieval_at_1") is not None:
        parts.append(f"retrieval_at_1={diag['retrieval_at_1']:.6f}")
    if diag.get("val_cosine") is not None:
        parts.append(f"val_cosine={diag['val_cosine']:.6f}")
    if diag.get("pred_std") is not None:
        parts.append(f"pred_std={diag['pred_std']:.6f}")
    if diag.get("target_std") is not None:
        parts.append(f"target_std={diag['target_std']:.6f}")
    if diag.get("shuffle_gap") is not None:
        parts.append(f"shuffle_gap={diag['shuffle_gap']:.6f}")
    if diag.get("collapse_penalty") is not None:
        parts.append(f"collapse_penalty={diag['collapse_penalty']:.6f}")
    return ", ".join(parts)


def current_signal_hint(history: list[dict], current_best: float | None) -> str:
    if current_best is None:
        return "No diagnostic hint yet."
    kept = top_kept_entries(history, 1)
    if not kept:
        return "No diagnostic hint yet."
    diag = diagnostics_for_entry(kept[0])
    retrieval = diag.get("retrieval_at_1")
    collapse = diag.get("collapse_penalty")
    if retrieval is not None and collapse is not None and retrieval < 0.12 and collapse == 0.0:
        return (
            "Current best looks retrieval-limited rather than collapse-limited: "
            "retrieval_at_1 is still low while collapse_penalty is zero."
        )
    return "Use the diagnostics below to pick one primary target signal."


def build_prompt(program: str, history: list[dict], current_best: float | None) -> str:
    is_first_attempt = len(history) == 0
    editable_path = f"./{TRAIN_SCRIPT.name}"
    program = sanitize_program_for_prompt(program, editable_path)
    best_str = "none yet" if current_best is None else f"{current_best:.6f}"
    top_examples_block = format_top_examples_for_prompt(history)
    near_misses_block = format_near_misses_for_prompt(history, current_best)
    repeated_failures_block = format_repeated_failures_for_prompt(history)
    baseline = latest_baseline_entry(history)
    baseline_str = "none recorded"
    if baseline is not None:
        baseline_str = (
            f"exp #{int(baseline['id'])}, val_bpb={float(baseline['val_bpb']):.6f}"
        )

    if is_first_attempt:
        opener = (
            "This is the first attempt.\n"
            f"You should create and write your edit to {editable_path}."
        )
    else:
        opener = (
            f"We've had {len(history)} attempts so far.\n"
            f"{editable_path} is the current best working variant in this namespace.\n"
            "Inspect previous attempts so you avoid repeating weak families, then add one targeted improvement."
        )

    return f"""You are running a simple JEPA inner-loop autoresearch step.

{opener}

Research program:
{program}

Important path rule:
- Ignore any path in the research program that points into `records/...`.
- The only file you may edit is {editable_path} in the current working directory.
- Do not use a full-file Read on {editable_path}; it exceeds the Read tool size cap. Inspect with targeted `rg`/`sed` or partial reads only.

Baseline source-current:
{baseline_str}
Current best val_bpb: {best_str}
Current best diagnostic summary:
{current_best_summary(history, current_best)}
Signal hint:
{current_signal_hint(history, current_best)}
Best kept examples:
{top_examples_block}
Near misses worth exploiting:
{near_misses_block}
Repeated failed families to avoid:
{repeated_failures_block}
Recent attempts:
{format_recent(history)}

Task:
1. Read {editable_path}: this is the current best working probe variant for this namespace.
2. If useful, inspect top examples in `./top_examples/`.
3. Pick one primary target signal for this experiment: `retrieval`, `mse`, `collapse`, or `stability`.
4. Develop one materially novel, focused change aimed at that signal. Minor rewordings or tiny parameter nudges of repeated failed families do not count as novel.
5. Edit the {editable_path} file to implement the plan.
6. Verify your edit is present in the file by reading the changed lines back.
7. Print these lines:
DESCRIPTION: <what changed and why>
NOVELTY: <why this is materially different from kept examples and repeated failures>
TARGET_SIGNAL: <retrieval|mse|collapse|stability>

Hard rules:
- Keep the script runnable.
- Do not edit files outside {editable_path}.
- Keep output format line `final_int8_zlib_roundtrip_exact ... val_bpb:<score>` intact.
- Do not merely describe a change. Apply it directly to {editable_path}.
- If a near miss suggests a local follow-up, make the follow-up concrete and still materially distinct from the original attempt."""


def build_no_change_retry_prompt(original_prompt: str, editable_path: str, previous_description: str) -> str:
    return f"""Your previous response described a change, but no edit to {editable_path} was detected.

Previous DESCRIPTION:
{previous_description}

You must now apply the edit directly to {editable_path}.
If the earlier change was too broad, implement a smaller concrete variant of the same idea rather than switching to a repeated failure family.

Required now:
1. Edit {editable_path}.
2. Verify the file content changed by reading the changed lines back.
3. Print these lines:
DESCRIPTION: <what changed and why>
NOVELTY: <why this is still materially distinct>
TARGET_SIGNAL: <retrieval|mse|collapse|stability>

Original research context:

{original_prompt}"""


def run_claude(prompt: str) -> tuple[str | None, str | None, str | None, str, int, float]:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        CLAUDE_MODEL,
        "--effort",
        CLAUDE_EFFORT,
        "--settings",
        CLAUDE_SETTINGS,
        "--permission-mode",
        CLAUDE_PERMISSION_MODE,
        "--output-format",
        "json",
    ]
    if CLAUDE_TOOLS:
        cmd.extend(["--tools", CLAUDE_TOOLS])
    if CLAUDE_ALLOWED_TOOLS:
        cmd.append(f"--allowedTools={CLAUDE_ALLOWED_TOOLS}")
    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(TRAIN_SCRIPT.parent),
            capture_output=True,
            text=True,
            timeout=PROPOSAL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        out = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return None, None, None, out, -1, elapsed

    elapsed = time.time() - started
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    output = ((stdout + "\n") if stdout else "") + stderr
    text = stdout
    if result.returncode == 0 and stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            if payload.get("is_error"):
                errors = payload.get("errors") or []
                if errors:
                    output += "\n" + "\n".join(str(err) for err in errors)
                text = ""
            else:
                text = str(payload.get("result") or "")
    match = re.search(r"DESCRIPTION:\s*(.+)", text)
    description = match.group(1).strip() if match else None
    novelty_match = re.search(r"NOVELTY:\s*(.+)", text)
    novelty = novelty_match.group(1).strip() if novelty_match else None
    signal_match = re.search(r"TARGET_SIGNAL:\s*(.+)", text)
    target_signal = signal_match.group(1).strip() if signal_match else None
    return description, novelty, target_signal, output, result.returncode, elapsed


def recover_misdirected_seed_edit(seed_prev_code: str | None) -> str | None:
    if seed_prev_code is None or SEED_SCRIPT is None or not SEED_SCRIPT.exists():
        return None
    seed_new_code = SEED_SCRIPT.read_text(encoding="utf-8")
    if seed_new_code == seed_prev_code:
        return None
    # Claude sometimes follows the source-path mentioned in the research program
    # and edits the records seed file instead of the namespace-local work file.
    TRAIN_SCRIPT.write_text(seed_new_code, encoding="utf-8")
    SEED_SCRIPT.write_text(seed_prev_code, encoding="utf-8")
    return seed_new_code


def run_training(experiment_id: int) -> tuple[str, int, float]:
    env = os.environ.copy()
    env["RUN_ID"] = f"autoresearch_{experiment_id}"
    timeout = EXPERIMENT_SECONDS + TRAIN_TIMEOUT_PADDING_SECONDS

    if TRAIN_SCRIPT.name == "target_embedding_probe.py":
        cmd = [sys.executable, str(TRAIN_SCRIPT)]
    else:
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={GPUS}", str(TRAIN_SCRIPT)]

    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        elapsed = time.time() - started
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return output, result.returncode, elapsed
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        output = (exc.stdout or "") + "\n" + (exc.stderr or "") + "\n[TIMEOUT]"
        return output, -1, elapsed


def parse_val_bpb(output: str) -> float | None:
    patterns = [
        r"final_int8_zlib_roundtrip_exact.*?val_bpb:([0-9]*\.?[0-9]+)",
        r"val_bpb:([0-9]*\.?[0-9]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, output, flags=re.DOTALL)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    return None


def main() -> int:
    ensure_dirs()

    if not PROGRAM_FILE.exists():
        print(f"Error: program file missing: {PROGRAM_FILE}", file=sys.stderr)
        return 1

    if not TRAIN_SCRIPT.exists():
        if SEED_SCRIPT is not None and SEED_SCRIPT.exists():
            shutil.copy2(SEED_SCRIPT, TRAIN_SCRIPT)
            print(f"Bootstrapped {TRAIN_SCRIPT} from {SEED_SCRIPT}")
        else:
            print(f"Warning: {TRAIN_SCRIPT} missing; first proposal must create it.")

    if not TRAIN_SCRIPT_BEST.exists() and TRAIN_SCRIPT.exists():
        shutil.copy2(TRAIN_SCRIPT, TRAIN_SCRIPT_BEST)

    history = load_history()
    sync_top_examples(history)
    best = best_bpb(history)
    program = PROGRAM_FILE.read_text(encoding="utf-8")
    next_id = max((int(h.get("id", 0)) for h in history), default=0) + 1

    print("=" * 70)
    print("JEPA TARGET-EMBEDDING AUTORESEARCH (SIMPLE LOOP)")
    print("=" * 70)
    print(f"  Namespace:       {AUTORESEARCH_DIR}")
    print(f"  Train script:    {TRAIN_SCRIPT}")
    print(f"  Current best:    {best if best is not None else 'none'}")
    print(f"  Attempts so far: {len(history)}")
    print(f"  Top-K saved:     {TOP_K} (in {TOP_EXAMPLES_DIR})")
    print(f"  Claude model:    {CLAUDE_MODEL}")
    print(f"  Claude effort:   {CLAUDE_EFFORT}")
    print(f"  Claude tools:    {CLAUDE_TOOLS or '(default cli behavior)'}")
    print(f"  Allowed tools:   {CLAUDE_ALLOWED_TOOLS}")
    print(f"  Claude settings: {CLAUDE_SETTINGS}")
    print("=" * 70)

    for step in range(MAX_EXPERIMENTS):
        exp_id = next_id + step
        timestamp = datetime.now(timezone.utc).isoformat()
        prev_code = TRAIN_SCRIPT.read_text(encoding="utf-8") if TRAIN_SCRIPT.exists() else ""
        seed_prev_code = (
            SEED_SCRIPT.read_text(encoding="utf-8")
            if SEED_SCRIPT is not None and SEED_SCRIPT.exists()
            else None
        )

        print(f"\n--- EXPERIMENT {exp_id} ({step + 1}/{MAX_EXPERIMENTS}) ---")
        prompt = build_prompt(program, history, best)
        description, novelty, target_signal, proposal_output, proposal_rc, propose_s = run_claude(prompt)
        (LOGS_DIR / f"experiment_{exp_id:04d}.proposal.log").write_text(proposal_output, encoding="utf-8")
        if description is None:
            entry = {
                "id": exp_id,
                "description": "Failed to propose",
                "val_bpb": None,
                "kept": False,
                "gate_reason": "invalid_proposal",
                "error": "invalid_proposal",
                "proposal_returncode": proposal_rc,
                "propose_seconds": round(propose_s, 2),
                "timestamp": timestamp,
            }
            append_history(entry)
            history.append(entry)
            print(f"  invalid proposal (rc={proposal_rc})")
            continue

        if not TRAIN_SCRIPT.exists():
            entry = {
                "id": exp_id,
                "description": f"{description} (MISSING_TARGET_SCRIPT)",
                "val_bpb": None,
                "kept": False,
                "gate_reason": "missing_target_script",
                "error": "missing_target_script",
                "propose_seconds": round(propose_s, 2),
                "timestamp": timestamp,
            }
            append_history(entry)
            history.append(entry)
            print("  missing target script after proposal")
            continue

        new_code = TRAIN_SCRIPT.read_text(encoding="utf-8")
        recovered_seed_code = recover_misdirected_seed_edit(seed_prev_code)
        if new_code == prev_code and recovered_seed_code is not None:
            new_code = recovered_seed_code
            description = (
                f"{description} | recovered: proposal edited seed/source file; "
                "applied the change to the namespace work file instead"
            )
            print("  recovered: proposal edited seed/source file instead of work file")
        if new_code == prev_code:
            editable_path = f"./{TRAIN_SCRIPT.name}"
            recovered = False
            if NO_CHANGE_RETRY_ATTEMPTS > 0:
                print(
                    f"  no_changes detected; retrying with direct file-edit reminder "
                    f"({NO_CHANGE_RETRY_ATTEMPTS} attempt(s))"
                )
            for attempt in range(NO_CHANGE_RETRY_ATTEMPTS):
                retry_seed_prev_code = (
                    SEED_SCRIPT.read_text(encoding="utf-8")
                    if SEED_SCRIPT is not None and SEED_SCRIPT.exists()
                    else None
                )
                retry_desc, retry_novelty, retry_target_signal, retry_output, retry_rc, retry_s = run_claude(
                    build_no_change_retry_prompt(prompt, editable_path, description)
                )
                propose_s += retry_s
                proposal_output += (
                    f"\n\n[NO_CHANGE_RETRY attempt {attempt + 1} rc={retry_rc}]\n{retry_output}"
                )
                (LOGS_DIR / f"experiment_{exp_id:04d}.proposal.log").write_text(proposal_output, encoding="utf-8")
                if retry_desc:
                    description = f"{description} | retry: {retry_desc}"
                if retry_novelty:
                    novelty = retry_novelty
                if retry_target_signal:
                    target_signal = retry_target_signal
                if TRAIN_SCRIPT.exists():
                    candidate = TRAIN_SCRIPT.read_text(encoding="utf-8")
                    if candidate != prev_code:
                        new_code = candidate
                        recovered = True
                        print("  recovered: retry produced a file edit")
                        break
                recovered_seed_code = recover_misdirected_seed_edit(retry_seed_prev_code)
                if recovered_seed_code is not None and recovered_seed_code != prev_code:
                    new_code = recovered_seed_code
                    recovered = True
                    description = (
                        f"{description} | recovered: retry edited seed/source file; "
                        "applied the change to the namespace work file instead"
                    )
                    print("  recovered: retry edited seed/source file instead of work file")
                    break
                print(f"  retry {attempt + 1}/{NO_CHANGE_RETRY_ATTEMPTS}: still no file changes")

            if recovered:
                pass
            else:
                entry = {
                    "id": exp_id,
                    "description": f"{description} (NO CHANGES)",
                    "val_bpb": None,
                    "kept": False,
                    "gate_reason": "no_changes",
                    "error": "no_changes",
                    "propose_seconds": round(propose_s, 2),
                    "timestamp": timestamp,
                }
                append_history(entry)
                history.append(entry)
                print("  no_changes")
                continue

        print(f"  proposal: {description}")
        output, rc, train_s = run_training(exp_id)
        (LOGS_DIR / f"experiment_{exp_id:04d}.train.log").write_text(output, encoding="utf-8")

        val = parse_val_bpb(output)
        diagnostics = parse_probe_diagnostics(output)
        keep = rc == 0 and val is not None and (best is None or val < best)

        if keep:
            old_best = best
            best = val
            KEPT_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            (KEPT_SNAPSHOTS_DIR / f"experiment_{exp_id:04d}.py").write_text(
                new_code, encoding="utf-8"
            )
            shutil.copy2(TRAIN_SCRIPT, TRAIN_SCRIPT_BEST)
            gate_reason = f"improved_val_bpb ({old_best} -> {val:.6f})" if old_best is not None else f"improved_val_bpb (none -> {val:.6f})"
            print(f"  kept: val_bpb={val:.6f}")
        else:
            TRAIN_SCRIPT.write_text(prev_code, encoding="utf-8")
            if rc == -1:
                gate_reason = "timeout"
            elif rc != 0:
                gate_reason = f"exit_{rc}"
            elif val is None:
                gate_reason = "missing_val_bpb"
            else:
                gate_reason = f"no_val_bpb_improvement (best={best}, got={val:.6f})"
            print(f"  reverted: {gate_reason}")

        entry = {
            "id": exp_id,
            "description": description,
            "novelty": novelty,
            "target_signal": target_signal,
            "val_bpb": val,
            "val_loss": diagnostics.get("val_loss"),
            "val_cosine": diagnostics.get("val_cosine"),
            "retrieval_at_1": diagnostics.get("retrieval_at_1"),
            "pred_std": diagnostics.get("pred_std"),
            "target_std": diagnostics.get("target_std"),
            "shuffle_gap": diagnostics.get("shuffle_gap"),
            "collapse_penalty": diagnostics.get("collapse_penalty"),
            "kept": bool(keep),
            "gate_reason": gate_reason,
            "error": None if keep else gate_reason,
            "proposal_returncode": proposal_rc,
            "train_returncode": rc,
            "propose_seconds": round(propose_s, 2),
            "train_seconds": round(train_s, 2),
            "timestamp": timestamp,
        }
        append_history(entry)
        history.append(entry)
        if keep:
            sync_top_examples(history)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
