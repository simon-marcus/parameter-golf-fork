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
PROPOSAL_TIMEOUT_SECONDS = int(os.environ.get("PROPOSAL_TIMEOUT_SECONDS", "240"))
GPUS = int(os.environ.get("GPUS", "1"))

CLAUDE_MODEL = os.environ.get("AUTORESEARCH_MODEL", "opus")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "medium")
CLAUDE_ALLOWED_TOOLS = os.environ.get("AUTORESEARCH_ALLOWED_TOOLS", "Bash,Read,Edit")
CLAUDE_PERMISSION_MODE = os.environ.get("AUTORESEARCH_PERMISSION_MODE", "bypassPermissions")
NO_CHANGE_RETRY_ATTEMPTS = int(os.environ.get("AUTORESEARCH_NO_CHANGE_RETRY_ATTEMPTS", "1"))
TOP_K = int(os.environ.get("AUTORESEARCH_TOP_K", "3"))


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


def build_prompt(program: str, history: list[dict], current_best: float | None) -> str:
    is_first_attempt = len(history) == 0
    editable_path = f"./{TRAIN_SCRIPT.name}"
    best_str = "none yet" if current_best is None else f"{current_best:.6f}"
    top_examples_block = format_top_examples_for_prompt(history)

    if is_first_attempt:
        opener = (
            "This is the first attempt.\n"
            f"You should create and write your edit to {editable_path}."
        )
    else:
        opener = (
            f"We've had {len(history)} attempts so far.\n"
            f"Take a look at the previous attempts so you don't repeat one, then add yours in {editable_path}."
        )

    return f"""You are running a simple JEPA inner-loop autoresearch step.

{opener}

Research program:
{program}

Current best val_bpb: {best_str}
Recent attempts:
{format_recent(history)}
Top saved examples (inspectable from cwd):
{top_examples_block}

Task:
1. Read {editable_path}: this is the last iteration of the probe. 
2. If useful, inspect top examples in `./top_examples/`.
3. Develop a plan to improve held-out proxy score (val_bpb). Keep the plan focussed -- try to make a targeted change -- so that we can track which experiments are working and which are not.
4. Edit the {editable_path} file to implement the plan.
5. Verify your edit is present in the file.
6. Print one line: DESCRIPTION: <what changed and why>

Hard rules:
- Keep the script runnable.
- Do not edit files outside {editable_path}.
- Keep output format line `final_int8_zlib_roundtrip_exact ... val_bpb:<score>` intact."""


def build_no_change_retry_prompt(editable_path: str) -> str:
    return (
        f"It looks like you didn't edit the {editable_path} file. "
        "Make the changes directly to the file now.\n"
        f"Required now:\n"
        f"1. Edit {editable_path}.\n"
        "2. Verify the file content changed.\n"
        "3. Print one line: DESCRIPTION: <what changed and why>."
    )


def run_claude(prompt: str) -> tuple[str | None, str, int, float]:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        CLAUDE_MODEL,
        "--effort",
        CLAUDE_EFFORT,
        "--allowedTools",
        CLAUDE_ALLOWED_TOOLS,
        "--permission-mode",
        CLAUDE_PERMISSION_MODE,
        "--output-format",
        "text",
    ]
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
        return None, out, -1, elapsed

    elapsed = time.time() - started
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    match = re.search(r"DESCRIPTION:\s*(.+)", output)
    description = match.group(1).strip() if match else None
    return description, output, result.returncode, elapsed


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
    print(f"  Allowed tools:   {CLAUDE_ALLOWED_TOOLS}")
    print("=" * 70)

    for step in range(MAX_EXPERIMENTS):
        exp_id = next_id + step
        timestamp = datetime.now(timezone.utc).isoformat()
        prev_code = TRAIN_SCRIPT.read_text(encoding="utf-8") if TRAIN_SCRIPT.exists() else ""

        print(f"\n--- EXPERIMENT {exp_id} ({step + 1}/{MAX_EXPERIMENTS}) ---")
        prompt = build_prompt(program, history, best)
        description, proposal_output, proposal_rc, propose_s = run_claude(prompt)
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
            (LOGS_DIR / f"experiment_{exp_id:04d}.proposal.log").write_text(proposal_output, encoding="utf-8")
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
        if new_code == prev_code:
            editable_path = f"./{TRAIN_SCRIPT.name}"
            recovered = False
            if NO_CHANGE_RETRY_ATTEMPTS > 0:
                print(
                    f"  no_changes detected; retrying with direct file-edit reminder "
                    f"({NO_CHANGE_RETRY_ATTEMPTS} attempt(s))"
                )
            for attempt in range(NO_CHANGE_RETRY_ATTEMPTS):
                retry_desc, retry_output, retry_rc, retry_s = run_claude(
                    build_no_change_retry_prompt(editable_path)
                )
                propose_s += retry_s
                proposal_output += (
                    f"\n\n[NO_CHANGE_RETRY attempt {attempt + 1} rc={retry_rc}]\n{retry_output}"
                )
                if retry_desc:
                    description = f"{description} | retry: {retry_desc}"
                if TRAIN_SCRIPT.exists():
                    candidate = TRAIN_SCRIPT.read_text(encoding="utf-8")
                    if candidate != prev_code:
                        new_code = candidate
                        recovered = True
                        print("  recovered: retry produced a file edit")
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
            "val_bpb": val,
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
