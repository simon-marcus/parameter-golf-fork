#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

NAMESPACE="${AUTORESEARCH_NAMESPACE:-jepa_target_embedding_discovery}"
NS_DIR="$ROOT_DIR/autoresearch/$NAMESPACE"
WORK_FILE="$NS_DIR/work/target_embedding_probe.py"
BEST_FILE="$NS_DIR/target_embedding_probe.best.py"
HISTORY_FILE="$NS_DIR/history.jsonl"
OUT_FILE="$NS_DIR/autoresearch.out"
REF_FILE="$ROOT_DIR/records/byte-level-jepa/target-embedding-autoresearch/target_embedding_probe.py"
MODE="${1:-summary}"
INTERVAL="${WATCH_INTERVAL_SECONDS:-5}"

python3 - "$MODE" "$INTERVAL" "$NS_DIR" "$WORK_FILE" "$BEST_FILE" "$HISTORY_FILE" "$OUT_FILE" "$REF_FILE" <<'PY'
import difflib
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime

mode = sys.argv[1]
interval = float(sys.argv[2])
ns_dir = pathlib.Path(sys.argv[3])
work_file = pathlib.Path(sys.argv[4])
best_file = pathlib.Path(sys.argv[5])
history_file = pathlib.Path(sys.argv[6])
out_file = pathlib.Path(sys.argv[7])
ref_file = pathlib.Path(sys.argv[8])


def clear():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def safe_mtime(path: pathlib.Path):
    if not path.exists():
        return "missing"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def file_diff_summary(a: pathlib.Path, b: pathlib.Path):
    if not a.exists() or not b.exists():
        return "n/a"
    a_lines = a.read_text().splitlines()
    b_lines = b.read_text().splitlines()
    diff = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))
    if not diff:
        return "no diff"
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    return f"+{added} / -{removed}"


def load_history():
    if not history_file.exists():
        return []
    rows = []
    for line in history_file.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def get_live_processes():
    try:
        out = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=,command="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    needles = ("jepa_target_embedding_autoresearch_simple.py", "autoresearch.py", "claude -p", "target_embedding_probe.py")
    lines = []
    for line in out.splitlines():
        if any(n in line for n in needles) and "watch_jepa_target_embedding_autoresearch.sh" not in line:
            lines.append(line.rstrip())
    return lines[-8:]


def tail_lines(path: pathlib.Path, count: int = 20):
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-count:]


def print_summary():
    clear()
    history = load_history()
    kept = [x for x in history if x.get("kept")]
    last = history[-1] if history else None
    print("JEPA Target-Embedding Autoresearch Watch")
    print(f"namespace:      {ns_dir}")
    print(f"history rows:   {len(history)}")
    print(f"kept rows:      {len(kept)}")
    print(f"work mtime:     {safe_mtime(work_file)}")
    print(f"best mtime:     {safe_mtime(best_file)}")
    print(f"work vs ref:    {file_diff_summary(ref_file, work_file)}")
    print(f"best vs ref:    {file_diff_summary(ref_file, best_file)}")
    print()
    print("live processes:")
    live = get_live_processes()
    if live:
        for line in live:
            print(f"  {line}")
    else:
        print("  none")
    print()
    print("last history entries:")
    if history:
        for row in history[-5:]:
            rid = row.get("id", "?")
            gate = row.get("gate_reason", "?")
            v = row.get("val_bpb")
            v_str = "FAILED" if v is None else f"{v:.6f}"
            desc = row.get("description", "").replace("\n", " ")
            if len(desc) > 120:
                desc = desc[:117] + "..."
            print(f"  #{rid:<3} {gate:<18} val_bpb={v_str:<10} kept={row.get('kept')}  {desc}")
    else:
        print("  none yet")
    print()
    print("tail autoresearch.out:")
    lines = tail_lines(out_file, 20)
    if lines:
        for line in lines:
            print(f"  {line}")
    else:
        print("  empty")
    print()
    print(f"refreshing every {interval:.0f}s  |  Ctrl-C to stop")


if mode == "tail":
    # Simple tail-follow mode for the raw log.
    last_size = 0
    while True:
        if out_file.exists():
            data = out_file.read_text(errors="replace")
            if len(data) > last_size:
                sys.stdout.write(data[last_size:])
                sys.stdout.flush()
                last_size = len(data)
        time.sleep(interval)
else:
    while True:
        print_summary()
        time.sleep(interval)
PY
