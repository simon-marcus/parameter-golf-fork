#!/usr/bin/env python3
"""
Analyze Scylla TTT sweep results.

Usage:
  python3 records/scylla_2_claude/analyze_sweep.py [SWEEP_ROOT]

Default SWEEP_ROOT: records/scylla_2_claude/runs/
"""
import json
import re
import sys
from pathlib import Path

SWEEP_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("records/scylla_2_claude/runs")

# Patterns to extract from train.log
PATTERNS = {
    "roundtrip_bpb": re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:[\d.]+ val_bpb:([\d.]+)"),
    "legal_ttt_bpb": re.compile(r"legal_ttt_exact\s+val_loss:[\d.]+ val_bpb:([\d.]+)"),
    "prequant_bpb": re.compile(r"final_prequant_sliding_exact\s+val_loss:[\d.]+ val_bpb:([\d.]+)"),
    "ttt_elapsed": re.compile(r"ttt_sliding:done.*?elapsed=([\d.]+)s"),
    "unfrozen_params": re.compile(r"ttt_sliding:params\s+unfrozen=(\d+)"),
    "frozen_params": re.compile(r"ttt_sliding:params\s+.*?frozen=(\d+)"),
    "num_chunks": re.compile(r"ttt_sliding:start\s+chunks=(\d+)"),
    "artifact_bytes": re.compile(r"artifact_bytes:(\d+)"),
}

# Also extract TTT chunk BPB trajectory (first and last)
TTT_CHUNK_RE = re.compile(r"ttt_chunk\s+\[(\d+)/(\d+)\]\s+bpb=([\d.]+)")


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text()
    result = {}
    for key, pat in PATTERNS.items():
        matches = pat.findall(text)
        if matches:
            result[key] = matches[-1]  # Take last match (final value)

    # TTT trajectory
    chunks = TTT_CHUNK_RE.findall(text)
    if chunks:
        result["ttt_first_bpb"] = chunks[0][2]
        result["ttt_last_bpb"] = chunks[-1][2]
        result["ttt_chunks_logged"] = len(chunks)

    return result


def main():
    if not SWEEP_ROOT.exists():
        print(f"Sweep root not found: {SWEEP_ROOT}")
        sys.exit(1)

    runs = sorted(SWEEP_ROOT.iterdir())
    if not runs:
        print(f"No run directories in {SWEEP_ROOT}")
        sys.exit(1)

    results = []
    for run_dir in runs:
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        log_path = run_dir / "train.log"

        if not log_path.exists():
            print(f"  SKIP {run_dir.name}: no train.log")
            continue

        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        metrics = parse_log(log_path)
        results.append({"name": run_dir.name, "config": config, "metrics": metrics})

    if not results:
        print("No results found.")
        sys.exit(1)

    # --- Print comparison table ---
    print("\n" + "=" * 120)
    print("SCYLLA-v2 TTT SWEEP RESULTS")
    print("=" * 120)

    header = f"{'Run':>4} | {'Chunk':>6} | {'LR':>7} | {'Ep':>2} | {'FrzBlk':>6} | {'FrzEmb':>6} | {'Roundtrip':>10} | {'Legal TTT':>10} | {'TTT Gain':>9} | {'Elapsed':>8} | {'Unfrozen':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        c = r["config"]
        m = r["metrics"]
        name = r["name"]
        chunk = c.get("ttt_chunk_tokens", "?")
        lr = c.get("ttt_lr", "?")
        epochs = c.get("ttt_epochs", "?")
        fblk = c.get("ttt_freeze_blocks", "?")
        femb = c.get("ttt_freeze_embeddings", "?")

        rt_bpb = float(m["roundtrip_bpb"]) if "roundtrip_bpb" in m else None
        ttt_bpb = float(m["legal_ttt_bpb"]) if "legal_ttt_bpb" in m else None
        gain = (rt_bpb - ttt_bpb) if rt_bpb and ttt_bpb else None
        elapsed = m.get("ttt_elapsed", "?")
        unfrozen = m.get("unfrozen_params", "?")

        rt_str = f"{rt_bpb:.6f}" if rt_bpb else "---"
        ttt_str = f"{ttt_bpb:.6f}" if ttt_bpb else "---"
        gain_str = f"{gain:+.6f}" if gain else "---"

        print(f"{name:>4} | {chunk:>6} | {lr:>7} | {epochs:>2} | {fblk:>6} | {femb:>6} | {rt_str:>10} | {ttt_str:>10} | {gain_str:>9} | {elapsed:>8} | {unfrozen:>10}")

    # --- Best run ---
    ttt_runs = [r for r in results if "legal_ttt_bpb" in r["metrics"]]
    if ttt_runs:
        best = min(ttt_runs, key=lambda r: float(r["metrics"]["legal_ttt_bpb"]))
        print(f"\nBEST: {best['name']} — legal_ttt_bpb = {best['metrics']['legal_ttt_bpb']}")

    # --- TTT trajectory comparison ---
    print("\n" + "-" * 80)
    print("TTT TRAJECTORY (first → last chunk BPB)")
    print("-" * 80)
    for r in results:
        m = r["metrics"]
        if "ttt_first_bpb" in m:
            print(f"  {r['name']:>4}: {m['ttt_first_bpb']} → {m['ttt_last_bpb']}  ({m.get('ttt_chunks_logged', '?')} logged chunks)")

    print()


if __name__ == "__main__":
    main()
