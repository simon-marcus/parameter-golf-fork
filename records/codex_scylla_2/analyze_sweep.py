#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

SWEEP_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("records/codex_scylla_2/runs")

PATTERNS = {
    "prequant_bpb": re.compile(r"final_prequant_sliding_exact\s+val_loss:[\d.]+ val_bpb:([\d.]+)"),
    "roundtrip_bpb": re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:[\d.]+ val_bpb:([\d.]+)"),
    "legal_ttt_bpb": re.compile(r"legal_ttt_exact\s+val_loss:[\d.]+ val_bpb:([\d.]+)"),
    "ttt_elapsed": re.compile(r"ttt_sliding:done.*?elapsed=([\d.]+)s"),
    "unfrozen_params": re.compile(r"ttt_sliding:params\s+unfrozen=(\d+)"),
    "frozen_params": re.compile(r"ttt_sliding:params\s+.*?frozen=(\d+)"),
    "num_chunks": re.compile(r"ttt_sliding:start\s+chunks=(\d+)"),
}
TTT_CHUNK_RE = re.compile(r"ttt_chunk\s+\[(\d+)/(\d+)\]\s+bpb=([\d.]+)")


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text()
    result = {}
    for key, pat in PATTERNS.items():
        matches = pat.findall(text)
        if matches:
            result[key] = matches[-1]
    chunks = TTT_CHUNK_RE.findall(text)
    if chunks:
        result["ttt_first_bpb"] = chunks[0][2]
        result["ttt_last_bpb"] = chunks[-1][2]
        result["ttt_chunks_logged"] = len(chunks)
    return result


def fget(d, key):
    return float(d[key]) if key in d else None


def ffmt(x, width=10):
    return f"{x:>{width}.6f}" if x is not None else f"{'---':>{width}}"


def main():
    if not SWEEP_ROOT.exists():
        print(f"Sweep root not found: {SWEEP_ROOT}")
        sys.exit(1)

    rows = []
    for run_dir in sorted(SWEEP_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        log_path = run_dir / "train.log"
        if not log_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        metrics = parse_log(log_path)
        rows.append((run_dir.name, cfg, metrics))

    if not rows:
        print("No results found.")
        sys.exit(1)

    print("\n" + "=" * 146)
    print("CODEX SCYLLA 2 SWEEP")
    print("=" * 146)
    header = f"{'Run':>3} | {'Chunk':>6} | {'LR':>7} | {'Ep':>2} | {'FrzBlk':>6} | {'FrzEmb':>6} | {'Batch':>5} | {'Roundtrip':>10} | {'Legal TTT':>10} | {'TTT Gain':>9} | {'TTT Elap':>8} | {'Chunks':>6}"
    print(header)
    print("-" * len(header))

    ranked = []
    for name, cfg, m in rows:
        rt = fget(m, "roundtrip_bpb")
        ttt = fget(m, "legal_ttt_bpb")
        gain = (rt - ttt) if rt is not None and ttt is not None else None
        ranked.append((name, cfg, m, ttt if ttt is not None else 999.0))

        print(
            f"{name:>3} | "
            f"{cfg.get('ttt_chunk_tokens', '?'):>6} | "
            f"{cfg.get('ttt_lr', '?'):>7} | "
            f"{cfg.get('ttt_epochs', '?'):>2} | "
            f"{cfg.get('ttt_freeze_blocks', '?'):>6} | "
            f"{cfg.get('ttt_freeze_embeddings', '?'):>6} | "
            f"{cfg.get('ttt_batch_seqs', '?'):>5} | "
            f"{ffmt(rt, 10)} | "
            f"{ffmt(ttt, 10)} | "
            f"{ffmt(gain, 9)} | "
            f"{m.get('ttt_elapsed', '?'):>8} | "
            f"{m.get('num_chunks', '?'):>6}"
        )

    ranked.sort(key=lambda x: x[3])
    best = ranked[0]
    print("\nBEST:")
    print(f"  {best[0]}  legal_ttt_bpb={best[2].get('legal_ttt_bpb', 'n/a')}")
    print("  config=" + json.dumps(best[1], sort_keys=True))

    print("\nTTT trajectory:")
    for name, _, m, _ in ranked:
        if "ttt_first_bpb" in m:
            print(f"  {name}: {m['ttt_first_bpb']} -> {m['ttt_last_bpb']} ({m['ttt_chunks_logged']} logged)")


if __name__ == "__main__":
    main()
