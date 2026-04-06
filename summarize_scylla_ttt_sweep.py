#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PATTERNS = {
    "roundtrip": re.compile(r"final_int8_zlib_roundtrip_exact val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)"),
    "legal_ttt": re.compile(r"legal_ttt_exact val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)"),
    "ttt_done": re.compile(r"ttt_sliding:done val_loss=(?P<loss>[0-9.]+) val_bpb=(?P<bpb>[0-9.]+) elapsed=(?P<elapsed>[0-9.]+)s"),
    "ttt_chunk": re.compile(r"ttt_chunk \[(?P<idx>\d+)/(?P<total>\d+)\] bpb=(?P<bpb>[0-9.]+) time=(?P<time>[0-9.]+)s"),
}


def extract_last(pattern: re.Pattern[str], text: str) -> dict[str, str] | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].groupdict()


def summarize_log(path: Path) -> dict[str, object]:
    text = path.read_text()
    row: dict[str, object] = {"path": str(path)}
    for key, pattern in PATTERNS.items():
        if key == "ttt_chunk":
            chunks = [m.groupdict() for m in pattern.finditer(text)]
            if chunks:
                row["ttt_last_chunk"] = chunks[-1]
                row["ttt_chunk_count"] = len(chunks)
            continue
        row[key] = extract_last(pattern, text)
    return row


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python summarize_scylla_ttt_sweep.py <log_dir>", file=sys.stderr)
        raise SystemExit(2)
    root = Path(sys.argv[1])
    rows = []
    for path in sorted(root.rglob("train.log")):
        rows.append(summarize_log(path))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
