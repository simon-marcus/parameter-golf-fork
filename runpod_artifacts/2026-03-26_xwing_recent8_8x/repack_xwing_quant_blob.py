#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zlib
from pathlib import Path


def ensure_zstandard():
    try:
        import zstandard as zstd  # type: ignore
        return zstd
    except Exception:
        raise SystemExit(
            "zstandard is not installed. Install it with "
            "`python3 -m pip install --break-system-packages zstandard` "
            "or run setup again after requirements.txt is updated."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Repack X-WING quant blob from zlib to zstd")
    ap.add_argument("input", help="Path to existing final_model.int6.ptz produced with zlib")
    ap.add_argument("--output", default="", help="Output path; default replaces .ptz suffix with .zst.ptz")
    ap.add_argument("--code-bytes", type=int, default=0, help="Optional code size for total submission size reporting")
    ap.add_argument("--level", type=int, default=22, help="Requested zstd compression level")
    args = ap.parse_args()

    zstd = ensure_zstandard()
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + ".zst.ptz")

    blob = in_path.read_bytes()
    raw = zlib.decompress(blob)
    level = min(args.level, zstd.ZstdCompressor().compression_level if hasattr(zstd.ZstdCompressor(), "compression_level") else args.level)
    try:
        repacked = zstd.ZstdCompressor(level=args.level).compress(raw)
    except Exception:
        repacked = zstd.ZstdCompressor(level=19).compress(raw)
        level = 19
    out_path.write_bytes(repacked)

    print(f"input_bytes {len(blob)}")
    print(f"raw_bytes {len(raw)}")
    print(f"zstd_level {level}")
    print(f"output_bytes {len(repacked)}")
    if args.code_bytes > 0:
        print(f"total_submission_bytes {len(repacked) + args.code_bytes}")
    print(f"output_path {out_path}")


if __name__ == "__main__":
    main()
