from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("/Users/simon/Code/parameter-golf-local/data")
DEFAULT_CANDIDATES = [
    ("0054", str(ROOT / "autoresearch" / "tokenmonster_discovery" / "experiments" / "0054" / "candidate.vocab")),
    ("0045", str(ROOT / "autoresearch" / "tokenmonster_discovery" / "experiments" / "0045" / "candidate.vocab")),
    ("clean1024", "english-1024-clean-v1"),
    ("balanced1024", "english-1024-balanced-v1"),
    ("consistent1024", "english-1024-consistent-v1"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and sample-audit byte-native TokenMonster variants")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--docs", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/tokenmonster_byte_native_scan"))
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate as name=ref. If omitted, uses the default short list.",
    )
    return parser


def parse_candidates(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_CANDIDATES)
    out: list[tuple[str, str]] = []
    for value in values:
        name, ref = value.split("=", 1)
        out.append((name, ref))
    return out


def main() -> None:
    args = build_parser().parse_args()
    candidates = parse_candidates(args.candidate)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for name, ref in candidates:
        vocab_out = output_dir / f"{name}_byte_native.yaml"
        build_cmd = [
            sys.executable,
            str(ROOT / "data" / "build_tokenmonster_byte_native_variant.py"),
            "--base-vocab",
            ref,
            "--output-path",
            str(vocab_out),
            "--charset",
            "none",
            "--normalization",
            "none",
        ]
        build = subprocess.run(build_cmd, cwd=str(ROOT), capture_output=True, text=True)
        row: dict[str, object] = {
            "name": name,
            "base_vocab": ref,
            "build_returncode": build.returncode,
        }
        if build.stdout.strip():
            row["build_stdout"] = json.loads(build.stdout)
        if build.returncode != 0:
            row["build_stderr"] = build.stderr.strip()
            rows.append(row)
            continue

        audits: dict[str, object] = {}
        for mode in ("utf-8", "latin-1"):
            audit_cmd = [
                sys.executable,
                str(ROOT / "data" / "sample_audit_tokenmonster_vocab.py"),
                "--source-root",
                str(args.source_root.expanduser().resolve()),
                "--vocab",
                str(vocab_out),
                "--docs",
                str(args.docs),
                "--bytes-mode",
                mode,
            ]
            audit = subprocess.run(audit_cmd, cwd=str(ROOT), capture_output=True, text=True)
            audits[mode] = {
                "returncode": audit.returncode,
                "stdout": json.loads(audit.stdout) if audit.stdout.strip() else None,
                "stderr": audit.stderr.strip(),
            }
        row["audits"] = audits
        rows.append(row)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
