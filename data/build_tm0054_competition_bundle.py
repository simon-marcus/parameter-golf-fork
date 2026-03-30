from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "data"
DEFAULT_TOKENIZER_PATH = ROOT / "autoresearch" / "tokenmonster_discovery" / "experiments" / "0054" / "candidate.vocab"
DEFAULT_OUTPUT_ROOT = Path("/Users/simon/Code/parameter-golf-local/tm0054_competition_export")


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_stats(manifest: dict, dataset_name: str) -> dict:
    for entry in manifest.get("datasets", []):
        if entry.get("name") == dataset_name:
            return entry.get("stats") or {}
    raise KeyError(f"dataset {dataset_name!r} not found in manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate a full-data tm0054 competition bundle")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-dataset", default="fineweb10B_sp1024")
    parser.add_argument("--source-tokenizer", default="fineweb_1024_bpe.model")
    parser.add_argument("--train-shards", type=int, default=80)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-name", default="tm0054_candidate")
    parser.add_argument("--dataset-name", default="fineweb10B_tm0054")
    parser.add_argument("--chunk-tokens", type=int, default=100_000_000)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-retokenize", action="store_true")
    parser.add_argument("--with-docs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    tokenizer_path = args.tokenizer_path.expanduser().resolve()

    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing source manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    source_stats = dataset_stats(manifest, args.source_dataset)
    max_train_shards = int(source_stats.get("files_train", 0))
    val_shards = int(source_stats.get("files_val", 0))
    if args.train_shards <= 0:
        raise ValueError("train_shards must be positive")
    if max_train_shards and args.train_shards > max_train_shards:
        raise ValueError(
            f"requested {args.train_shards} train shards but source manifest only has {max_train_shards}"
        )

    if not args.skip_download:
        cmd = [
            sys.executable,
            str(ROOT / "data" / "cached_challenge_fineweb.py"),
            "--variant",
            "sp1024",
            "--train-shards",
            str(args.train_shards),
        ]
        if args.with_docs:
            cmd.append("--with-docs")
        run(cmd, cwd=ROOT)

    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"missing tm0054 tokenizer vocab: {tokenizer_path}")

    if not args.skip_retokenize:
        run(
            [
                sys.executable,
                str(ROOT / "data" / "retokenize_shards_to_tokenmonster.py"),
                "--source-root",
                str(source_root),
                "--source-dataset",
                args.source_dataset,
                "--source-tokenizer",
                args.source_tokenizer,
                "--tokenizer-path",
                str(tokenizer_path),
                "--output-root",
                str(output_root),
                "--tokenizer-name",
                args.tokenizer_name,
                "--dataset-name",
                args.dataset_name,
                "--chunk-tokens",
                str(args.chunk_tokens),
                "--max-train-shards",
                str(args.train_shards),
                "--max-val-shards",
                str(val_shards),
            ],
            cwd=ROOT,
        )

    bundle_manifest_path = output_root / "manifest.json"
    if not bundle_manifest_path.is_file():
        raise FileNotFoundError(f"missing output manifest: {bundle_manifest_path}")
    bundle_manifest = load_manifest(bundle_manifest_path)
    bundle_stats = dataset_stats(bundle_manifest, args.dataset_name)
    expected_train = int(bundle_stats.get("files_train", 0))
    expected_val = int(bundle_stats.get("files_val", 0))
    if expected_train <= 0 or expected_val <= 0:
        raise RuntimeError(
            f"unexpected output shard counts in bundle manifest: train={expected_train} val={expected_val}"
        )

    data_path = output_root / "datasets" / args.dataset_name
    meta_path = output_root / "tokenizers" / f"{tokenizer_path.stem}.meta.npz"
    run(
        [
            str(ROOT / "verify_runpod_data_ready.sh"),
            str(data_path),
            str(meta_path),
            str(expected_train),
            str(expected_val),
        ],
        cwd=ROOT,
    )

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_dataset": args.source_dataset,
        "dataset_name": args.dataset_name,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_meta_path": str(meta_path),
        "source_train_shards_requested": args.train_shards,
        "source_val_shards": val_shards,
        "bundle_train_shards": expected_train,
        "bundle_val_shards": expected_val,
        "bundle_stats": bundle_stats,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
