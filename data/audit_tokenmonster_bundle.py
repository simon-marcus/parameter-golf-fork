from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenmonster_utils import load_tokenmonster_vocab

DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1


def read_tokens(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        header = np.frombuffer(fh.read(256 * 4), dtype="<i4")
        if int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
            raise ValueError(f"unsupported datafile header in {path}")
        expected = int(header[2])
        toks = np.frombuffer(fh.read(), dtype="<u2")
    if toks.size != expected:
        raise ValueError(f"token count mismatch in {path}: expected {expected}, got {toks.size}")
    return toks


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_path(manifest: dict, dataset_name: str, bundle_root: Path) -> Path:
    for entry in manifest.get("datasets", []):
        if entry.get("name") == dataset_name:
            return bundle_root / Path(entry["path"])
    raise KeyError(f"dataset {dataset_name!r} not found in manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a TokenMonster bundle against the fixed SP1024 validation source")
    parser.add_argument("--source-root", type=Path, required=True, help="Root containing the source SP1024 manifest/tokenizer/dataset")
    parser.add_argument("--source-dataset", default="fineweb10B_sp1024")
    parser.add_argument("--source-tokenizer", default="fineweb_1024_bpe.model")
    parser.add_argument("--bundle-root", type=Path, required=True, help="TokenMonster bundle root containing manifest.json")
    parser.add_argument("--bundle-dataset", required=True)
    parser.add_argument("--bundle-tokenizer", required=True, help="Relative tokenizer vocab path inside the bundle")
    parser.add_argument("--bundle-meta", required=True, help="Relative tokenizer meta path inside the bundle")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if the exact decoded bytes differ from source bytes")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece and tokenmonster are required for audit_tokenmonster_bundle.py") from exc

    source_root = args.source_root.expanduser().resolve()
    bundle_root = args.bundle_root.expanduser().resolve()
    source_manifest = load_manifest(source_root / "manifest.json")
    bundle_manifest = load_manifest(bundle_root / "manifest.json")

    source_dataset_dir = dataset_path(source_manifest, args.source_dataset, source_root)
    bundle_dataset_dir = dataset_path(bundle_manifest, args.bundle_dataset, bundle_root)
    source_val_files = sorted(source_dataset_dir.glob("fineweb_val_*.bin"))
    bundle_val_files = sorted(bundle_dataset_dir.glob("fineweb_val_*.bin"))
    if not source_val_files or not bundle_val_files:
        raise FileNotFoundError("missing fineweb_val shards in source or bundle dataset")
    source_val_path = source_val_files[0]
    bundle_val_path = bundle_val_files[0]

    sp = spm.SentencePieceProcessor(model_file=str(source_root / "tokenizers" / args.source_tokenizer))
    vocab = load_tokenmonster_vocab(str(bundle_root / args.bundle_tokenizer))
    meta = np.load(bundle_root / args.bundle_meta, allow_pickle=False)

    source_val = read_tokens(source_val_path)
    bundle_val = read_tokens(bundle_val_path)

    source_bytes = 0
    chunk = 200_000
    for i in range(0, len(source_val), chunk):
        text = sp.DecodeIds(source_val[i : i + chunk].astype(int).tolist())
        source_bytes += len(text.encode("utf-8"))

    meta_bytes = int(np.asarray(meta["base_bytes"], dtype=np.int64)[bundle_val].sum())

    decoded_bytes = 0
    dec = vocab.decoder()
    for i in range(0, len(bundle_val), chunk):
        text = dec.decode(bundle_val[i : i + chunk])
        decoded_bytes += len(text.encode("utf-8"))

    summary = {
        "source_val_tokens": int(len(source_val)),
        "bundle_val_tokens": int(len(bundle_val)),
        "source_bytes": int(source_bytes),
        "meta_bytes": int(meta_bytes),
        "decoded_bytes": int(decoded_bytes),
        "meta_overcount_frac": float(meta_bytes / source_bytes - 1.0),
        "decoded_drift_frac": float(decoded_bytes / source_bytes - 1.0),
        "normalization": str(vocab.normalization()),
    }
    print(json.dumps(summary, indent=2))

    if args.strict and decoded_bytes != source_bytes:
        raise SystemExit(
            "TokenMonster bundle does not decode back to the exact source validation byte stream: "
            f"source_bytes={source_bytes} decoded_bytes={decoded_bytes}"
        )


if __name__ == "__main__":
    main()
