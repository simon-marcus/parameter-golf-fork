from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FORMAT_VERSION = 1


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def build_tokenmonster_luts(vocab_ref: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    try:
        import tokenmonster
    except ImportError as exc:
        raise RuntimeError("tokenmonster is required for export_tokenmonster_meta.py") from exc

    vocab = tokenmonster.load(vocab_ref)
    vocab_size = int(vocab.vocab_size)
    base_bytes = np.zeros((vocab_size,), dtype=np.int16)
    has_leading_space = np.zeros((vocab_size,), dtype=np.bool_)
    is_boundary_token = np.zeros((vocab_size,), dtype=np.bool_)
    dictionary = vocab.get_dictionary()
    for token_id in range(vocab_size):
        item = dictionary[token_id]
        piece = str(item["token_decoded"])
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary_token, vocab_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export runtime tokenizer metadata for a TokenMonster vocabulary")
    parser.add_argument("vocab_ref", help="TokenMonster vocabulary reference, e.g. english-4096-clean-v1")
    parser.add_argument("--output", default="", help="Output .meta.npz path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    vocab_ref = args.vocab_ref
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = Path.cwd() / f"{sanitize_name(vocab_ref)}.meta.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_bytes, has_leading_space, is_boundary_token, vocab_size = build_tokenmonster_luts(vocab_ref)
    np.savez_compressed(
        output_path,
        format_version=np.int64(FORMAT_VERSION),
        tokenizer_kind=np.array("tokenmonster"),
        source_model_name=np.array(vocab_ref),
        vocab_size=np.int64(vocab_size),
        base_bytes=base_bytes,
        has_leading_space=has_leading_space,
        is_boundary_token=is_boundary_token,
    )
    summary = {
        "vocab_ref": vocab_ref,
        "output_path": str(output_path),
        "vocab_size": vocab_size,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
