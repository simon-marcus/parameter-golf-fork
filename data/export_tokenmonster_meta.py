from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenmonster_utils import load_tokenmonster_vocab


FORMAT_VERSION = 1


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def require_exact_byte_preserving_vocab(vocab, *, vocab_ref: str, allow_normalized: bool) -> str:
    normalization = str(vocab.normalization())
    if normalization != "None" and not allow_normalized:
        raise RuntimeError(
            "refusing to export competition metadata for a normalizing TokenMonster vocabulary: "
            f"{vocab_ref!r} reports normalization={normalization!r}. "
            "This vocabulary can change the underlying byte stream and must not be used for exact leaderboard accounting."
        )
    return normalization


def build_tokenmonster_luts(vocab_ref: str, *, allow_normalized: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    vocab = load_tokenmonster_vocab(vocab_ref)
    normalization = require_exact_byte_preserving_vocab(
        vocab, vocab_ref=vocab_ref, allow_normalized=allow_normalized
    )
    vocab_size = int(vocab.vocab_size)
    base_bytes = np.zeros((vocab_size,), dtype=np.int16)
    has_leading_space = np.zeros((vocab_size,), dtype=np.bool_)
    is_boundary_token = np.zeros((vocab_size,), dtype=np.bool_)
    dictionary = vocab.get_dictionary()
    for token_id in range(vocab_size):
        item = dictionary[token_id]
        piece = str(item["token_decoded"])
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary_token, vocab_size, normalization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export runtime tokenizer metadata for a TokenMonster vocabulary")
    parser.add_argument("vocab_ref", help="TokenMonster vocabulary reference, e.g. english-4096-clean-v1")
    parser.add_argument("--output", default="", help="Output .meta.npz path")
    parser.add_argument(
        "--allow-normalized",
        action="store_true",
        help="Allow export even if the vocabulary reports non-None normalization. Unsafe for exact competition accounting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    vocab_ref = args.vocab_ref
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = Path.cwd() / f"{sanitize_name(vocab_ref)}.meta.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_bytes, has_leading_space, is_boundary_token, vocab_size, normalization = build_tokenmonster_luts(
        vocab_ref, allow_normalized=args.allow_normalized
    )
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
        "normalization": normalization,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
