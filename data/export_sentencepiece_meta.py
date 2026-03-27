#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TOKENIZER_META_FORMAT_VERSION = 1
TOKENIZER_META_SUFFIX = ".meta.npz"


def derive_meta_path(tokenizer_path: Path) -> Path:
    if tokenizer_path.suffix == ".model":
        return tokenizer_path.with_suffix(TOKENIZER_META_SUFFIX)
    return tokenizer_path.with_name(tokenizer_path.name + TOKENIZER_META_SUFFIX)


def build_sentencepiece_luts(sp, vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary_token


def load_meta(meta_path: Path, vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(meta_path, allow_pickle=False) as data:
        format_version = int(data["format_version"])
        if format_version != TOKENIZER_META_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported tokenizer metadata format_version={format_version} in {meta_path}; "
                f"expected {TOKENIZER_META_FORMAT_VERSION}"
            )
        meta_vocab_size = int(data["vocab_size"])
        table_size = max(meta_vocab_size, vocab_size)
        base_bytes = np.zeros((table_size,), dtype=np.int16)
        has_leading_space = np.zeros((table_size,), dtype=np.bool_)
        is_boundary_token = np.ones((table_size,), dtype=np.bool_)
        base_src = np.asarray(data["base_bytes"], dtype=np.int16)
        lead_src = np.asarray(data["has_leading_space"], dtype=np.bool_)
        boundary_src = np.asarray(data["is_boundary_token"], dtype=np.bool_)
        if not (len(base_src) == len(lead_src) == len(boundary_src) == meta_vocab_size):
            raise ValueError(f"Tokenizer metadata arrays in {meta_path} do not match vocab_size={meta_vocab_size}")
        base_bytes[:meta_vocab_size] = base_src
        has_leading_space[:meta_vocab_size] = lead_src
        is_boundary_token[:meta_vocab_size] = boundary_src
        manifest = {
            "format_version": format_version,
            "tokenizer_kind": str(data["tokenizer_kind"].item()),
            "source_model_name": str(data["source_model_name"].item()),
            "vocab_size": meta_vocab_size,
        }
    return base_bytes, has_leading_space, is_boundary_token, manifest


def export_meta(model_path: Path, output_path: Path) -> dict[str, object]:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required to export tokenizer metadata") from exc

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = int(sp.vocab_size())
    base_bytes, has_leading_space, is_boundary_token = build_sentencepiece_luts(sp, vocab_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        format_version=np.array(TOKENIZER_META_FORMAT_VERSION, dtype=np.int32),
        tokenizer_kind=np.array("sentencepiece_lut", dtype=np.str_),
        source_model_name=np.array(model_path.name, dtype=np.str_),
        vocab_size=np.array(vocab_size, dtype=np.int32),
        base_bytes=base_bytes,
        has_leading_space=has_leading_space,
        is_boundary_token=is_boundary_token,
    )
    return {
        "format_version": TOKENIZER_META_FORMAT_VERSION,
        "tokenizer_kind": "sentencepiece_lut",
        "source_model_name": model_path.name,
        "vocab_size": vocab_size,
        "meta_path": str(output_path),
    }


def validate_export(model_path: Path, meta_path: Path) -> dict[str, object]:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required to validate tokenizer metadata parity") from exc

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = int(sp.vocab_size())
    expected = build_sentencepiece_luts(sp, vocab_size)
    actual = load_meta(meta_path, vocab_size)[:3]
    for name, actual_arr, expected_arr in zip(
        ("base_bytes", "has_leading_space", "is_boundary_token"),
        actual,
        expected,
        strict=True,
    ):
        if not np.array_equal(actual_arr, expected_arr):
            raise SystemExit(
                f"metadata parity failed for {name}: model={model_path} meta={meta_path}"
            )
    return {
        "validated": True,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "vocab_size": vocab_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export precomputed tokenizer LUT metadata from a SentencePiece .model file."
    )
    parser.add_argument("model_path", type=Path, help="Path to the SentencePiece .model file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .meta.npz path. Defaults to <model>.meta.npz beside the model.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Re-load the exported metadata and assert exact LUT parity against SentencePiece.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_path = (args.output.expanduser().resolve() if args.output else derive_meta_path(model_path))
    export_info = export_meta(model_path, output_path)
    result = {"export": export_info}
    if args.validate:
        result["validate"] = validate_export(model_path, output_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
