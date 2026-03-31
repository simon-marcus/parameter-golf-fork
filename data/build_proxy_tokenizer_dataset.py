from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenmonster_utils import load_tokenmonster_vocab


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1


def write_datafile(path: Path, toks: np.ndarray) -> None:
    if len(toks) >= 2**31:
        raise ValueError("token count too large")
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(toks)
    toks = np.asarray(toks)
    if toks.dtype != np.uint16:
        if not ((0 <= toks).all() and (toks < 2**16).all()):
            raise ValueError("token dictionary too large for uint16")
        toks = toks.astype("<u2", copy=False)
    else:
        toks = toks.astype("<u2", copy=False)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def iter_texts(path: Path, max_chunks: int | None = None):
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_chunks is not None and i >= max_chunks:
                break
            payload = json.loads(line)
            yield str(payload["text"])


def encode_sentencepiece(model_path: Path, text_path: Path, *, max_chunks: int | None) -> np.ndarray:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required for build_proxy_tokenizer_dataset.py") from exc
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    pieces: list[np.ndarray] = []
    for text in iter_texts(text_path, max_chunks=max_chunks):
        ids = sp.encode(text, out_type=int)
        if ids:
            pieces.append(np.asarray(ids, dtype=np.uint16))
    if not pieces:
        raise ValueError(f"No encoded text from {text_path}")
    return np.concatenate(pieces)


def encode_tokenmonster(vocab_ref: str, text_path: Path, *, max_chunks: int | None) -> np.ndarray:
    vocab = load_tokenmonster_vocab(vocab_ref)
    pieces: list[np.ndarray] = []
    for text in iter_texts(text_path, max_chunks=max_chunks):
        ids = np.asarray(vocab.tokenize(text), dtype=np.uint16).reshape(-1)
        if ids.size:
            pieces.append(ids)
    if not pieces:
        raise ValueError(f"No encoded text from {text_path}")
    return np.concatenate(pieces)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retokenize decoded text samples into proxy FineWeb shard files")
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--val-text", required=True)
    parser.add_argument("--family", choices=("sentencepiece", "tokenmonster"), default="sentencepiece")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-max-chunks", type=int, default=0)
    parser.add_argument("--val-max-chunks", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_text = Path(args.train_text).expanduser().resolve()
    val_text = Path(args.val_text).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_max_chunks = None if args.train_max_chunks <= 0 else int(args.train_max_chunks)
    val_max_chunks = None if args.val_max_chunks <= 0 else int(args.val_max_chunks)

    if args.family == "sentencepiece":
        tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
        train_tokens = encode_sentencepiece(tokenizer_path, train_text, max_chunks=train_max_chunks)
        val_tokens = encode_sentencepiece(tokenizer_path, val_text, max_chunks=val_max_chunks)
        tokenizer_descriptor = str(tokenizer_path)
    else:
        train_tokens = encode_tokenmonster(args.tokenizer_path, train_text, max_chunks=train_max_chunks)
        val_tokens = encode_tokenmonster(args.tokenizer_path, val_text, max_chunks=val_max_chunks)
        tokenizer_descriptor = args.tokenizer_path

    write_datafile(output_dir / "fineweb_train_000000.bin", train_tokens)
    write_datafile(output_dir / "fineweb_val_000000.bin", val_tokens)

    summary = {
        "family": args.family,
        "tokenizer_path": tokenizer_descriptor,
        "output_dir": str(output_dir),
        "train_tokens": int(train_tokens.size),
        "val_tokens": int(val_tokens.size),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
