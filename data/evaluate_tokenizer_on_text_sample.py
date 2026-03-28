from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def iter_texts(path: Path, max_chunks: int | None = None):
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_chunks is not None and i >= max_chunks:
                break
            yield json.loads(line)["text"]


def evaluate_sentencepiece(model_path: Path, sample_path: Path, *, max_chunks: int | None) -> dict[str, object]:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required for sentencepiece evaluation") from exc
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = int(sp.vocab_size())
    token_counter: Counter[int] = Counter()
    total_bytes = 0
    total_tokens = 0
    total_chunks = 0
    for text in iter_texts(sample_path, max_chunks=max_chunks):
        ids = sp.encode(text, out_type=int)
        total_chunks += 1
        total_tokens += len(ids)
        total_bytes += len(text.encode("utf-8"))
        token_counter.update(ids)
    unique_used = len(token_counter)
    return {
        "family": "sentencepiece",
        "model_path": str(model_path),
        "vocab_size": vocab_size,
        "chunks": total_chunks,
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "tokens_per_byte": total_tokens / max(total_bytes, 1),
        "bytes_per_token": total_bytes / max(total_tokens, 1),
        "unique_tokens_used": unique_used,
        "dead_vocab_frac": max(vocab_size - unique_used, 0) / max(vocab_size, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate tokenizer quality on a decoded text sample")
    parser.add_argument("--family", choices=("sentencepiece",), default="sentencepiece")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_path = Path(args.sample_path).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    max_chunks = None if args.max_chunks <= 0 else int(args.max_chunks)
    if args.family != "sentencepiece":
        raise ValueError(f"Unsupported tokenizer family: {args.family}")
    result = evaluate_sentencepiece(model_path, sample_path, max_chunks=max_chunks)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
