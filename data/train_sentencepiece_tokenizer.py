from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_texts(path: Path, max_chunks: int | None = None):
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_chunks is not None and i >= max_chunks:
                break
            text = json.loads(line)["text"].replace("\x00", " ").strip()
            if text:
                yield text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer on a decoded text sample")
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--model-type", choices=("bpe", "unigram"), default="bpe")
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--input-sentence-size", type=int, default=200000)
    parser.add_argument("--shuffle-input-sentence", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required for train_sentencepiece_tokenizer.py") from exc

    sample_path = Path(args.sample_path).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    max_chunks = None if args.max_chunks <= 0 else int(args.max_chunks)

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter_texts(sample_path, max_chunks=max_chunks),
        model_prefix=str(output_prefix),
        vocab_size=int(args.vocab_size),
        model_type=args.model_type,
        character_coverage=float(args.character_coverage),
        input_sentence_size=int(args.input_sentence_size),
        shuffle_input_sentence=bool(int(args.shuffle_input_sentence)),
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_id=0,
    )


if __name__ == "__main__":
    main()
