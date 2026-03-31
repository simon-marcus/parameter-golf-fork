from __future__ import annotations

import argparse
import json
from collections import Counter
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


def first_diff(a: bytes, b: bytes) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def classify(src: bytes, dec: bytes) -> str:
    idx = first_diff(src, dec)
    src_tail = src[idx : idx + 8]
    dec_tail = dec[idx : idx + 8]
    if any(x >= 0x80 for x in src_tail):
        return "non_ascii_or_utf8"
    if dec[idx : idx + 1] == b" " and src[idx : idx + 1] != b" ":
        return "inserted_space"
    if src[idx : idx + 1] == b" " and dec[idx : idx + 1] != b" ":
        return "missing_space"
    if dec[idx : idx + 1].lower() == src[idx : idx + 1].lower() and dec[idx : idx + 1] != src[idx : idx + 1]:
        return "case_shift"
    if len(dec) < len(src):
        return "byte_drop"
    if len(dec) > len(src):
        return "byte_insert"
    return "other"


def context_window(data: bytes, idx: int, radius: int = 24) -> str:
    start = max(0, idx - radius)
    end = min(len(data), idx + radius)
    return data[start:end].hex()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze exactness failures for a TokenMonster vocab against SP1024 val docs")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-tokenizer", default="fineweb_1024_bpe.model")
    parser.add_argument("--source-val", default="datasets/fineweb10B_sp1024/fineweb_val_000000.bin")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--docs", type=int, default=200, help="Number of docs to analyze; <=0 means all docs in the source val shard")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required") from exc

    source_root = args.source_root.expanduser().resolve()
    sp = spm.SentencePieceProcessor(model_file=str(source_root / "tokenizers" / args.source_tokenizer))
    vocab = load_tokenmonster_vocab(args.vocab)
    source = read_tokens(source_root / args.source_val)
    bos = int(sp.bos_id())

    docs: list[list[int]] = []
    current: list[int] = []
    for token in source.tolist():
        if token == bos:
            if current:
                docs.append(current)
                if args.docs > 0 and len(docs) >= args.docs:
                    break
            current = []
        else:
            current.append(int(token))
    if current and (args.docs <= 0 or len(docs) < args.docs):
        docs.append(current)

    texts = sp.decode(docs if args.docs <= 0 else docs[: args.docs])
    failures: list[dict[str, object]] = []
    cluster_counter: Counter[str] = Counter()
    for idx, text in enumerate(texts):
        src = text.encode("utf-8")
        toks = np.asarray(vocab.tokenize(src)).reshape(-1).tolist()
        decoded = vocab.decode(toks)
        out = decoded if isinstance(decoded, bytes) else decoded.encode("latin-1")
        if out == src:
            continue
        diff_at = first_diff(src, out)
        cluster = classify(src, out)
        cluster_counter.update([cluster])
        failures.append(
            {
                "doc_index": idx,
                "cluster": cluster,
                "src_len": len(src),
                "dec_len": len(out),
                "delta": len(out) - len(src),
                "first_diff": diff_at,
                "src_context_hex": context_window(src, diff_at),
                "dec_context_hex": context_window(out, diff_at),
                "tokens_head": toks[:24],
                "raw_tokens_head": [str(vocab.id_to_token(int(t))) for t in toks[:24]],
            }
        )

    print(
        json.dumps(
            {
                "vocab": args.vocab,
                "docs_checked": len(texts),
                "bad_docs": len(failures),
                "clusters": dict(cluster_counter),
                "failures": failures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
