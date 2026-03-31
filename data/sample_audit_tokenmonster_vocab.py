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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample exactness audit for a TokenMonster vocab against SP1024 docs")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-tokenizer", default="fineweb_1024_bpe.model")
    parser.add_argument("--source-val", default="datasets/fineweb10B_sp1024/fineweb_val_000000.bin")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--docs", type=int, default=200, help="Number of docs to audit; <=0 means all docs in the source val shard")
    parser.add_argument("--max-bad", type=int, default=10)
    parser.add_argument("--bytes-mode", default="utf-8", choices=("utf-8", "latin-1"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece and tokenmonster are required for sample audit") from exc

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
    ok_docs = 0
    bad_examples: list[dict[str, object]] = []
    bad_token_counter: Counter[int] = Counter()
    bad_raw_counter: Counter[str] = Counter()
    total_src_bytes = 0
    total_dec_bytes = 0
    for idx, text in enumerate(texts):
        src = text.encode("utf-8")
        toks = vocab.tokenize(src)
        decoded = vocab.decode(toks)
        if isinstance(decoded, str):
            out = decoded.encode("latin-1" if args.bytes_mode == "latin-1" else "utf-8")
        else:
            out = decoded
        total_src_bytes += len(src)
        total_dec_bytes += len(out)
        if out == src:
            ok_docs += 1
            continue
        token_list = np.asarray(toks).reshape(-1).tolist()
        bad_token_counter.update(int(t) for t in token_list[:20])
        for tid in token_list[:20]:
            try:
                bad_raw_counter.update([str(vocab.id_to_token(int(tid)))])
            except Exception:
                pass
        if len(bad_examples) < args.max_bad:
            bad_examples.append(
                {
                    "doc_index": idx,
                    "src_len": len(src),
                    "dec_len": len(out),
                    "src_prefix": src[:80].hex(),
                    "dec_prefix": out[:80].hex(),
                    "tokens_head": np.asarray(toks).reshape(-1)[:20].tolist(),
                }
            )

    summary = {
        "vocab": args.vocab,
        "normalization": str(vocab.normalization()),
        "capcode": int(vocab.capcode()),
        "docs_checked": len(texts),
        "ok_docs": ok_docs,
        "bad_docs": len(texts) - ok_docs,
        "src_bytes": total_src_bytes,
        "dec_bytes": total_dec_bytes,
        "byte_drift": total_dec_bytes - total_src_bytes,
        "bytes_mode": args.bytes_mode,
        "bad_token_ids_top": [[int(k), int(v)] for k, v in bad_token_counter.most_common(15)],
        "bad_raw_tokens_top": [[k, int(v)] for k, v in bad_raw_counter.most_common(15)],
        "bad_examples": bad_examples,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
