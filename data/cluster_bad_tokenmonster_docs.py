from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
    parser = argparse.ArgumentParser(description="Cluster bad TokenMonster audit docs by failure mechanism")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-tokenizer", default="fineweb_1024_bpe.model")
    parser.add_argument("--source-val", default="datasets/fineweb10B_sp1024/fineweb_val_000000.bin")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--docs", type=int, default=200)
    parser.add_argument("--max-bad", type=int, default=13)
    parser.add_argument("--bytes-mode", default="latin-1", choices=("utf-8", "latin-1"))
    return parser


def first_diff(src: bytes, dec: bytes) -> int | None:
    for i, (a, b) in enumerate(zip(src, dec)):
        if a != b:
            return i
    if len(src) != len(dec):
        return min(len(src), len(dec))
    return None


def classify(delta: int, src: bytes) -> str:
    has_non_ascii = any(b >= 128 for b in src)
    if delta == -1 and not has_non_ascii:
        return "leading_minus1_ascii"
    if delta in (-3, -4) or (delta < 0 and has_non_ascii and delta > -10):
        return "small_multibyte_or_repeated_loss"
    if delta <= -10 or (has_non_ascii and delta <= -10):
        return "large_multibyte"
    if has_non_ascii:
        return "mixed_non_ascii"
    return "other"


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
                if len(docs) >= args.docs:
                    break
            current = []
        else:
            current.append(int(token))
    if len(docs) < args.docs and current:
        docs.append(current)

    texts = sp.decode(docs[: args.docs])
    bad_rows: list[dict[str, object]] = []
    for idx, text in enumerate(texts):
        src = text.encode("utf-8")
        toks = np.asarray(vocab.tokenize(src)).reshape(-1).tolist()
        decoded = vocab.decode(toks)
        if isinstance(decoded, str):
            dec = decoded.encode("latin-1" if args.bytes_mode == "latin-1" else "utf-8")
        else:
            dec = bytes(decoded)
        if dec == src:
            continue
        raw_head = []
        for tid in toks[:20]:
            try:
                raw_head.append(str(vocab.id_to_token(int(tid))))
            except Exception:
                raw_head.append(f"<id:{tid}>")
        diff_at = first_diff(src, dec)
        row = {
            "doc_index": idx,
            "src_len": len(src),
            "dec_len": len(dec),
            "delta": len(dec) - len(src),
            "first_diff": diff_at,
            "has_non_ascii": any(b >= 128 for b in src),
            "src_prefix": src[:80].hex(),
            "dec_prefix": dec[:80].hex(),
            "tokens_head": toks[:20],
            "raw_tokens_head": raw_head,
        }
        row["cluster"] = classify(int(row["delta"]), src)
        bad_rows.append(row)
        if len(bad_rows) >= args.max_bad:
            break

    clusters: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in bad_rows:
        clusters[str(row["cluster"])].append(row)

    summary_clusters: list[dict[str, object]] = []
    for name, rows in sorted(clusters.items()):
        token_counter: Counter[int] = Counter()
        raw_counter: Counter[str] = Counter()
        deltas = [int(r["delta"]) for r in rows]
        firsts = [r["first_diff"] for r in rows if r["first_diff"] is not None]
        for r in rows:
            token_counter.update(int(t) for t in r["tokens_head"])
            raw_counter.update(str(t) for t in r["raw_tokens_head"])
        summary_clusters.append(
            {
                "cluster": name,
                "doc_indexes": [int(r["doc_index"]) for r in rows],
                "count": len(rows),
                "deltas": deltas,
                "first_diff_positions": firsts,
                "top_token_ids": [[int(k), int(v)] for k, v in token_counter.most_common(12)],
                "top_raw_tokens": [[k, int(v)] for k, v in raw_counter.most_common(12)],
            }
        )

    print(
        json.dumps(
            {
                "vocab": args.vocab,
                "bytes_mode": args.bytes_mode,
                "bad_doc_indexes": [int(r["doc_index"]) for r in bad_rows],
                "clusters": summary_clusters,
                "bad_rows": bad_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
