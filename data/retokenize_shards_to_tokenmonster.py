from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np


VERSION = "10B"
DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
APPEND_EOS = False
DEFAULT_SHARD_SIZE = 10**8
DECODE_BATCH_DOCS = max(1, int(os.environ.get("RETOKENIZE_DECODE_BATCH_DOCS", "512")))


def write_datafile(path: Path, toks: np.ndarray) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(toks)
    toks = np.asarray(toks, dtype="<u2")
    with path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(toks.tobytes())


def maybe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def relativize_manifest_paths(value, root: Path):
    if isinstance(value, dict):
        return {k: relativize_manifest_paths(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [relativize_manifest_paths(v, root) for v in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                return value
    return value


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


def iter_docs_from_shards(paths: list[Path], *, bos_id: int) -> Iterable[list[int]]:
    current: list[int] | None = None
    for path in paths:
        toks = read_tokens(path)
        for token in toks.tolist():
            if token == bos_id:
                if current is not None:
                    yield current
                current = []
            else:
                if current is None:
                    raise ValueError(f"encountered token before BOS while reading {path}")
                current.append(int(token))
    if current is not None:
        yield current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retokenize existing SentencePiece shards into a TokenMonster dataset bundle")
    parser.add_argument("--source-root", required=True, help="Root with manifest.json, tokenizers/, datasets/")
    parser.add_argument("--source-dataset", default="fineweb10B_sp1024")
    parser.add_argument("--source-tokenizer", default="fineweb_1024_bpe.model")
    parser.add_argument("--tokenizer-path", required=True, help="Target TokenMonster .vocab path or ref")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tokenizer-name", default="tm_candidate")
    parser.add_argument("--dataset-name", default="fineweb10B_tm_candidate")
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--max-train-shards", type=int, default=0)
    parser.add_argument("--max-val-shards", type=int, default=0)
    parser.add_argument("--max-train-docs", type=int, default=0)
    parser.add_argument("--max-val-docs", type=int, default=0)
    return parser


def flush_buf(buf: np.ndarray, fill: int, split: str, shard_idx: dict[str, int], output_dir: Path, stats: dict[str, int]) -> int:
    if fill == 0:
        return 0
    write_datafile(output_dir / f"fineweb_{split}_{shard_idx[split]:06d}.bin", buf[:fill])
    shard_idx[split] += 1
    stats["files_total"] += 1
    stats[f"files_{split}"] += 1
    return 0


def append_doc_to_shards(
    toks: np.ndarray,
    *,
    split: str,
    buf: np.ndarray,
    fill: int,
    shard_idx: dict[str, int],
    output_dir: Path,
    stats: dict[str, int],
) -> int:
    pos = 0
    while pos < len(toks):
        take = min(buf.size - fill, len(toks) - pos)
        buf[fill : fill + take] = toks[pos : pos + take]
        fill += take
        pos += take
        if fill == buf.size:
            fill = flush_buf(buf, fill, split, shard_idx, output_dir, stats)
    return fill


def build_target_tokens(encoded: np.ndarray, *, bos_id: int | None, eos_id: int | None, vocab_size: int) -> np.ndarray:
    extra = int(bos_id is not None) + int(APPEND_EOS and eos_id is not None)
    toks = np.empty((encoded.size + extra,), dtype=np.int32)
    pos = 0
    if bos_id is not None:
        toks[pos] = bos_id
        pos += 1
    toks[pos : pos + encoded.size] = encoded
    pos += encoded.size
    if APPEND_EOS and eos_id is not None:
        toks[pos] = eos_id
    if not ((0 <= toks).all() and (toks < vocab_size).all()):
        raise ValueError("target token id out of range")
    return toks.astype(np.uint16, copy=False)


def main() -> None:
    args = build_parser().parse_args()
    try:
        import sentencepiece as spm
        import tokenmonster
    except ImportError as exc:
        raise RuntimeError("sentencepiece and tokenmonster are required") from exc

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    dataset_entry = next(x for x in source_manifest["datasets"] if x["name"] == args.source_dataset)
    num_val_docs = int(source_manifest.get("num_val_docs", 50_000))
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    source_tokenizer_path = source_root / "tokenizers" / args.source_tokenizer
    sp = spm.SentencePieceProcessor(model_file=str(source_tokenizer_path))
    source_bos = int(sp.bos_id())
    source_train = sorted((source_root / str(dataset_entry["path"])).glob("fineweb_train_*.bin"))
    source_val = sorted((source_root / str(dataset_entry["path"])).glob("fineweb_val_*.bin"))
    if args.max_train_shards > 0:
        source_train = source_train[: int(args.max_train_shards)]
    if args.max_val_shards > 0:
        source_val = source_val[: int(args.max_val_shards)]

    vocab_ref = args.tokenizer_path
    vocab = tokenmonster.load(vocab_ref)
    vocab_size = int(vocab.vocab_size)
    bos_id: int | None = None
    eos_id: int | None = None

    tokenizer_src = Path(vocab_ref).expanduser()
    tokenizer_dst = tokenizers_dir / tokenizer_src.name
    if tokenizer_src.exists():
        maybe_link(tokenizer_src.resolve(), tokenizer_dst)
    else:
        vocab.save(str(tokenizer_dst))
    meta_path = tokenizers_dir / f"{tokenizer_dst.stem}.meta.npz"
    np.savez_compressed(
        meta_path,
        format_version=np.int64(1),
        tokenizer_kind=np.array("tokenmonster"),
        source_model_name=np.array(str(vocab_ref)),
        vocab_size=np.int64(vocab_size),
        base_bytes=np.asarray(
            [len(str(vocab.get_dictionary()[i]["token_decoded"]).encode("utf-8")) for i in range(vocab_size)],
            dtype=np.int16,
        ),
        has_leading_space=np.zeros((vocab_size,), dtype=np.bool_),
        is_boundary_token=np.zeros((vocab_size,), dtype=np.bool_),
    )

    output_dir = datasets_dir / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("fineweb_*.bin"):
        stale.unlink()

    stats = {
        "docs_total": 0,
        "docs_val": 0,
        "docs_train": 0,
        "files_total": 0,
        "files_val": 0,
        "files_train": 0,
        "tokens_total": 0,
        "tokens_val": 0,
        "tokens_train": 0,
    }
    shard_idx = {"train": 0, "val": 0}
    buf = np.empty((int(args.chunk_tokens),), dtype=np.uint16)
    fill = 0
    split = "val"

    def process_docs(doc_iter: Iterable[list[int]], split_name: str, *, max_docs: int) -> None:
        nonlocal fill, split
        batch: list[list[int]] = []
        processed = 0
        for doc_tokens in doc_iter:
            if max_docs > 0 and processed >= max_docs:
                break
            batch.append(doc_tokens)
            if len(batch) >= DECODE_BATCH_DOCS:
                texts = sp.decode(batch)
                encoded_docs = vocab.tokenize(texts)
                for encoded in encoded_docs:
                    if split != split_name:
                        fill = flush_buf(buf, fill, split, shard_idx, output_dir, stats)
                        split = split_name
                    arr = np.asarray(encoded, dtype=np.int32).reshape(-1)
                    toks = build_target_tokens(arr, bos_id=bos_id, eos_id=eos_id, vocab_size=vocab_size)
                    fill = append_doc_to_shards(
                        toks, split=split_name, buf=buf, fill=fill, shard_idx=shard_idx, output_dir=output_dir, stats=stats
                    )
                    stats["docs_total"] += 1
                    stats[f"docs_{split_name}"] += 1
                    stats["tokens_total"] += int(toks.size)
                    stats[f"tokens_{split_name}"] += int(toks.size)
                    processed += 1
                    if stats["docs_total"] and stats["docs_total"] % 100_000 == 0:
                        print(f"{args.dataset_name}: {stats['docs_total']} docs", flush=True)
                batch.clear()
        if batch:
            if max_docs > 0:
                remaining = max(0, max_docs - processed)
                batch = batch[:remaining]
            if not batch:
                return
            texts = sp.decode(batch)
            encoded_docs = vocab.tokenize(texts)
            for encoded in encoded_docs:
                if split != split_name:
                    fill = flush_buf(buf, fill, split, shard_idx, output_dir, stats)
                    split = split_name
                arr = np.asarray(encoded, dtype=np.int32).reshape(-1)
                toks = build_target_tokens(arr, bos_id=bos_id, eos_id=eos_id, vocab_size=vocab_size)
                fill = append_doc_to_shards(
                    toks, split=split_name, buf=buf, fill=fill, shard_idx=shard_idx, output_dir=output_dir, stats=stats
                )
                stats["docs_total"] += 1
                stats[f"docs_{split_name}"] += 1
                stats["tokens_total"] += int(toks.size)
                stats[f"tokens_{split_name}"] += int(toks.size)
                processed += 1

    process_docs(iter_docs_from_shards(source_val, bos_id=source_bos), "val", max_docs=int(args.max_val_docs))
    process_docs(iter_docs_from_shards(source_train, bos_id=source_bos), "train", max_docs=int(args.max_train_docs))
    flush_buf(buf, fill, split, shard_idx, output_dir, stats)

    manifest = {
        "version": source_manifest.get("version", VERSION),
        "num_docs": stats["docs_total"],
        "num_val_docs": num_val_docs,
        "shuffle_seed": source_manifest.get("shuffle_seed"),
        "dataset_revision": source_manifest.get("dataset_revision"),
        "shard_size": int(args.chunk_tokens),
        "append_eos": APPEND_EOS,
        "docs_jsonl": source_manifest.get("docs_jsonl"),
        "docs_meta": source_manifest.get("docs_meta", {}),
        "tokenizer_specs": [],
        "tokenizers": [
            {
                "name": args.tokenizer_name,
                "kind": "tokenmonster",
                "vocab_size": vocab_size,
                "bos_id": -1,
                "eos_id": -1,
                "recommended_bigram_vocab_size": ((vocab_size + 127) // 128) * 128 * 5,
                "path": str(tokenizer_dst),
                "meta_path": str(meta_path),
                "source_spec": {"kind": "tokenmonster", "source_model": str(vocab_ref)},
            }
        ],
        "datasets": [
            {
                "name": args.dataset_name,
                "tokenizer_name": args.tokenizer_name,
                "tokenizer_kind": "tokenmonster",
                "path": str(output_dir),
                "train_glob": str(output_dir / "fineweb_train_*.bin"),
                "val_glob": str(output_dir / "fineweb_val_*.bin"),
                "vocab_size": vocab_size,
                "bos_id": -1,
                "eos_id": -1,
                "recommended_bigram_vocab_size": ((vocab_size + 127) // 128) * 128 * 5,
                "stats": stats,
            }
        ],
    }
    manifest = relativize_manifest_paths(manifest, output_root)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
