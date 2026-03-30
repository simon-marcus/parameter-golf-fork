from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "10B"
NUM_VAL_DOCS = 50_000
SHARD_SIZE = 10**8
APPEND_EOS = False
DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
TOKENMONSTER_BATCH_SIZE = max(1, int(os.environ.get("TOKENMONSTER_BATCH_SIZE", "1024")))


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


def maybe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def relativize_manifest_paths(value: Any, root: Path) -> Any:
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


def docs_sidecar_path(docs_jsonl: Path) -> Path:
    return docs_jsonl.with_name(f"{docs_jsonl.stem}.source_manifest.json")


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def count_docs(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def batched_docs_jsonl(path: Path, batch_size: int):
    batch: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            batch.append(json.loads(line)["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def export_shards(
    docs_jsonl: Path,
    *,
    vocab: object,
    vocab_size: int,
    bos_id: int,
    eos_id: int,
    output_dir: Path,
    num_val_docs: int,
    shard_size: int,
    docs_total: int,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("fineweb_train_*.bin", "fineweb_val_*.bin"):
        for stale in output_dir.glob(pattern):
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
    buf = np.empty((shard_size,), dtype=np.uint16)
    fill = 0
    split = "val"
    shards = {"val": 0, "train": 0}

    def flush() -> None:
        nonlocal fill
        if fill == 0:
            return
        write_datafile(output_dir / f"fineweb_{split}_{shards[split]:06d}.bin", buf[:fill])
        stats["files_total"] += 1
        stats[f"files_{split}"] += 1
        shards[split] += 1
        fill = 0

    for texts in batched_docs_jsonl(docs_jsonl, TOKENMONSTER_BATCH_SIZE):
        encoded_docs = vocab.tokenize(texts)
        for encoded in encoded_docs:
            split_for_doc = "val" if stats["docs_total"] < num_val_docs else "train"
            if split_for_doc != split:
                flush()
                split = split_for_doc

            encoded_arr = np.asarray(encoded, dtype=np.int32).reshape(-1)
            toks = np.empty((encoded_arr.size + 1 + int(APPEND_EOS),), dtype=np.int32)
            toks[0] = bos_id
            toks[1 : 1 + encoded_arr.size] = encoded_arr
            if APPEND_EOS:
                toks[-1] = eos_id
            if not ((0 <= toks).all() and (toks < vocab_size).all()):
                bad = int(toks[(toks < 0) | (toks >= vocab_size)][0])
                raise ValueError(f"token id {bad} outside declared vocab_size={vocab_size}")
            toks = toks.astype("<u2", copy=False)

            stats["docs_total"] += 1
            stats[f"docs_{split}"] += 1
            stats["tokens_total"] += len(toks)
            stats[f"tokens_{split}"] += len(toks)

            pos = 0
            while pos < len(toks):
                take = min(shard_size - fill, len(toks) - pos)
                buf[fill : fill + take] = toks[pos : pos + take]
                fill += take
                pos += take
                if fill == shard_size:
                    flush()

        if stats["docs_total"] and stats["docs_total"] % 100_000 == 0:
            print(f"{output_dir.name}: {stats['docs_total']}/{docs_total} docs", flush=True)

    flush()
    if stats["docs_total"] != docs_total:
        raise ValueError(f"expected {docs_total} docs, exported {stats['docs_total']}")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a full FineWeb dataset bundle using a TokenMonster vocabulary")
    parser.add_argument("--source-root", required=True, help="Root containing manifest.json and docs_selected.jsonl")
    parser.add_argument("--tokenizer-path", required=True, help="Local TokenMonster .vocab path or named ref")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tokenizer-name", default="tm_candidate")
    parser.add_argument("--dataset-name", default="fineweb10B_tm_candidate")
    parser.add_argument("--chunk-tokens", type=int, default=SHARD_SIZE)
    parser.add_argument("--num-val-docs", type=int, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive")

    try:
        import tokenmonster
    except ImportError as exc:
        raise RuntimeError("tokenmonster is required for export_tokenmonster_docs_and_tokenize.py") from exc

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source_root / "manifest.json"
    source_manifest = maybe_load_json(source_manifest_path) or {}
    docs_jsonl = source_root / str(source_manifest.get("docs_jsonl") or "docs_selected.jsonl")
    if not docs_jsonl.is_file():
        raise FileNotFoundError(f"docs_selected.jsonl not found at {docs_jsonl}")
    docs_sidecar = maybe_load_json(docs_sidecar_path(docs_jsonl))
    docs_total = int((docs_sidecar or {}).get("num_docs") or source_manifest.get("num_docs") or count_docs(docs_jsonl))
    if args.num_val_docs >= 0:
        num_val_docs = int(args.num_val_docs)
    elif docs_sidecar is not None and docs_sidecar.get("docs_val") is not None:
        num_val_docs = int(docs_sidecar["docs_val"])
    else:
        num_val_docs = int(source_manifest.get("num_val_docs") or NUM_VAL_DOCS)

    tokenizer_ref = args.tokenizer_path
    vocab = tokenmonster.load(tokenizer_ref)
    vocab_size = int(vocab.vocab_size)
    bos_id = int(vocab.token_to_id("<s>"))
    eos_id = int(vocab.token_to_id("</s>"))
    if bos_id < 0 or eos_id < 0:
        raise ValueError("TokenMonster vocab must provide <s> and </s> tokens")

    tokenizer_src_path = Path(tokenizer_ref).expanduser()
    tokenizer_dst_path = tokenizers_dir / tokenizer_src_path.name
    if tokenizer_src_path.exists():
        maybe_link(tokenizer_src_path.resolve(), tokenizer_dst_path)
    else:
        vocab.save(str(tokenizer_dst_path))
    meta_path = tokenizers_dir / f"{tokenizer_dst_path.stem}.meta.npz"
    np.savez_compressed(
        meta_path,
        format_version=np.int64(1),
        tokenizer_kind=np.array("tokenmonster"),
        source_model_name=np.array(str(tokenizer_ref)),
        vocab_size=np.int64(vocab_size),
        base_bytes=np.asarray(
            [len(str(vocab.get_dictionary()[i]["token_decoded"]).encode("utf-8")) for i in range(vocab_size)],
            dtype=np.int16,
        ),
        has_leading_space=np.zeros((vocab_size,), dtype=np.bool_),
        is_boundary_token=np.zeros((vocab_size,), dtype=np.bool_),
    )

    docs_copy = output_root / "docs_selected.jsonl"
    maybe_link(docs_jsonl, docs_copy)
    sidecar_src = docs_sidecar_path(docs_jsonl)
    if sidecar_src.is_file():
        maybe_link(sidecar_src, output_root / sidecar_src.name)

    output_dir = datasets_dir / args.dataset_name
    print(f"Exporting dataset: {args.dataset_name}", flush=True)
    stats = export_shards(
        docs_jsonl,
        vocab=vocab,
        vocab_size=vocab_size,
        bos_id=bos_id,
        eos_id=eos_id,
        output_dir=output_dir,
        num_val_docs=num_val_docs,
        shard_size=int(args.chunk_tokens),
        docs_total=docs_total,
    )

    recommended_bigram_vocab_size = ((vocab_size + 127) // 128) * 128 * 5
    manifest = {
        "version": VERSION,
        "num_docs": docs_total,
        "num_val_docs": num_val_docs,
        "shuffle_seed": source_manifest.get("shuffle_seed"),
        "dataset_revision": source_manifest.get("dataset_revision"),
        "shard_size": int(args.chunk_tokens),
        "append_eos": APPEND_EOS,
        "docs_jsonl": str(docs_copy),
        "docs_meta": source_manifest.get("docs_meta") or {},
        "tokenizer_specs": [],
        "tokenizers": [
            {
                "name": args.tokenizer_name,
                "kind": "tokenmonster",
                "vocab_size": vocab_size,
                "bos_id": bos_id,
                "eos_id": eos_id,
                "recommended_bigram_vocab_size": recommended_bigram_vocab_size,
                "path": str(tokenizer_dst_path),
                "meta_path": str(meta_path),
                "source_spec": {
                    "name": args.tokenizer_name,
                    "kind": "tokenmonster",
                    "source_model": str(tokenizer_ref),
                },
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
                "bos_id": bos_id,
                "eos_id": eos_id,
                "recommended_bigram_vocab_size": recommended_bigram_vocab_size,
                "stats": stats,
            }
        ],
    }
    manifest = relativize_manifest_paths(manifest, output_root)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "dataset_name": args.dataset_name,
                "tokenizer_name": args.tokenizer_name,
                "vocab_size": vocab_size,
                "stats": stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
