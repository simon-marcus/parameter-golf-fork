from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HEADER_BYTES = 256 * np.dtype("<i4").itemsize
TOKEN_DTYPE = np.dtype("<u2")
SHARD_MAGIC = 20240520
SHARD_VERSION = 1


def load_data_shard(file: Path) -> np.ndarray:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = HEADER_BYTES + num_tokens * TOKEN_DTYPE.itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens = np.fromfile(file, dtype=TOKEN_DTYPE, count=num_tokens, offset=HEADER_BYTES)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return tokens.astype(np.uint16, copy=False)


def select_window(
    shard: np.ndarray,
    *,
    take_tokens: int,
    shard_index: int,
    total_shards: int,
    offset_mode: str,
) -> np.ndarray:
    take_tokens = min(max(int(take_tokens), 0), int(shard.size))
    if take_tokens <= 0:
        return shard[:0]
    if take_tokens >= shard.size:
        return shard
    available = int(shard.size) - take_tokens
    if offset_mode == "start":
        start = 0
    elif offset_mode == "center":
        start = available // 2
    elif offset_mode == "staggered":
        start = round(available * ((shard_index + 1) / (total_shards + 1)))
    else:
        raise ValueError(f"Unsupported offset mode: {offset_mode}")
    end = start + take_tokens
    return shard[start:end]


def iter_token_windows(
    files: list[Path],
    *,
    max_tokens: int,
    max_shards: int,
    tokens_per_shard: int,
    offset_mode: str,
):
    selected_files = files if max_shards <= 0 else files[:max_shards]
    if tokens_per_shard > 0:
        total_shards = len(selected_files)
        for shard_index, file in enumerate(selected_files):
            shard = load_data_shard(file)
            window = select_window(
                shard,
                take_tokens=tokens_per_shard,
                shard_index=shard_index,
                total_shards=total_shards,
                offset_mode=offset_mode,
            )
            if window.size:
                yield file, window
        return

    remaining = max_tokens
    for file in selected_files:
        if remaining <= 0:
            break
        shard = load_data_shard(file)
        if shard.size > remaining:
            shard = shard[:remaining]
        yield file, shard
        remaining -= int(shard.size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode tokenized FineWeb shards into a text sample")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--tokens-per-shard", type=int, default=0)
    parser.add_argument("--offset-mode", choices=("start", "center", "staggered"), default="start")
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required for extract_text_sample_from_shards.py") from exc

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    files = sorted(dataset_dir.glob(f"fineweb_{args.split}_*.bin"))
    if not files:
        raise FileNotFoundError(f"No shard files for split={args.split} under {dataset_dir}")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    chunk_tokens = max(1, int(args.chunk_tokens))
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    chunk_index = 0
    total_source_shards = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for file, shard in iter_token_windows(
            files,
            max_tokens=int(args.max_tokens),
            max_shards=int(args.max_shards),
            tokens_per_shard=int(args.tokens_per_shard),
            offset_mode=str(args.offset_mode),
        ):
            total_source_shards += 1
            for start in range(0, shard.size, chunk_tokens):
                piece = shard[start : start + chunk_tokens]
                if piece.size == 0:
                    continue
                text = sp.decode(piece.tolist())
                payload = {
                    "chunk_index": chunk_index,
                    "source_shard": file.name,
                    "token_count": int(piece.size),
                    "text": text,
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                chunk_index += 1
                written += int(piece.size)
                if int(args.tokens_per_shard) <= 0 and written >= args.max_tokens:
                    break
            if int(args.tokens_per_shard) <= 0 and written >= args.max_tokens:
                break
    print(
        json.dumps(
            {
                "output": str(out_path),
                "chunks": chunk_index,
                "written_tokens": written,
                "source_shards": total_source_shards,
                "offset_mode": args.offset_mode,
                "tokens_per_shard": int(args.tokens_per_shard),
                "max_shards": int(args.max_shards),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
