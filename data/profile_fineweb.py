from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HEADER_BYTES = 256 * np.dtype("<i4").itemsize
TOKEN_DTYPE = np.dtype("<u2")
SHARD_MAGIC = 20240520
SHARD_VERSION = 1

URLISH_RE = re.compile(r"(https?://|www\.|\.com|\.org|\.net|/@|://)")
NUMERIC_RE = re.compile(r"^[0-9][0-9,.\-/%:]*$")
ALPHA_RE = re.compile(r"^[A-Za-z]+$")
ALNUM_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
PUNCT_RE = re.compile(r"^[^\w\s]+$")
MARKUP_RE = re.compile(r"[<>{}\[\]=/]|&[a-z]+;")


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


def collect_tokens(files: list[Path], max_tokens: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    remaining = max_tokens
    for file in files:
        if remaining <= 0:
            break
        shard = load_data_shard(file)
        if shard.size > remaining:
            shard = shard[:remaining]
        parts.append(shard)
        remaining -= int(shard.size)
    if not parts:
        raise ValueError("No tokens collected; check dataset path or token limits")
    return np.concatenate(parts)


def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def load_sentencepiece_tables(tokenizer_path: Path) -> tuple[object, list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("sentencepiece is required for profile_fineweb.py") from exc
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    vocab_size = int(sp.vocab_size())
    token_rows: list[dict[str, object]] = []
    base_bytes = np.zeros((vocab_size,), dtype=np.int16)
    has_leading_space = np.zeros((vocab_size,), dtype=np.bool_)
    is_boundary = np.ones((vocab_size,), dtype=np.bool_)
    for token_id in range(vocab_size):
        row = {
            "id": token_id,
            "piece": sp.id_to_piece(token_id),
            "is_control": bool(sp.is_control(token_id)),
            "is_unknown": bool(sp.is_unknown(token_id)),
            "is_unused": bool(sp.is_unused(token_id)),
            "is_byte": bool(sp.is_byte(token_id)),
        }
        piece = row["piece"]
        clean = piece
        if row["is_control"] or row["is_unknown"] or row["is_unused"]:
            row["clean_piece"] = ""
            row["class"] = "special"
            token_rows.append(row)
            continue
        is_boundary[token_id] = False
        if row["is_byte"]:
            base_bytes[token_id] = 1
            row["clean_piece"] = f"<byte:{token_id}>"
            row["class"] = "byte"
            token_rows.append(row)
            continue
        if clean.startswith("▁"):
            clean = clean[1:]
            has_leading_space[token_id] = True
        row["clean_piece"] = clean
        base_bytes[token_id] = len(clean.encode("utf-8"))
        row["class"] = classify_piece(clean, bool(has_leading_space[token_id]))
        token_rows.append(row)
    return sp, token_rows, base_bytes, has_leading_space, is_boundary


def classify_piece(clean_piece: str, has_leading_space: bool) -> str:
    if clean_piece == "":
        return "whitespace" if has_leading_space else "empty"
    if "\n" in clean_piece or "\r" in clean_piece:
        return "newline"
    if URLISH_RE.search(clean_piece):
        return "urlish"
    if MARKUP_RE.search(clean_piece):
        return "markupish"
    if NUMERIC_RE.fullmatch(clean_piece):
        return "numeric"
    if ALPHA_RE.fullmatch(clean_piece):
        return "alpha"
    if ALNUM_RE.fullmatch(clean_piece):
        return "alnum"
    if PUNCT_RE.fullmatch(clean_piece):
        return "punct"
    return "other"


def stream_byte_counts(
    tokens: np.ndarray,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary: np.ndarray,
) -> np.ndarray:
    out = np.zeros(tokens.shape[0], dtype=np.int32)
    prev_boundary = True
    for i, tok in enumerate(tokens.tolist()):
        token_bytes = int(base_bytes[tok])
        if bool(has_leading_space[tok]) and not prev_boundary:
            token_bytes += 1
        out[i] = token_bytes
        prev_boundary = bool(is_boundary[tok])
    return out


def summarize_token_classes(
    tokens: np.ndarray,
    byte_counts: np.ndarray,
    token_rows: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    by_class: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "bytes": 0.0, "unique_tokens": 0.0})
    seen_ids: dict[str, set[int]] = defaultdict(set)
    for tok, token_bytes in zip(tokens.tolist(), byte_counts.tolist()):
        klass = str(token_rows[tok]["class"])
        by_class[klass]["tokens"] += 1
        by_class[klass]["bytes"] += float(token_bytes)
        seen_ids[klass].add(int(tok))
    for klass, token_ids in seen_ids.items():
        by_class[klass]["unique_tokens"] = float(len(token_ids))
    total_tokens = float(len(tokens))
    total_bytes = float(byte_counts.sum())
    out: dict[str, dict[str, float]] = {}
    for klass, stats in sorted(by_class.items(), key=lambda kv: (-kv[1]["bytes"], kv[0])):
        out[klass] = {
            "tokens": stats["tokens"],
            "bytes": stats["bytes"],
            "token_frac": stats["tokens"] / max(total_tokens, 1.0),
            "byte_frac": stats["bytes"] / max(total_bytes, 1.0),
            "avg_bytes_per_token": stats["bytes"] / max(stats["tokens"], 1.0),
            "unique_tokens": stats["unique_tokens"],
        }
    return out


def top_token_tables(
    tokens: np.ndarray,
    byte_counts: np.ndarray,
    token_rows: list[dict[str, object]],
    *,
    top_k: int,
) -> dict[str, list[dict[str, object]]]:
    freq = Counter(tokens.tolist())
    bytes_by_tok: dict[int, int] = defaultdict(int)
    for tok, token_bytes in zip(tokens.tolist(), byte_counts.tolist()):
        bytes_by_tok[int(tok)] += int(token_bytes)

    def build_rows(items: list[tuple[int, int]], value_key: str) -> list[dict[str, object]]:
        rows = []
        for tok, value in items[:top_k]:
            meta = token_rows[tok]
            rows.append(
                {
                    "id": tok,
                    "piece": meta["piece"],
                    "clean_piece": meta["clean_piece"],
                    "class": meta["class"],
                    value_key: int(value),
                }
            )
        return rows

    by_count = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    by_bytes = sorted(bytes_by_tok.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "top_by_count": build_rows(by_count, "count"),
        "top_by_bytes": build_rows(by_bytes, "bytes"),
    }


def repetition_profile(tokens: np.ndarray, context_lengths: list[int], max_positions: int) -> dict[str, dict[str, float]]:
    usable = min(int(tokens.shape[0]) - 1, max_positions)
    if usable <= max(context_lengths):
        return {}
    profile: dict[str, dict[str, float]] = {}
    for n in context_lengths:
        contexts: dict[tuple[int, ...], Counter[int]] = {}
        repeated = 0
        deterministic_mass = 0.0
        entropy_sum = 0.0
        support_sum = 0.0
        seen = 0
        for i in range(n, usable):
            ctx = tuple(int(x) for x in tokens[i - n : i])
            nxt = int(tokens[i])
            counts = contexts.get(ctx)
            if counts:
                total = sum(counts.values())
                max_prob = max(counts.values()) / total
                repeated += 1
                deterministic_mass += max_prob
                support_sum += total
                entropy = 0.0
                for c in counts.values():
                    p = c / total
                    entropy -= p * math.log2(p)
                entropy_sum += entropy
            else:
                contexts[ctx] = Counter()
            contexts[ctx][nxt] += 1
            seen += 1
        if seen == 0:
            continue
        repeat_frac = repeated / seen
        avg_max_prob = deterministic_mass / max(repeated, 1)
        avg_entropy = entropy_sum / max(repeated, 1)
        avg_support = support_sum / max(repeated, 1)
        profile[str(n)] = {
            "positions": seen,
            "repeat_context_frac": repeat_frac,
            "avg_repeat_max_next_prob": avg_max_prob,
            "avg_repeat_entropy_bits": avg_entropy,
            "avg_repeat_context_support": avg_support,
        }
    return profile


def build_markdown_report(summary: dict) -> str:
    lines = [
        f"# FineWeb Profile: {summary['variant']}",
        "",
        "## Overview",
        f"- tokenizer: `{summary['tokenizer_path']}`",
        f"- train tokens sampled: `{summary['train']['token_count']}`",
        f"- val tokens sampled: `{summary['val']['token_count']}`",
        f"- vocab size: `{summary['vocab_size']}`",
        "",
        "## Split Metrics",
    ]
    for split in ("train", "val"):
        stats = summary[split]
        lines.extend(
            [
                f"- `{split}` bytes/token: `{stats['avg_bytes_per_token']:.4f}`",
                f"- `{split}` unique tokens used: `{stats['unique_tokens_observed']}`",
                f"- `{split}` top class by byte mass: `{next(iter(stats['class_stats'].keys()), 'n/a')}`",
            ]
        )
    lines.extend(["", "## Repetition Profile (val)"])
    if not summary["val"]["repetition_profile"]:
        lines.append("- insufficient sample")
    else:
        for n, stats in summary["val"]["repetition_profile"].items():
            lines.append(
                f"- context `{n}`: repeat_frac=`{stats['repeat_context_frac']:.3f}`, "
                f"avg_max_next_prob=`{stats['avg_repeat_max_next_prob']:.3f}`, "
                f"avg_entropy=`{stats['avg_repeat_entropy_bits']:.3f}`"
            )
    lines.extend(["", "## Leading Classes By Byte Mass"])
    for split in ("train", "val"):
        lines.append(f"- `{split}`:")
        for klass, stats in list(summary[split]["class_stats"].items())[:6]:
            lines.append(
                f"  - `{klass}` byte_frac=`{stats['byte_frac']:.3f}` token_frac=`{stats['token_frac']:.3f}` "
                f"avg_bytes/token=`{stats['avg_bytes_per_token']:.2f}`"
            )
    return "\n".join(lines) + "\n"


def summarize_split(
    name: str,
    tokens: np.ndarray,
    byte_counts: np.ndarray,
    token_rows: list[dict[str, object]],
    repetition_lengths: list[int],
    repetition_tokens: int,
    top_k: int,
) -> dict[str, object]:
    return {
        "split": name,
        "token_count": int(tokens.shape[0]),
        "byte_count": int(byte_counts.sum()),
        "avg_bytes_per_token": float(byte_counts.mean()),
        "unique_tokens_observed": int(len(np.unique(tokens))),
        "class_stats": summarize_token_classes(tokens, byte_counts, token_rows),
        "top_tokens": top_token_tables(tokens, byte_counts, token_rows, top_k=top_k),
        "repetition_profile": repetition_profile(tokens, repetition_lengths, repetition_tokens),
    }


def resolve_tokenizer_path(args: argparse.Namespace) -> Path:
    if args.tokenizer_path:
        return Path(args.tokenizer_path).expanduser().resolve()
    default = Path("data/tokenizers/fineweb_1024_bpe.model")
    return default.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile FineWeb token structure and repetition")
    parser.add_argument("--variant", default="sp1024")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--train-tokens", type=int, default=1_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--repetition-contexts", default="1,2,4,8,16,32")
    parser.add_argument("--repetition-tokens", type=int, default=200_000)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path("data/datasets") / dataset_dir_for_variant(args.variant)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Download it with `python3 data/cached_challenge_fineweb.py --variant sp1024`."
        )
    tokenizer_path = resolve_tokenizer_path(args)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer model not found: {tokenizer_path}. "
            "Download it with `python3 data/cached_challenge_fineweb.py --variant sp1024`."
        )

    _, token_rows, base_bytes, has_leading_space, is_boundary = load_sentencepiece_tables(tokenizer_path)
    train_files = sorted(dataset_dir.glob("fineweb_train_*.bin"))
    val_files = sorted(dataset_dir.glob("fineweb_val_*.bin"))
    if not train_files or not val_files:
        raise FileNotFoundError(f"Expected train/val shard files under {dataset_dir}")

    train_tokens = collect_tokens(train_files, args.train_tokens)
    val_tokens = collect_tokens(val_files, args.val_tokens)
    train_bytes = stream_byte_counts(train_tokens, base_bytes, has_leading_space, is_boundary)
    val_bytes = stream_byte_counts(val_tokens, base_bytes, has_leading_space, is_boundary)
    repetition_lengths = [int(x) for x in args.repetition_contexts.split(",") if x.strip()]

    summary = {
        "variant": args.variant,
        "dataset_dir": str(dataset_dir.resolve()),
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": int(len(token_rows)),
        "train": summarize_split(
            "train", train_tokens, train_bytes, token_rows, repetition_lengths, args.repetition_tokens, args.top_k
        ),
        "val": summarize_split(
            "val", val_tokens, val_bytes, token_rows, repetition_lengths, args.repetition_tokens, args.top_k
        ),
    }

    summary_json = json.dumps(summary, indent=2)
    print(summary_json)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(summary_json + "\n", encoding="utf-8")
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_markdown_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
