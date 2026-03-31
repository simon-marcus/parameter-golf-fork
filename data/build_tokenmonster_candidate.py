from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tokenmonster_utils import load_tokenmonster_vocab

def is_regular(item: dict[str, object]) -> bool:
    return str(item.get("type", "")) == "regular"


def should_delete(
    item: dict[str, object],
    *,
    max_decoded_bytes: int,
    delete_regexes: list[re.Pattern[str]],
    delete_multiword: bool,
    delete_space_punct: bool,
) -> bool:
    decoded = str(item["token_decoded"])
    if len(decoded.encode("utf-8")) > max_decoded_bytes:
        return True
    if delete_multiword and decoded.count(" ") >= 2:
        return True
    if delete_space_punct and " " in decoded and any(not ch.isalnum() and not ch.isspace() for ch in decoded):
        return True
    for pattern in delete_regexes:
        if pattern.search(decoded):
            return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local TokenMonster candidate by pruning/editing a base vocab")
    parser.add_argument("--base-vocab", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--resize", type=int, default=0)
    parser.add_argument("--max-decoded-bytes", type=int, default=24)
    parser.add_argument("--delete-regex", action="append", default=[])
    parser.add_argument("--delete-multiword", action="store_true")
    parser.add_argument("--delete-space-punct", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    vocab = load_tokenmonster_vocab(args.base_vocab)
    before_size = int(vocab.vocab_size)
    delete_regexes = [re.compile(p) for p in args.delete_regex]
    dictionary = vocab.get_dictionary()
    delete_tokens: list[str] = []
    delete_ids: list[int] = []
    for item in dictionary.values():
        if not is_regular(item):
            continue
        if should_delete(
            item,
            max_decoded_bytes=args.max_decoded_bytes,
            delete_regexes=delete_regexes,
            delete_multiword=args.delete_multiword,
            delete_space_punct=args.delete_space_punct,
        ):
            delete_tokens.append(str(item["token_decoded"]))
            delete_ids.append(int(item["id"]))

    if not args.dry_run and delete_ids:
        for token_id in sorted(delete_ids, reverse=True):
            vocab.delete_token_by_id(int(token_id))
    resized_to = before_size
    if args.resize > 0 and not args.dry_run:
        resized_to = int(vocab.resize(args.resize, reset_token_ids=True))
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        vocab.save(str(output_path))
    summary = {
        "base_vocab": args.base_vocab,
        "output_path": str(output_path),
        "before_size": before_size,
        "after_delete_size": before_size - len(delete_ids) if args.dry_run else int(vocab.vocab_size),
        "resize_target": int(args.resize),
        "resized_to": resized_to,
        "deleted_count": len(delete_ids),
        "deleted_ids_sample": delete_ids[:20],
        "deleted_decoded_sample": delete_tokens[:20],
        "max_decoded_bytes": args.max_decoded_bytes,
        "delete_regex": args.delete_regex,
        "delete_multiword": bool(args.delete_multiword),
        "delete_space_punct": bool(args.delete_space_punct),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
