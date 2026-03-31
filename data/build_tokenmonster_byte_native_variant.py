from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tokenmonster_utils import load_tokenmonster_vocab

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a byte-native TokenMonster YAML variant from an existing vocab/YAML export")
    parser.add_argument("--base-vocab", required=True, help="Source TokenMonster vocab path or ref")
    parser.add_argument("--output-path", required=True, help="Where to save the derived YAML spec")
    parser.add_argument("--charset", default="none", choices=("none", "utf-8", "utf8", "latin-1", "latin1"))
    parser.add_argument("--normalization", default="none", help='Normalization setting, typically "none"')
    parser.add_argument("--capcode", type=int, choices=(0, 1, 2), default=None, help="Optional capcode override")
    parser.add_argument(
        "--delete-raw-regex",
        action="append",
        default=[],
        help="Delete raw tokens whose encoded token string matches this regex",
    )
    parser.add_argument(
        "--delete-decoded-regex",
        action="append",
        default=[],
        help="Delete decoded tokens whose decoded form matches this regex",
    )
    parser.add_argument(
        "--add-missing-high-byte-tokens",
        action="store_true",
        help="Append TokenMonsterHexEncode{XX} entries for any missing bytes in 0x80..0xFF",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import tokenmonster
    except ImportError as exc:
        raise RuntimeError("tokenmonster is required for build_tokenmonster_byte_native_variant.py") from exc

    base_vocab = load_tokenmonster_vocab(args.base_vocab)
    yaml_text = base_vocab.export_yaml().decode("utf-8")
    yaml_text = re.sub(r"^charset: .*$", f"charset: {args.charset}", yaml_text, flags=re.M)
    yaml_text = re.sub(r'^normalization: .*$', f'normalization: {args.normalization}', yaml_text, flags=re.M)
    if args.capcode is not None:
        yaml_text = re.sub(r"^capcode: .*$", f"capcode: {args.capcode}", yaml_text, flags=re.M)
    derived = tokenmonster.new(yaml_text)

    raw_patterns = [re.compile(p) for p in args.delete_raw_regex]
    decoded_patterns = [re.compile(p) for p in args.delete_decoded_regex]
    delete_ids: list[int] = []
    if raw_patterns or decoded_patterns:
        dictionary = derived.get_dictionary()
        for item in dictionary.values():
            token = str(item["token"])
            decoded = str(item["token_decoded"])
            if any(p.search(token) for p in raw_patterns) or any(p.search(decoded) for p in decoded_patterns):
                delete_ids.append(int(item["id"]))
        for token_id in sorted(delete_ids, reverse=True):
            derived.delete_token_by_id(token_id)

    if args.add_missing_high_byte_tokens:
        yaml_lines = derived.export_yaml().decode("utf-8").splitlines()
        existing_hex: set[str] = set()
        max_id = -1
        for line in yaml_lines:
            match = re.search(r"TokenMonsterHexEncode\{([0-9A-Fa-f]{2})\}", line)
            if match:
                existing_hex.add(match.group(1).lower())
            id_match = re.search(r"\bid:\s+(\d+)", line)
            if id_match:
                max_id = max(max_id, int(id_match.group(1)))
        missing_hex = [f"{byte:02x}" for byte in range(0x80, 0x100) if f"{byte:02x}" not in existing_hex]
        if missing_hex:
            insert_at = max(i for i, line in enumerate(yaml_lines) if "TokenMonsterHexEncode{" in line) + 4
            appended: list[str] = []
            next_id = max_id
            for hx in missing_hex:
                next_id += 1
                appended.extend(
                    [
                        f"    - token:   \"TokenMonsterHexEncode{{{hx}}}\"",
                        f"      id:      {next_id}",
                        "      score:   0.00001",
                        "      encoded: true",
                    ]
                )
            yaml_text = "\n".join(yaml_lines[:insert_at] + appended + yaml_lines[insert_at:]) + "\n"
            derived = tokenmonster.new(yaml_text)
        else:
            missing_hex = []
    else:
        missing_hex = []

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = derived.export_yaml()
    output_path.write_bytes(output_bytes)

    summary = {
        "base_vocab": args.base_vocab,
        "output_path": str(output_path),
        "charset": args.charset,
        "normalization": str(derived.normalization()),
        "capcode": int(derived.capcode()),
        "requested_capcode": args.capcode,
        "vocab_size": int(derived.vocab_size),
        "deleted_count": len(delete_ids),
        "deleted_ids_sample": delete_ids[:20],
        "delete_raw_regex": args.delete_raw_regex,
        "delete_decoded_regex": args.delete_decoded_regex,
        "add_missing_high_byte_tokens": bool(args.add_missing_high_byte_tokens),
        "missing_high_byte_tokens_added": len(missing_hex),
        "missing_high_byte_values_sample": missing_hex[:20],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
