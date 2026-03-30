from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate_params(
    *,
    vocab_size: int,
    model_dim: int,
    tie_embeddings: bool,
) -> dict[str, int]:
    embedding_params = vocab_size * model_dim
    head_params = 0 if tie_embeddings else vocab_size * model_dim
    return {
        "embedding_params": embedding_params,
        "head_params": head_params,
        "total_vocab_linked_params": embedding_params + head_params,
    }


def build_rows(
    vocab_sizes: list[int],
    *,
    baseline_vocab: int,
    model_dim: int,
    tie_embeddings: bool,
) -> list[dict[str, object]]:
    baseline = estimate_params(vocab_size=baseline_vocab, model_dim=model_dim, tie_embeddings=tie_embeddings)
    rows = []
    for vocab_size in vocab_sizes:
        current = estimate_params(vocab_size=vocab_size, model_dim=model_dim, tie_embeddings=tie_embeddings)
        delta = current["total_vocab_linked_params"] - baseline["total_vocab_linked_params"]
        ratio = current["total_vocab_linked_params"] / max(baseline["total_vocab_linked_params"], 1)
        rows.append(
            {
                "vocab_size": vocab_size,
                "embedding_params": current["embedding_params"],
                "head_params": current["head_params"],
                "total_vocab_linked_params": current["total_vocab_linked_params"],
                "delta_vs_baseline": delta,
                "ratio_vs_baseline": ratio,
            }
        )
    return rows


def build_markdown(summary: dict) -> str:
    lines = [
        "# Tokenizer Tradeoff Report",
        "",
        f"- model_dim: `{summary['model_dim']}`",
        f"- tie_embeddings: `{summary['tie_embeddings']}`",
        f"- baseline_vocab: `{summary['baseline_vocab']}`",
        "",
        "| vocab | embed params | head params | total | delta vs baseline | ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['vocab_size']} | {row['embedding_params']} | {row['head_params']} | "
            f"{row['total_vocab_linked_params']} | {row['delta_vs_baseline']} | {row['ratio_vs_baseline']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate embedding/head budget across tokenizer vocab sizes")
    parser.add_argument("--vocab-sizes", default="256,384,512,768,1024,1536,2048,4096,8192,28416")
    parser.add_argument("--baseline-vocab", type=int, default=1024)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--tie-embeddings", type=int, default=1)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    vocab_sizes = [int(x) for x in args.vocab_sizes.split(",") if x.strip()]
    summary = {
        "baseline_vocab": args.baseline_vocab,
        "model_dim": args.model_dim,
        "tie_embeddings": bool(args.tie_embeddings),
        "rows": build_rows(
            vocab_sizes,
            baseline_vocab=args.baseline_vocab,
            model_dim=args.model_dim,
            tie_embeddings=bool(args.tie_embeddings),
        ),
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text + "\n", encoding="utf-8")
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
