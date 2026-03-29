from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_cmd_json(cmd: list[str]) -> dict[str, object]:
    import subprocess

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def estimate_vocab_linked_params(vocab_size: int, *, model_dim: int, tie_embeddings: bool) -> int:
    return int(vocab_size) * int(model_dim) * (1 if tie_embeddings else 2)


def enrich_score(
    metrics: dict[str, object],
    *,
    baseline_vocab_size: int,
    model_dim: int,
    tie_embeddings: bool,
    artifact_budget_bytes: int,
    vocab_artifact_bytes_per_param: float,
    dead_vocab_weight: float,
    artifact_weight: float,
) -> dict[str, object]:
    enriched = dict(metrics)
    vocab_size = int(enriched["vocab_size"])
    vocab_linked_params = estimate_vocab_linked_params(vocab_size, model_dim=model_dim, tie_embeddings=tie_embeddings)
    baseline_params = estimate_vocab_linked_params(
        baseline_vocab_size, model_dim=model_dim, tie_embeddings=tie_embeddings
    )
    vocab_artifact_bytes = vocab_linked_params * vocab_artifact_bytes_per_param
    enriched["estimated_vocab_linked_params"] = vocab_linked_params
    enriched["estimated_vocab_artifact_bytes"] = vocab_artifact_bytes
    enriched["estimated_artifact_budget_frac"] = vocab_artifact_bytes / max(artifact_budget_bytes, 1)
    enriched["estimated_vocab_linked_params_delta"] = vocab_linked_params - baseline_params
    enriched["estimated_vocab_artifact_bytes_delta"] = vocab_artifact_bytes - (
        baseline_params * vocab_artifact_bytes_per_param
    )
    enriched["budget_aware_score"] = (
        float(enriched["tokens_per_byte"])
        + float(enriched["dead_vocab_frac"]) * dead_vocab_weight
        + float(enriched["estimated_artifact_budget_frac"]) * artifact_weight
    )
    return enriched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare SentencePiece and TokenMonster tokenizers on one text sample")
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--sentencepiece-model", action="append", default=[])
    parser.add_argument("--tokenmonster-vocab", action="append", default=[])
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--baseline-vocab-size", type=int, default=1024)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--tie-embeddings", type=int, default=1)
    parser.add_argument("--artifact-budget-bytes", type=int, default=16_000_000)
    parser.add_argument("--vocab-artifact-bytes-per-param", type=float, default=0.278)
    parser.add_argument("--dead-vocab-weight", type=float, default=0.02)
    parser.add_argument("--artifact-weight", type=float, default=1.0)
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent.parent
    eval_script = root / "data" / "evaluate_tokenizer_on_text_sample.py"
    sample_path = Path(args.sample_path).expanduser().resolve()
    rows: list[dict[str, object]] = []

    for model_path in args.sentencepiece_model:
        raw = run_cmd_json(
            [
                __import__("sys").executable,
                str(eval_script),
                "--family",
                "sentencepiece",
                "--model-path",
                str(Path(model_path).expanduser().resolve()),
                "--sample-path",
                str(sample_path),
                "--max-chunks",
                str(args.max_chunks),
            ]
        )
        raw["name"] = Path(model_path).stem
        rows.append(
            enrich_score(
                raw,
                baseline_vocab_size=args.baseline_vocab_size,
                model_dim=args.model_dim,
                tie_embeddings=bool(args.tie_embeddings),
                artifact_budget_bytes=args.artifact_budget_bytes,
                vocab_artifact_bytes_per_param=args.vocab_artifact_bytes_per_param,
                dead_vocab_weight=args.dead_vocab_weight,
                artifact_weight=args.artifact_weight,
            )
        )

    for vocab_ref in args.tokenmonster_vocab:
        raw = run_cmd_json(
            [
                __import__("sys").executable,
                str(eval_script),
                "--family",
                "tokenmonster",
                "--model-path",
                vocab_ref,
                "--sample-path",
                str(sample_path),
                "--max-chunks",
                str(args.max_chunks),
            ]
        )
        raw["name"] = vocab_ref
        rows.append(
            enrich_score(
                raw,
                baseline_vocab_size=args.baseline_vocab_size,
                model_dim=args.model_dim,
                tie_embeddings=bool(args.tie_embeddings),
                artifact_budget_bytes=args.artifact_budget_bytes,
                vocab_artifact_bytes_per_param=args.vocab_artifact_bytes_per_param,
                dead_vocab_weight=args.dead_vocab_weight,
                artifact_weight=args.artifact_weight,
            )
        )

    rows.sort(key=lambda row: float(row["budget_aware_score"]))
    summary = {
        "sample_path": str(sample_path),
        "baseline_vocab_size": args.baseline_vocab_size,
        "model_dim": args.model_dim,
        "tie_embeddings": bool(args.tie_embeddings),
        "artifact_budget_bytes": args.artifact_budget_bytes,
        "vocab_artifact_bytes_per_param": args.vocab_artifact_bytes_per_param,
        "dead_vocab_weight": args.dead_vocab_weight,
        "artifact_weight": args.artifact_weight,
        "rows": rows,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
