from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_history_candidates(history_path: Path, top_k: int) -> list[dict[str, object]]:
    if not history_path.exists():
        return []
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [row for row in rows if row.get("kept")]
    kept.sort(key=lambda row: float(row.get("score", 1e9)))
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for row in kept:
        model_path = str(row["metrics"]["model_path"])
        if model_path in seen:
            continue
        seen.add(model_path)
        out.append(
            {
                "name": f"sp_exp_{int(row['experiment_id']):04d}",
                "family": "sentencepiece",
                "model_path": model_path,
                "history_score": float(row.get("score", 0.0)),
                "history_experiment_id": int(row["experiment_id"]),
            }
        )
        if len(out) >= top_k:
            break
    return out


def build_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Tokenizer Candidate Report",
        "",
        f"- sample_path: `{summary['sample_path']}`",
        f"- rows: `{len(summary['rows'])}`",
        "",
        "| rank | name | family | vocab | score | tpb | dead | artifact frac | source |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(summary["rows"], start=1):
        lines.append(
            f"| {i} | {row.get('name')} | {row.get('family')} | {row.get('vocab_size')} | "
            f"{float(row.get('budget_aware_score', 0.0)):.6f} | {float(row.get('tokens_per_byte', 0.0)):.6f} | "
            f"{100.0 * float(row.get('dead_vocab_frac', 0.0)):.2f}% | "
            f"{100.0 * float(row.get('estimated_artifact_budget_frac', 0.0)):.2f}% | {row.get('source')} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a unified tokenizer candidate leaderboard")
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--baseline-sentencepiece-model", required=True)
    parser.add_argument("--history-path", default=str(ROOT / "autoresearch" / "tokenizer_discovery" / "history.jsonl"))
    parser.add_argument("--top-k-sentencepiece", type=int, default=3)
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
    parser.add_argument("--output-md", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    history_path = Path(args.history_path).expanduser().resolve()
    compare_script = ROOT / "data" / "compare_tokenizer_backends.py"
    sentencepiece_candidates = parse_history_candidates(history_path, args.top_k_sentencepiece)

    sentencepiece_models = [str(Path(args.baseline_sentencepiece_model).expanduser().resolve())]
    sentencepiece_models.extend(candidate["model_path"] for candidate in sentencepiece_candidates)

    cmd = [
        sys.executable,
        str(compare_script),
        "--sample-path",
        str(Path(args.sample_path).expanduser().resolve()),
        "--max-chunks",
        str(args.max_chunks),
        "--baseline-vocab-size",
        str(args.baseline_vocab_size),
        "--model-dim",
        str(args.model_dim),
        "--tie-embeddings",
        str(args.tie_embeddings),
        "--artifact-budget-bytes",
        str(args.artifact_budget_bytes),
        "--vocab-artifact-bytes-per-param",
        str(args.vocab_artifact_bytes_per_param),
        "--dead-vocab-weight",
        str(args.dead_vocab_weight),
        "--artifact-weight",
        str(args.artifact_weight),
    ]
    for model in sentencepiece_models:
        cmd.extend(["--sentencepiece-model", model])
    for vocab in args.tokenmonster_vocab:
        cmd.extend(["--tokenmonster-vocab", vocab])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    summary = json.loads(result.stdout)

    source_map = {
        str(Path(args.baseline_sentencepiece_model).expanduser().resolve()): {
            "name": "sentencepiece_baseline",
            "source": "baseline_sentencepiece",
        }
    }
    for candidate in sentencepiece_candidates:
        source_map[candidate["model_path"]] = {
            "name": candidate["name"],
            "source": f"sentencepiece_history_{candidate['history_experiment_id']:04d}",
            "history_score": candidate["history_score"],
        }

    for row in summary["rows"]:
        if row["family"] == "sentencepiece":
            metadata = source_map.get(str(row["model_path"]), {})
            row["name"] = metadata.get("name", row.get("name"))
            row["source"] = metadata.get("source", "sentencepiece")
            if "history_score" in metadata:
                row["history_score"] = metadata["history_score"]
        else:
            row["source"] = "tokenmonster_sidecar"

    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_json:
        out_json = Path(args.output_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text + "\n", encoding="utf-8")
    if args.output_md:
        out_md = Path(args.output_md).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
