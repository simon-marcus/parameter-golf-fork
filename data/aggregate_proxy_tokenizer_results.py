from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate proxy tokenizer summary files")
    parser.add_argument("--summary", action="append", required=True, help="Path to a proxy summary.json")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def build_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Proxy Tokenizer Aggregate",
        "",
        f"- runs: `{summary['num_runs']}`",
        f"- candidates: `{len(summary['rows'])}`",
        "",
        "| rank | name | family | vocab | n | mean postquant bpb | std | mean artifact bytes | mean params |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(summary["rows"], start=1):
        lines.append(
            f"| {i} | {row['name']} | {row['family']} | {row['vocab_size']} | {row['n']} | "
            f"{row['postquant_val_bpb_mean']:.6f} | {row['postquant_val_bpb_std']:.6f} | "
            f"{row['artifact_bytes_mean']:.0f} | {row['model_params_mean']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    source_runs: list[str] = []
    for summary_path_str in args.summary:
        summary_path = Path(summary_path_str).expanduser().resolve()
        source_runs.append(str(summary_path))
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in rows:
            groups[(str(row["family"]), str(row["name"]))].append(row)

    out_rows: list[dict[str, object]] = []
    for (family, name), rows in groups.items():
        post_vals = [float(row["postquant_val_bpb"]) for row in rows if "postquant_val_bpb" in row]
        art_vals = [float(row["artifact_bytes"]) for row in rows if "artifact_bytes" in row]
        param_vals = [float(row["model_params"]) for row in rows if "model_params" in row]
        pre_vals = [float(row["prequant_val_bpb"]) for row in rows if "prequant_val_bpb" in row]
        post_mean, post_std = mean_std(post_vals)
        art_mean, _ = mean_std(art_vals)
        param_mean, _ = mean_std(param_vals)
        pre_mean, pre_std = mean_std(pre_vals)
        row0 = rows[0]
        out_rows.append(
            {
                "name": name,
                "family": family,
                "vocab_size": int(row0["vocab_size"]),
                "n": len(rows),
                "postquant_val_bpb_mean": post_mean,
                "postquant_val_bpb_std": post_std,
                "prequant_val_bpb_mean": pre_mean,
                "prequant_val_bpb_std": pre_std,
                "artifact_bytes_mean": art_mean,
                "model_params_mean": param_mean,
                "tokenizer_path": row0["tokenizer_path"],
            }
        )
    out_rows.sort(key=lambda row: (row["postquant_val_bpb_mean"], row["postquant_val_bpb_std"]))
    summary = {
        "num_runs": len(source_runs),
        "source_runs": source_runs,
        "rows": out_rows,
    }
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
