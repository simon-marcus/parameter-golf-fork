#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "autoresearch" / "tokenizer_discovery"


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def tail_lines(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-count:]


def format_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_pct(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def detect_active_status(log_tail: list[str]) -> str:
    for line in reversed(log_tail):
        if "review:start" in line:
            return "reviewing"
        if "experiment:" in line and ":start" in line:
            return "training/evaluating"
        if "run:done" in line:
            return "idle"
        if "run:start" in line:
            return "starting"
    return "unknown"


def render_dashboard(base_dir: Path, *, tail_count: int, history_count: int) -> str:
    frontier = read_json(base_dir / "frontier.json") or {}
    manifest = read_json(base_dir / "run_manifest.json") or {}
    latest_review = (base_dir / "latest_review.md").read_text(encoding="utf-8") if (base_dir / "latest_review.md").exists() else ""
    history = read_history(base_dir / "history.jsonl")
    log_tail = tail_lines(base_dir / "run.log", tail_count)

    best_metrics = frontier.get("best_metrics", {})
    latest = history[-1] if history else None
    latest_metrics = (latest or {}).get("metrics", {})
    latest_decision = (latest or {}).get("decision", {})

    lines: list[str] = []
    lines.append("Tokenizer Autoresearch")
    lines.append("=" * 80)
    lines.append(
        f"Status: {detect_active_status(log_tail)}"
        f" | Completed: {len(history)}"
        f" | Best: {frontier.get('best_label', '-')}"
        f" | Best score: {format_float(frontier.get('best_score'))}"
    )
    lines.append(
        f"Best tpb: {format_float(best_metrics.get('tokens_per_byte'))}"
        f" | Best dead: {format_pct(best_metrics.get('dead_vocab_frac'))}"
        f" | Best vocab: {best_metrics.get('vocab_size', '-')}"
    )
    lines.append(
        f"Strategy: exp {frontier.get('strategy_experiment_id', '-')}"
        f" | score {format_float(frontier.get('strategy_score'))}"
        f" | hash {manifest.get('config_hash', '-')}"
    )
    lines.append(f"Headline: {frontier.get('headline', '-')}")
    lines.append(f"Next: {frontier.get('next_direction', '-')}")
    lines.append("")

    lines.append("Latest Experiment")
    lines.append("-" * 80)
    if latest is None:
        lines.append("No experiments recorded yet.")
    else:
        params = latest.get("params", {})
        lines.append(
            f"Exp {latest.get('experiment_id', '-')}: score {format_float(latest.get('score'))}"
            f" | keep {bool(latest_decision.get('keep'))}"
            f" | tpb {format_float(latest_metrics.get('tokens_per_byte'))}"
            f" | dead {format_pct(latest_metrics.get('dead_vocab_frac'))}"
        )
        lines.append(
            f"Params: vocab={params.get('vocab_size', '-')}"
            f" model={params.get('model_type', '-')}"
            f" cc={params.get('character_coverage', '-')}"
            f" iss={params.get('input_sentence_size', '-')}"
            f" chunks={params.get('max_chunks', '-')}"
        )
        lines.append(f"Decision: {latest_decision.get('headline', '-')}")
    lines.append("")

    lines.append(f"Recent Experiments ({min(history_count, len(history))})")
    lines.append("-" * 80)
    if not history:
        lines.append("No history yet.")
    else:
        for row in history[-history_count:]:
            metrics = row.get("metrics", {})
            decision = row.get("decision", {})
            params = row.get("params", {})
            lines.append(
                f"{int(row.get('experiment_id', 0)):04d} "
                f"score={format_float(row.get('score'))} "
                f"tpb={format_float(metrics.get('tokens_per_byte'))} "
                f"dead={format_pct(metrics.get('dead_vocab_frac'))} "
                f"keep={bool(decision.get('keep'))} "
                f"vocab={params.get('vocab_size', '-')}"
                f" cc={params.get('character_coverage', '-')}"
            )
    lines.append("")

    if latest_review.strip():
        lines.append("Latest Review")
        lines.append("-" * 80)
        for line in latest_review.strip().splitlines():
            lines.append(line)
        lines.append("")

    lines.append(f"Run Log Tail ({len(log_tail)})")
    lines.append("-" * 80)
    if not log_tail:
        lines.append("No log output yet.")
    else:
        lines.extend(log_tail)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretty watcher for tokenizer autoresearch output.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Autoresearch tokenizer directory")
    parser.add_argument("--tail", type=int, default=12, help="Run log lines to show")
    parser.add_argument("--history", type=int, default=8, help="Recent history rows to show")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.dir.expanduser().resolve()
    if args.once:
        print(render_dashboard(base_dir, tail_count=args.tail, history_count=args.history))
        return

    try:
        while True:
            os.system("clear")
            print(render_dashboard(base_dir, tail_count=args.tail, history_count=args.history))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
