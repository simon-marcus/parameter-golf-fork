from __future__ import annotations

import csv
import math
import re
import statistics
import sys
from pathlib import Path


EXACT_PATTERNS = {
    "final_int6_roundtrip_bpb": re.compile(r"final_int6_roundtrip_exact val_loss:[^\s]+ val_bpb:(?P<value>[0-9.]+)"),
    "final_int6_sliding_window_bpb": re.compile(r"final_int6_sliding_window_exact val_loss:[^\s]+ val_bpb:(?P<value>[0-9.]+)"),
    "final_int6_sliding_window_s64_bpb": re.compile(r"final_int6_sliding_window_s64_exact val_loss:[^\s]+ val_bpb:(?P<value>[0-9.]+)"),
    "legal_ttt_bpb": re.compile(r"legal_ttt_exact val_loss:[^\s]+ val_bpb:(?P<value>[0-9.]+)"),
    "diagnostic_post_ema_bpb": re.compile(r"DIAGNOSTIC post_ema val_loss:[^\s]+ val_bpb:(?P<value>[0-9.]+)"),
}

META_PATTERNS = {
    "activation_mode": re.compile(r"activation_mode:(?P<value>\S+)"),
    "seed": re.compile(r"seed:(?P<value>\d+)"),
    "ttt_mode": re.compile(r"ttt:start mode=(?P<value>\S+)"),
    "ttt_param_mode": re.compile(r"ttt:start mode=\S+ param_mode=(?P<value>\S+)"),
}


def parse_log(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    row: dict[str, object] = {
        "log_path": str(log_path),
        "status": "ok" if "legal_ttt_exact" in text else "incomplete",
    }
    for key, pattern in META_PATTERNS.items():
        match = pattern.search(text)
        if match:
            row[key] = match.group("value")
    for key, pattern in EXACT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            row[key] = float(match.group("value"))
    row["case_name"] = log_path.parent.parent.name
    row["seed_dir"] = log_path.parent.name
    if "seed" not in row and row["seed_dir"].startswith("seed_"):
        row["seed"] = row["seed_dir"].removeprefix("seed_")
    return row


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_case: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_name"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    metric_keys = [*EXACT_PATTERNS.keys()]
    for case_name, case_rows in sorted(by_case.items()):
        out: dict[str, object] = {"case_name": case_name, "runs": len(case_rows)}
        out["completed_runs"] = sum(1 for row in case_rows if row.get("status") == "ok")
        out["activation_mode"] = case_rows[0].get("activation_mode", "")
        out["ttt_mode"] = case_rows[0].get("ttt_mode", "")
        out["ttt_param_mode"] = case_rows[0].get("ttt_param_mode", "")
        for key in metric_keys:
            values = [float(row[key]) for row in case_rows if key in row]
            if values:
                out[f"{key}_mean"] = statistics.fmean(values)
                out[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
                out[f"{key}_best"] = min(values)
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# 1-GPU Activation Suite Summary",
        "",
        "| Case | Runs | Activation | TTT | Param mode | Roundtrip mean | Sliding mean | Legal TTT mean | Legal TTT std | Best legal TTT |",
        "|------|------|------------|-----|------------|----------------|--------------|----------------|---------------|----------------|",
    ]
    for row in summary_rows:
        lines.append(
            "| {case} | {runs}/{completed} | {activation} | {ttt} | {param} | {roundtrip} | {sliding} | {ttt_mean} | {ttt_std} | {ttt_best} |".format(
                case=row.get("case_name", ""),
                runs=row.get("runs", 0),
                completed=row.get("completed_runs", 0),
                activation=row.get("activation_mode", ""),
                ttt=row.get("ttt_mode", ""),
                param=row.get("ttt_param_mode", ""),
                roundtrip=fmt(row.get("final_int6_roundtrip_bpb_mean")),  # type: ignore[arg-type]
                sliding=fmt(row.get("final_int6_sliding_window_bpb_mean")),  # type: ignore[arg-type]
                ttt_mean=fmt(row.get("legal_ttt_bpb_mean")),  # type: ignore[arg-type]
                ttt_std=fmt(row.get("legal_ttt_bpb_std")),  # type: ignore[arg-type]
                ttt_best=fmt(row.get("legal_ttt_bpb_best")),  # type: ignore[arg-type]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python summarize_streaming_ttt_activation_suite.py <suite_root>")
        return 1
    suite_root = Path(sys.argv[1]).resolve()
    logs = sorted(suite_root.glob("*/seed_*/train.log"))
    if not logs:
        print(f"No train.log files found under {suite_root}")
        return 1

    rows = [parse_log(log_path) for log_path in logs]
    summary_rows = aggregate(rows)

    raw_csv = suite_root / "results_raw.csv"
    summary_csv = suite_root / "results_summary.csv"
    summary_md = suite_root / "results_summary.md"
    write_csv(raw_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_markdown(summary_md, summary_rows)

    print(f"Wrote {raw_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
