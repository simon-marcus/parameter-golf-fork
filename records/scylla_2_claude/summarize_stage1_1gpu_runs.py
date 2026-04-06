#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("records/scylla_2_claude/stage1_1gpu_runs")

PATTERNS = {
    "diag_bpb": re.compile(r"DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:(?P<v>[0-9.]+)"),
    "roundtrip_bpb": re.compile(r"final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:(?P<v>[0-9.]+)"),
    "sliding_bpb": re.compile(r"final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:(?P<v>[0-9.]+)"),
    "int8_bpb": re.compile(r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:(?P<v>[0-9.]+)"),
    "artifact_bytes": re.compile(r"Total submission size int6\+lzma: (?P<v>\d+) bytes"),
    "step": re.compile(r"step:(?P<cur>\d+)/(?P<total>\d+)"),
}


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, object] = {"completed": False}
    for key, pattern in PATTERNS.items():
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        last = matches[-1]
        if key == "step":
            out["step"] = int(last.group("cur"))
            out["total_steps"] = int(last.group("total"))
        elif key == "artifact_bytes":
            out[key] = int(last.group("v"))
        else:
            out[key] = float(last.group("v"))
    out["completed"] = "final_int8_zlib_roundtrip_exact" in text
    return out


def score(metrics: dict) -> float:
    for key in ("int8_bpb", "sliding_bpb", "roundtrip_bpb", "diag_bpb"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float("inf")


def main() -> int:
    if not ROOT.exists():
        print(f"missing root: {ROOT}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for config_path in sorted(ROOT.glob("*/*/seed_*/config.json")):
        log_path = config_path.with_name("train.log")
        if not log_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = parse_log(log_path)
        rows.append(
            {
                "role": config["role"],
                "case": config["case_name"],
                "seed": config["seed"],
                "dataset_family": config["dataset_family"],
                "qk_gain_init": config["qk_gain_init"],
                "xsa_last_n": config["xsa_last_n"],
                "mlp_mult": config["mlp_mult"],
                "warmdown_iters": config["warmdown_iters"],
                "config_path": str(config_path),
                "log_path": str(log_path),
                "metrics": metrics,
            }
        )

    if not rows:
        print("no stage1 results found", file=sys.stderr)
        return 1

    rows.sort(key=lambda row: (score(row["metrics"]), row["role"], row["case"], row["seed"]))

    header = (
        f"{'role':<18} {'case':<24} {'state':<10} {'diag':>8} "
        f"{'int6':>8} {'slide':>8} {'int8':>8} {'bytes':>10} {'step':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        m = row["metrics"]
        state = "done" if m.get("completed") else "running"
        diag = f"{m['diag_bpb']:.4f}" if "diag_bpb" in m else "-"
        int6 = f"{m['roundtrip_bpb']:.4f}" if "roundtrip_bpb" in m else "-"
        slide = f"{m['sliding_bpb']:.4f}" if "sliding_bpb" in m else "-"
        int8 = f"{m['int8_bpb']:.4f}" if "int8_bpb" in m else "-"
        artifact = str(m["artifact_bytes"]) if "artifact_bytes" in m else "-"
        step = (
            f"{m.get('step', '-')}/{m.get('total_steps', '-')}"
            if "step" in m or "total_steps" in m
            else "-"
        )
        print(
            f"{row['role']:<18} {row['case']:<24} {state:<10} {diag:>8} "
            f"{int6:>8} {slide:>8} {int8:>8} {artifact:>10} {step:>9}"
        )

    winners: dict[str, dict] = {}
    for row in rows:
        family = row["dataset_family"]
        if family not in winners:
            winners[family] = row

    print("\nWinners by dataset family")
    for family, row in winners.items():
        value = score(row["metrics"])
        rendered = f"{value:.8f}" if value != float("inf") else "pending"
        print(f"{family}: {row['case']}  score={rendered}  config={row['config_path']}")

    print("\nJSON")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
