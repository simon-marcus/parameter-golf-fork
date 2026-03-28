"""
Autoresearch loop for cache-only parameter optimization.

Uses Claude Code (`claude -p`) to propose cache hyperparameter changes,
runs eval_cache_only.py (~40s per experiment), and keeps or reverts.

No GPU needed. Runs on any machine with Python + numpy.

Usage:
    # Basic (uses default pass1 arrays path):
    PASS1_PATH=runpod_artifacts/2026-03-27_beatcm_8x/pass1_arrays.npz \
        python3 autoresearch_cache.py

    # With custom settings:
    MAX_EXPERIMENTS=200 AUTORESEARCH_MODEL=sonnet \
    PASS1_PATH=pass1_arrays.npz \
        python3 autoresearch_cache.py

Environment variables:
    PASS1_PATH          Path to pass1_arrays.npz (required)
    MAX_EXPERIMENTS     Max experiments before stopping (default: 100)
    AUTORESEARCH_MODEL  Claude model: opus, sonnet, haiku (default: sonnet)
    CLAUDE_EFFORT       Thinking effort: high, medium, low (default: high)
    BASELINE_BPB        Seed BPB for first comparison (default: 0, auto-detect)
    PROPOSAL_TIMEOUT    Seconds to wait for Claude proposal (default: 180)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------
# Configuration
# ---------------------

EVAL_SCRIPT = Path(__file__).parent / "eval_cache_only.py"
PASS1_PATH = os.environ.get("PASS1_PATH", "")
MAX_EXPERIMENTS = int(os.environ.get("MAX_EXPERIMENTS", "100"))
CLAUDE_MODEL = os.environ.get("AUTORESEARCH_MODEL", "sonnet")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "high")
BASELINE_BPB = float(os.environ.get("BASELINE_BPB", "0"))
PROPOSAL_TIMEOUT = int(os.environ.get("PROPOSAL_TIMEOUT", "180"))
EVAL_TIMEOUT = int(os.environ.get("EVAL_TIMEOUT", "300"))

AUTORESEARCH_DIR = Path(__file__).parent / "autoresearch_cache"
HISTORY_FILE = AUTORESEARCH_DIR / "history.jsonl"
EXPERIMENTS_DIR = AUTORESEARCH_DIR / "experiments"
CONFIG_FILE = AUTORESEARCH_DIR / "cache_config.json"
BEST_CONFIG_FILE = AUTORESEARCH_DIR / "best_config.json"

# The parameters Claude can tune
TUNABLE_PARAMS = {
    "ngram_min_order": {"type": "int", "default": 2, "range": [2, 2], "desc": "Minimum n-gram order"},
    "ngram_max_order": {"type": "int", "default": 12, "range": [6, 24], "desc": "Maximum n-gram order"},
    "ngram_num_buckets": {"type": "int", "default": 4_194_304, "range": [2_097_152, 67_108_864], "desc": "N-gram hash table buckets (power of 2, min 2M to avoid collision exploit)"},
    "ngram_min_count": {"type": "int", "default": 2, "range": [1, 10], "desc": "Minimum context count for n-gram match"},
    "ngram_alpha_min": {"type": "float", "default": 0.05, "range": [0.0, 0.3], "desc": "Minimum blend alpha"},
    "ngram_alpha_max": {"type": "float", "default": 0.80, "range": [0.3, 1.0], "desc": "Maximum blend alpha (before order mult)"},
    "ngram_entropy_center": {"type": "float", "default": 3.0, "range": [1.0, 5.0], "desc": "Entropy sigmoid center"},
    "ngram_entropy_scale": {"type": "float", "default": 2.0, "range": [0.5, 5.0], "desc": "Entropy sigmoid scale"},
    "leave_one_out": {"type": "int", "default": 1, "range": [0, 1], "desc": "Leave-one-out correction (0 or 1)"},
    "phrase_enabled": {"type": "int", "default": 0, "range": [0, 1], "desc": "Enable phrase cache"},
    "phrase_probe_lengths": {"type": "str", "default": "64,56,48,36,28,20,16", "desc": "Phrase probe lengths (comma-separated)"},
    "phrase_num_buckets": {"type": "int", "default": 4_194_304, "range": [1_048_576, 67_108_864], "desc": "Phrase hash table buckets per length"},
    "phrase_min_count": {"type": "int", "default": 1, "range": [1, 20], "desc": "Minimum count for phrase match"},
    "phrase_alpha_min": {"type": "float", "default": 0.88, "range": [0.0, 1.0], "desc": "Minimum phrase blend alpha"},
    "phrase_alpha_max": {"type": "float", "default": 0.995, "range": [0.0, 1.0], "desc": "Maximum phrase blend alpha"},
    "phrase_entropy_center": {"type": "float", "default": 2.5, "range": [1.0, 5.0], "desc": "Phrase entropy sigmoid center"},
    "phrase_entropy_scale": {"type": "float", "default": 2.0, "range": [0.5, 5.0], "desc": "Phrase entropy sigmoid scale"},
    "calibration_frac": {"type": "float", "default": 0.05, "range": [0.01, 0.20], "desc": "Fraction of tokens for calibration"},
    "calibration_alpha_grid": {"type": "str", "default": "0.70,0.80,0.90,0.95,0.99", "desc": "Alpha max calibration grid"},
    "calibration_center_grid": {"type": "str", "default": "2.0,2.5,3.0,3.5", "desc": "Entropy center calibration grid"},
    "calibration_phrase_grid": {"type": "str", "default": "0.980,0.990,0.995,0.999", "desc": "Phrase alpha max calibration grid"},
}


def ensure_dirs():
    AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)


def load_config() -> dict:
    """Load current config or create default."""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {k: v["default"] for k, v in TUNABLE_PARAMS.items()}


def save_config(config: dict):
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    entries = []
    for line in HISTORY_FILE.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def append_history(entry: dict):
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def config_to_env_args(config: dict) -> list[str]:
    """Convert config dict to CLI args for eval_cache_only.py."""
    args = []
    mapping = {
        "ngram_min_order": "--ngram-min-order",
        "ngram_max_order": "--ngram-max-order",
        "ngram_num_buckets": "--ngram-num-buckets",
        "ngram_min_count": "--ngram-min-count",
        "ngram_alpha_min": "--ngram-alpha-min",
        "ngram_alpha_max": "--ngram-alpha-max",
        "ngram_entropy_center": "--ngram-entropy-center",
        "ngram_entropy_scale": "--ngram-entropy-scale",
        "leave_one_out": "--leave-one-out",
        "phrase_enabled": "--phrase-enabled",
        "phrase_probe_lengths": "--phrase-probe-lengths",
        "phrase_num_buckets": "--phrase-num-buckets",
        "phrase_min_count": "--phrase-min-count",
        "phrase_alpha_min": "--phrase-alpha-min",
        "phrase_alpha_max": "--phrase-alpha-max",
        "phrase_entropy_center": "--phrase-entropy-center",
        "phrase_entropy_scale": "--phrase-entropy-scale",
        "calibration_frac": "--calibration-frac",
        "calibration_alpha_grid": "--calibration-alpha-grid",
        "calibration_center_grid": "--calibration-center-grid",
        "calibration_phrase_grid": "--calibration-phrase-grid",
    }
    for key, flag in mapping.items():
        if key in config:
            args.extend([flag, str(config[key])])
    return args


def run_eval(config: dict) -> tuple[float | None, str]:
    """Run eval_cache_only.py with given config. Returns (val_bpb, output)."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--pass1", PASS1_PATH,
    ] + config_to_env_args(config)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=EVAL_TIMEOUT,
        )
        output = result.stdout + result.stderr
        # Parse RESULT line
        for line in output.split("\n"):
            m = re.search(r"RESULT val_bpb=([\d.]+)", line)
            if m:
                return float(m.group(1)), output
        return None, output
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, f"ERROR: {e}"


def build_proposal_prompt(config: dict, history: list[dict], best_bpb: float) -> str:
    """Build the prompt for Claude to propose a config change."""
    param_desc = "\n".join(
        f"  {k}: {config.get(k, v['default'])} (range: {v['range']}, {v['desc']})"
        if v["type"] != "str" else
        f"  {k}: {config.get(k, v['default'])} ({v['desc']})"
        for k, v in TUNABLE_PARAMS.items()
    )

    recent = history[-15:] if len(history) > 15 else history
    history_str = ""
    for h in recent:
        kept = "KEPT" if h.get("kept") else "REVERTED"
        bpb = h.get("val_bpb", "N/A")
        desc = h.get("description", "")[:120]
        changes = h.get("changes", {})
        history_str += f"  [{kept}] bpb={bpb} changes={changes} — {desc}\n"

    return f"""You are optimizing an n-gram cache engine for a language model compression competition.

OBJECTIVE: Minimize val_bpb. Current best: {best_bpb:.8f}. Target: < 0.0804 (PR #933).

The cache is a two-pass n-gram hash-table system that blends cached probabilities with neural model probabilities.
The neural model is fixed (1.121 BPB). Only cache hyperparameters can change.

CRITICAL INSIGHT: Hash collisions are a FEATURE not a bug. With 4M buckets and 62M tokens, collision-inflated
counts act as pseudo-Bayesian smoothing. Bigger tables (32M) performed catastrophically worse (0.267 vs 0.089)
because singleton n-grams get filtered by min_count after leave-one-out correction.

The phrase cache at 4M buckets also failed catastrophically (0.131) because it matched 100% of tokens with
random collision noise at high alpha, overwriting good n-gram estimates. If enabling phrase cache, use MUCH
larger buckets (64M+) and higher min_count (10+).

CURRENT CONFIGURATION:
{param_desc}

RECENT EXPERIMENTS:
{history_str if history_str else "  (none yet — this is the first experiment)"}

Propose ONE surgical change to the configuration. Return a JSON object with:
- "description": brief explanation of what you're changing and why
- "changes": dict of param_name -> new_value (only changed params)

RULES:
- Change 1-3 parameters at a time, not more
- Each change should have a clear hypothesis
- The calibration grid search will find optimal alpha_max and entropy_center,
  so focus on structural params (bucket count, orders, min_count, phrase settings)
- Consider the collision-as-smoothing effect when changing bucket counts
- Power-of-2 bucket counts only

Return ONLY the JSON object, no other text.
"""


def propose_change(config: dict, history: list[dict], best_bpb: float) -> tuple[dict, str]:
    """Ask Claude to propose a config change. Returns (changes_dict, description)."""
    prompt = build_proposal_prompt(config, history, best_bpb)

    cmd = [
        "claude", "-p", prompt,
        "--model", CLAUDE_MODEL,
        "--output-format", "text",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=PROPOSAL_TIMEOUT,
        )
        text = result.stdout.strip()
        # Extract JSON from response — try multiple strategies
        # Strategy 1: find ```json ... ``` block
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_match:
            text = code_match.group(1)
        # Strategy 2: find outermost { ... } containing "changes"
        brace_depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    candidate = text[start:i+1]
                    if '"changes"' in candidate or '"description"' in candidate:
                        try:
                            proposal = json.loads(candidate)
                            return proposal.get("changes", {}), proposal.get("description", "no description")
                        except json.JSONDecodeError:
                            continue
        return {}, f"Failed to parse proposal: {text[:300]}"
    except subprocess.TimeoutExpired:
        return {}, "Proposal timed out"
    except Exception as e:
        return {}, f"Proposal error: {e}"


def save_experiment(
    exp_id: int, description: str, config: dict,
    changes: dict, val_bpb: float | None, kept: bool,
    output: str, elapsed: float,
):
    exp_dir = EXPERIMENTS_DIR / f"{exp_id:04d}"
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "output.txt").write_text(output)
    kept_str = "KEPT" if kept else "REVERTED"
    bpb_str = f"{val_bpb:.8f}" if val_bpb is not None else "FAILED"
    (exp_dir / "README.md").write_text(
        f"# Experiment {exp_id}\n\n"
        f"**Result:** {kept_str}\n"
        f"**val_bpb:** {bpb_str}\n"
        f"**Changes:** {json.dumps(changes)}\n"
        f"**Time:** {elapsed:.1f}s\n\n"
        f"## Description\n{description}\n"
    )


def main():
    if not PASS1_PATH or not Path(PASS1_PATH).exists():
        print(f"ERROR: PASS1_PATH={PASS1_PATH!r} not found. Set PASS1_PATH to pass1_arrays.npz")
        sys.exit(1)

    if not EVAL_SCRIPT.exists():
        print(f"ERROR: eval script not found at {EVAL_SCRIPT}")
        sys.exit(1)

    ensure_dirs()
    config = load_config()
    history = load_history()
    best_bpb = BASELINE_BPB

    # Run baseline if no history
    if not history or best_bpb == 0:
        print("=== Running baseline evaluation ===")
        bpb, output = run_eval(config)
        if bpb is None:
            print(f"Baseline eval failed:\n{output}")
            sys.exit(1)
        best_bpb = bpb
        entry = {
            "experiment_id": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": "Baseline config",
            "changes": {},
            "config": config,
            "val_bpb": bpb,
            "kept": True,
        }
        append_history(entry)
        save_config(config)
        json.loads(BEST_CONFIG_FILE.read_text()) if BEST_CONFIG_FILE.exists() else None
        BEST_CONFIG_FILE.write_text(json.dumps(config, indent=2))
        print(f"Baseline: val_bpb={bpb:.8f}")
        history.append(entry)

    # Find best from history
    for h in history:
        if h.get("kept") and h.get("val_bpb") is not None:
            if h["val_bpb"] < best_bpb or best_bpb == 0:
                best_bpb = h["val_bpb"]
                config = h.get("config", config)

    print(f"\n=== Starting autoresearch cache loop ===")
    print(f"Best BPB: {best_bpb:.8f}")
    print(f"Target: < 0.0804")
    print(f"Model: {CLAUDE_MODEL}")
    print(f"Max experiments: {MAX_EXPERIMENTS}")
    print(f"Pass1 path: {PASS1_PATH}\n")

    start_exp = len(history)
    for i in range(MAX_EXPERIMENTS):
        exp_id = start_exp + i
        print(f"\n{'='*60}")
        print(f"Experiment {exp_id} (best={best_bpb:.8f}, target=0.0804)")
        print(f"{'='*60}")

        # Propose
        t_propose = time.perf_counter()
        changes, description = propose_change(config, history, best_bpb)
        t_proposed = time.perf_counter()
        print(f"Proposed ({t_proposed - t_propose:.1f}s): {description}")
        print(f"  Changes: {changes}")

        if not changes:
            print("  No changes proposed, skipping")
            continue

        # Apply changes with bounds enforcement
        test_config = dict(config)
        clamped = False
        for k, v in changes.items():
            if k not in TUNABLE_PARAMS:
                print(f"  WARNING: Unknown param {k}, ignoring")
                continue
            spec = TUNABLE_PARAMS[k]
            if "range" in spec and spec["type"] in ("int", "float"):
                lo, hi = spec["range"]
                orig_v = v
                v = max(lo, min(hi, type(lo)(v)))
                if v != orig_v:
                    print(f"  CLAMPED {k}: {orig_v} -> {v} (range [{lo}, {hi}])")
                    clamped = True
            test_config[k] = v

        # Evaluate
        t_eval = time.perf_counter()
        val_bpb, output = run_eval(test_config)
        t_done = time.perf_counter()
        eval_time = t_done - t_eval

        if val_bpb is None:
            print(f"  FAILED ({eval_time:.1f}s): {output[:200]}")
            entry = {
                "experiment_id": exp_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": description,
                "changes": changes,
                "config": test_config,
                "val_bpb": None,
                "kept": False,
                "error": output[:500],
                "eval_seconds": eval_time,
            }
            append_history(entry)
            history.append(entry)
            save_experiment(exp_id, description, test_config, changes, None, False, output, eval_time)
            continue

        improved = val_bpb < best_bpb
        kept = improved

        print(f"  val_bpb={val_bpb:.8f} (best={best_bpb:.8f}) "
              f"{'KEPT ✓' if kept else 'REVERTED ✗'} ({eval_time:.1f}s)")

        if kept:
            best_bpb = val_bpb
            config = test_config
            save_config(config)
            BEST_CONFIG_FILE.write_text(json.dumps(config, indent=2))

        entry = {
            "experiment_id": exp_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "changes": changes,
            "config": test_config,
            "val_bpb": val_bpb,
            "kept": kept,
            "eval_seconds": eval_time,
        }
        append_history(entry)
        history.append(entry)
        save_experiment(exp_id, description, test_config, changes, val_bpb, kept, output, eval_time)

    print(f"\n{'='*60}")
    print(f"Autoresearch complete: {MAX_EXPERIMENTS} experiments")
    print(f"Best val_bpb: {best_bpb:.8f}")
    print(f"Best config: {BEST_CONFIG_FILE}")
    print(f"History: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
