#!/usr/bin/env python3
"""
Minimal dummy autoresearch loop to debug Claude CLI + loop control.

Game:
- A hidden country is stored in country.txt (not in Claude's working directory).
- Claude writes guesses to country_guess.txt, one per line.
- Loop checks guesses independently and stops when the country is guessed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def normalize_country(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def read_guess_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    out: list[str] = []
    for line in raw.splitlines():
        cleaned = normalize_country(line)
        if cleaned:
            out.append(cleaned)
    return out


def build_prompt(guess_count: int) -> str:
    if guess_count == 0:
        return (
            "This is the first attempt: you should create and write your guess "
            "to country_guess.txt.\n"
            "Rules:\n"
            "- Write exactly one country name on one line.\n"
            "- Do not explain.\n"
            "- Ensure country_guess.txt exists after your edit."
        )
    return (
        f"We've had {guess_count} guesses so far.\n"
        "Take a look at previous guesses in country_guess.txt so you don't repeat one, "
        "then add exactly one new country guess.\n"
        "Rules:\n"
        "- Keep one country name per line.\n"
        "- Append one new line only.\n"
        "- Do not explain."
    )


@dataclass
class StepResult:
    step: int
    prompt: str
    returncode: int
    elapsed_s: float
    status: str
    total_guesses: int
    new_guess: str | None
    solved: bool
    output_tail: str


def run_agent(
    workspace: Path,
    prompt: str,
    model: str,
    effort: str,
    allowed_tools: str,
    timeout_s: int,
    permission_mode: str,
) -> tuple[int, str, float]:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--effort",
        effort,
        "--allowedTools",
        allowed_tools,
        "--permission-mode",
        permission_mode,
        "--output-format",
        "text",
    ]
    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.time() - started
        return result.returncode, (result.stdout + "\n" + result.stderr), elapsed
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        output = (exc.stdout or "") + "\n" + (exc.stderr or "") + "\n[TIMEOUT]"
        return -1, output, elapsed


def write_history(history_file: Path, result: StepResult) -> None:
    entry = {
        "step": result.step,
        "status": result.status,
        "returncode": result.returncode,
        "elapsed_s": round(result.elapsed_s, 2),
        "total_guesses": result.total_guesses,
        "new_guess": result.new_guess,
        "solved": result.solved,
        "prompt": result.prompt,
        "output_tail": result.output_tail,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with history_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dummy country-guess autoresearch loop.")
    parser.add_argument("--country-file", default="country.txt", help="Hidden country file path.")
    parser.add_argument(
        "--workspace",
        default="autoresearch/dummy_country_game",
        help="Claude working directory (contains country_guess.txt).",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--allowed-tools", default="Bash,Read,Edit")
    parser.add_argument("--permission-mode", default="bypassPermissions")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--reset", action="store_true", help="Delete prior guess/history files.")
    parser.add_argument("--verbose", action="store_true", help="Print full Claude output each step.")
    args = parser.parse_args()

    country_file = Path(args.country_file).resolve()
    if not country_file.exists():
        print(f"Error: hidden country file not found: {country_file}", file=sys.stderr)
        return 1

    hidden_country = normalize_country(country_file.read_text(encoding="utf-8"))
    if not hidden_country:
        print(f"Error: hidden country file is empty: {country_file}", file=sys.stderr)
        return 1

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    guess_file = workspace / "country_guess.txt"
    history_file = workspace / "history.jsonl"

    if args.reset:
        for p in (guess_file, history_file):
            if p.exists():
                p.unlink()

    print("Dummy country autoresearch loop")
    print(f"  hidden country file: {country_file}")
    print(f"  workspace:           {workspace}")
    print(f"  guess file:          {guess_file}")
    print(f"  max steps:           {args.max_steps}")
    print(f"  agent cmd:           claude -p ... --allowedTools \"{args.allowed_tools}\"")
    print("")

    for step in range(1, args.max_steps + 1):
        before = read_guess_lines(guess_file)
        if hidden_country in before:
            print(f"Already solved before step {step}: {hidden_country}")
            return 0

        prompt = build_prompt(len(before))
        rc, output, elapsed = run_agent(
            workspace=workspace,
            prompt=prompt,
            model=args.model,
            effort=args.effort,
            allowed_tools=args.allowed_tools,
            timeout_s=args.timeout_s,
            permission_mode=args.permission_mode,
        )
        after = read_guess_lines(guess_file)

        new_guess = None
        status = "ok"
        if rc == -1:
            status = "timeout"
        elif rc != 0:
            status = f"agent_exit_{rc}"
        elif not guess_file.exists():
            status = "missing_guess_file"
        elif after == before:
            status = "no_changes"
        elif len(after) <= len(before):
            status = "no_new_guess"
        else:
            candidate = after[-1]
            new_guess = candidate
            if candidate in before:
                status = "repeated_guess"

        solved = hidden_country in after
        tail = "\n".join((output or "").strip().splitlines()[-8:])
        result = StepResult(
            step=step,
            prompt=prompt,
            returncode=rc,
            elapsed_s=elapsed,
            status=status,
            total_guesses=len(after),
            new_guess=new_guess,
            solved=solved,
            output_tail=tail,
        )
        write_history(history_file, result)

        print(
            f"step={step:02d} status={status:<16} rc={rc:<3} "
            f"elapsed={elapsed:5.1f}s guesses={len(after):02d} new_guess={new_guess!r}"
        )
        if args.verbose and tail:
            print("  output tail:")
            for line in tail.splitlines():
                print(f"    {line}")

        if solved:
            print(f"\nSolved in {step} step(s): {hidden_country}")
            return 0

    print(f"\nNot solved after {args.max_steps} step(s). Hidden country: {hidden_country}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
