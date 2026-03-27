#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline


DEFAULT_PR_REFS = [
    64,
    168,
    179,
    180,
    198,
    205,
    208,
    232,
    236,
    244,
    264,
    272,
    274,
    286,
    294,
    305,
    329,
]

STEP_RE = re.compile(
    r"step:(\d+)/(\d+) "
    r"(?:(val_loss):([0-9.]+) val_bpb:([0-9.]+)|train_loss:([0-9.]+))"
    r"(?:.*?train_time:(\d+)ms step_avg:([0-9.]+)ms)"
)
STOP_RE = re.compile(r"stopping_early: wallclock_cap train_time:(\d+)ms step:(\d+)")
FINAL_EXACT_RE = re.compile(
    r"final_[^\n]*?_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)(?: eval_time:(\d+)ms)?"
)
FINAL_ANY_RE = re.compile(
    r"final_[^\n]*? val_loss:([0-9.]+) val_bpb:([0-9.]+)(?: eval_time:(\d+)ms)?"
)
MODEL_PARAMS_RE = re.compile(r"model_params:(\d+)")
WORLD_RE = re.compile(r"world_size:(\d+)")
TRAIN_CFG_RE = re.compile(r"train_batch_tokens:(\d+) train_seq_len:(\d+)")
LR_RE = re.compile(
    r"tie_embeddings:(?:True|False) embed_lr:([0-9.]+) matrix_lr:([0-9.]+) scalar_lr:([0-9.]+)"
)
SEED_RE = re.compile(r"seed:(\d+)")


@dataclass
class StepPoint:
    step: int
    time_ms: int
    step_avg_ms: float
    train_loss: float
    val_loss: float
    val_bpb: float


@dataclass
class RunLog:
    path: str
    group: str
    final_bpb: float
    train_time_ms: int
    model_params: float
    world_size: float
    train_batch_tokens: float
    train_seq_len: float
    embed_lr: float
    matrix_lr: float
    scalar_lr: float
    seed: float
    flag_paidprefix: int
    flag_valonly: int
    flag_correction: int
    flag_ttt: int
    flag_sliding: int
    steps: list[StepPoint]

    @property
    def mainstream(self) -> bool:
        return not (self.flag_paidprefix or self.flag_valonly or self.flag_correction)


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def fetch_pr_refs(pr_numbers: Iterable[int]) -> None:
    for pr in pr_numbers:
        subprocess.run(
            ["git", "fetch", "origin", f"pull/{pr}/head:refs/remotes/origin/pr/{pr}"],
            capture_output=True,
            text=True,
            check=False,
        )


def list_git_logs(ref: str) -> list[str]:
    out = run_git(["ls-tree", "-r", "--name-only", ref, "records/track_10min_16mb"])
    if out.returncode != 0:
        return []
    paths = []
    for path in out.stdout.splitlines():
        if not (path.endswith(".log") or path.endswith("logs.txt")):
            continue
        if "/screen_" in path or path.endswith("screen_batch.log"):
            continue
        paths.append(path)
    return paths


def read_git_file(ref: str, path: str) -> str:
    out = run_git(["show", f"{ref}:{path}"])
    return out.stdout if out.returncode == 0 else ""


def collect_git_log_texts(refs: list[str]) -> dict[str, tuple[str, str]]:
    texts: dict[str, tuple[str, str]] = {}
    for ref in refs:
        for path in list_git_logs(ref):
            text = read_git_file(ref, path)
            if not text:
                continue
            digest = hashlib.sha1(text.encode()).hexdigest()
            texts.setdefault(digest, (path, text))
    return texts


def collect_worktree_log_texts(root: Path) -> dict[str, tuple[str, str]]:
    texts: dict[str, tuple[str, str]] = {}
    for path in sorted(root.glob("records/track_10min_16mb/**/*.log")):
        if "/screen_" in str(path) or path.name == "screen_batch.log":
            continue
        text = path.read_text()
        digest = hashlib.sha1(text.encode()).hexdigest()
        rel = path.relative_to(root).as_posix()
        texts.setdefault(digest, (rel, text))
    return texts


def last_non_nan(points: list[StepPoint], attr: str) -> StepPoint | None:
    valid = [p for p in points if not math.isnan(getattr(p, attr))]
    return valid[-1] if valid else None


def parse_run(path: str, text: str) -> RunLog | None:
    stop = STOP_RE.search(text)
    if not stop:
        return None
    train_time_ms = int(stop.group(1))
    if not (540000 <= train_time_ms <= 660000):
        return None
    finals = FINAL_EXACT_RE.findall(text) or FINAL_ANY_RE.findall(text)
    if not finals:
        return None
    final_bpb = float(finals[-1][1])
    model_params_match = MODEL_PARAMS_RE.search(text)
    world_match = WORLD_RE.search(text)
    cfg_match = TRAIN_CFG_RE.search(text)
    lr_match = LR_RE.search(text)
    seed_match = SEED_RE.search(text)
    lower = path.lower()
    points: list[StepPoint] = []
    for match in STEP_RE.finditer(text):
        if match.group(3) == "val_loss":
            train_loss = math.nan
            val_loss = float(match.group(4))
            val_bpb = float(match.group(5))
        else:
            train_loss = float(match.group(6))
            val_loss = math.nan
            val_bpb = math.nan
        points.append(
            StepPoint(
                step=int(match.group(1)),
                time_ms=int(match.group(7)),
                step_avg_ms=float(match.group(8)),
                train_loss=train_loss,
                val_loss=val_loss,
                val_bpb=val_bpb,
            )
        )
    if not points:
        return None
    return RunLog(
        path=path,
        group=str(PurePosixPath(path).parent),
        final_bpb=final_bpb,
        train_time_ms=train_time_ms,
        model_params=float(model_params_match.group(1)) if model_params_match else math.nan,
        world_size=float(world_match.group(1)) if world_match else math.nan,
        train_batch_tokens=float(cfg_match.group(1)) if cfg_match else math.nan,
        train_seq_len=float(cfg_match.group(2)) if cfg_match else math.nan,
        embed_lr=float(lr_match.group(1)) if lr_match else math.nan,
        matrix_lr=float(lr_match.group(2)) if lr_match else math.nan,
        scalar_lr=float(lr_match.group(3)) if lr_match else math.nan,
        seed=float(seed_match.group(1)) if seed_match else math.nan,
        flag_paidprefix=int("paidprefix" in lower),
        flag_valonly=int("valonly" in lower),
        flag_correction=int("correction" in lower),
        flag_ttt=int("ttt" in lower or "lora" in lower),
        flag_sliding=int("sliding" in lower or "stride32" in lower or "docsliding" in lower),
        steps=points,
    )


def build_runs(repo_root: Path, include_worktree: bool, refs: list[str]) -> list[RunLog]:
    sources = collect_git_log_texts(refs)
    if include_worktree:
        sources.update(collect_worktree_log_texts(repo_root))
    runs: list[RunLog] = []
    for path, text in sources.values():
        run = parse_run(path, text)
        if run:
            runs.append(run)
    runs.sort(key=lambda run: (run.path, run.final_bpb))
    return runs


FEATURE_NAMES = [
    "last_step",
    "last_step_avg_ms",
    "steps_per_sec",
    "train_loss_last",
    "train_loss_min",
    "train_loss_mean_last5",
    "train_loss_delta_last5",
    "train_loss_slope_last5",
    "val_bpb_last",
    "val_bpb_delta_last3",
    "model_params",
    "train_batch_tokens",
    "train_seq_len",
    "embed_lr",
    "matrix_lr",
    "scalar_lr",
    "flag_ttt",
    "flag_sliding",
]


def features_for_run(run: RunLog, cutoff_ms: int) -> dict[str, float] | None:
    pts = [point for point in run.steps if point.time_ms <= cutoff_ms]
    if len(pts) < 5:
        return None
    train_pts = [point for point in pts if not math.isnan(point.train_loss)]
    if not train_pts:
        return None
    val_pts = [point for point in pts if not math.isnan(point.val_bpb)]
    last = pts[-1]
    feats = {
        "last_step": float(last.step),
        "last_step_avg_ms": last.step_avg_ms,
        "steps_per_sec": last.step / (last.time_ms / 1000.0),
        "train_loss_last": train_pts[-1].train_loss,
        "train_loss_min": min(point.train_loss for point in train_pts),
        "train_loss_mean_last5": float(np.mean([point.train_loss for point in train_pts[-5:]])),
        "train_loss_delta_last5": (
            train_pts[-1].train_loss - train_pts[-5].train_loss if len(train_pts) >= 5 else math.nan
        ),
        "train_loss_slope_last5": (
            float(
                np.polyfit(
                    [point.time_ms for point in train_pts[-5:]],
                    [point.train_loss for point in train_pts[-5:]],
                    1,
                )[0]
            )
            if len(train_pts) >= 5
            else math.nan
        ),
        "val_bpb_last": val_pts[-1].val_bpb if val_pts else math.nan,
        "val_bpb_delta_last3": (
            val_pts[-1].val_bpb - val_pts[-3].val_bpb if len(val_pts) >= 3 else math.nan
        ),
        "model_params": run.model_params,
        "train_batch_tokens": run.train_batch_tokens,
        "train_seq_len": run.train_seq_len,
        "embed_lr": run.embed_lr,
        "matrix_lr": run.matrix_lr,
        "scalar_lr": run.scalar_lr,
        "flag_ttt": float(run.flag_ttt),
        "flag_sliding": float(run.flag_sliding),
    }
    return feats


def build_dataset(runs: list[RunLog], cutoff_ms: int, mainstream_only: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[RunLog]]:
    rows = []
    targets = []
    groups = []
    kept_runs = []
    for run in runs:
        if mainstream_only and not run.mainstream:
            continue
        feats = features_for_run(run, cutoff_ms)
        if feats is None:
            continue
        rows.append([feats[name] for name in FEATURE_NAMES])
        targets.append(run.final_bpb)
        groups.append(run.group)
        kept_runs.append(run)
    return np.array(rows, dtype=float), np.array(targets), np.array(groups), FEATURE_NAMES, kept_runs


def make_estimator(model_name: str) -> Pipeline:
    if model_name == "ridge":
        model = Ridge(alpha=1.0)
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=300, random_state=0, min_samples_leaf=2)
    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=300, random_state=0, min_samples_leaf=2)
    else:
        raise ValueError(f"unknown model: {model_name}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def evaluate_cutoff(runs: list[RunLog], cutoff_ms: int, mainstream_only: bool, model_name: str) -> dict[str, float]:
    X, y, groups, _, kept_runs = build_dataset(runs, cutoff_ms, mainstream_only)
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    estimator = make_estimator(model_name)
    pred = cross_val_predict(estimator, X, y, cv=cv, groups=groups)
    abs_err = np.abs(y - pred)
    q95 = float(np.quantile(abs_err, 0.95))
    return {
        "cutoff_ms": cutoff_ms,
        "n_runs": int(len(y)),
        "n_groups": int(len(set(groups))),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
        "q95_interval_halfwidth": q95,
        "coverage": float(np.mean((y >= pred - q95) & (y <= pred + q95))),
    }


def predict_log(
    runs: list[RunLog],
    target_log_path: Path,
    cutoff_ms: int,
    mainstream_only: bool,
    model_name: str,
) -> dict[str, float | str]:
    target = parse_run(str(target_log_path), target_log_path.read_text())
    if target is None:
        raise ValueError(f"could not parse {target_log_path}")
    feats = features_for_run(target, cutoff_ms)
    if feats is None:
        raise ValueError(f"{target_log_path} has insufficient data before cutoff {cutoff_ms}ms")
    X, y, groups, _, _ = build_dataset(runs, cutoff_ms, mainstream_only)
    estimator = make_estimator(model_name)
    estimator.fit(X, y)
    prediction = float(estimator.predict(np.array([[feats[name] for name in FEATURE_NAMES]], dtype=float))[0])
    metrics = evaluate_cutoff(runs, cutoff_ms, mainstream_only, model_name)
    return {
        "path": str(target_log_path),
        "cutoff_ms": cutoff_ms,
        "prediction_bpb": prediction,
        "prediction_interval_95_low": prediction - metrics["q95_interval_halfwidth"],
        "prediction_interval_95_high": prediction + metrics["q95_interval_halfwidth"],
        "historical_mae": metrics["mae"],
        "historical_q95_halfwidth": metrics["q95_interval_halfwidth"],
        "mode": "mainstream" if mainstream_only else "all",
        "model": model_name,
    }


def cmd_summary(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    refs = ["origin/main"] + [f"refs/remotes/origin/pr/{pr}" for pr in DEFAULT_PR_REFS]
    if args.fetch_pr_refs:
        fetch_pr_refs(DEFAULT_PR_REFS)
    runs = build_runs(repo_root, include_worktree=not args.no_worktree, refs=refs)
    print(json.dumps({"total_runs": len(runs), "groups": len({run.group for run in runs})}, indent=2))
    for mainstream_only in (False, True):
        label = "mainstream" if mainstream_only else "all"
        print(f"\n[{label}]")
        for cutoff_ms in args.cutoffs:
            metrics = evaluate_cutoff(runs, cutoff_ms, mainstream_only, args.model)
            print(
                json.dumps(
                    {
                        "cutoff_ms": metrics["cutoff_ms"],
                        "n_runs": metrics["n_runs"],
                        "n_groups": metrics["n_groups"],
                        "mae": round(metrics["mae"], 6),
                        "rmse": round(metrics["rmse"], 6),
                        "q95_interval_halfwidth": round(metrics["q95_interval_halfwidth"], 6),
                        "coverage": round(metrics["coverage"], 6),
                    }
                )
            )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    refs = ["origin/main"] + [f"refs/remotes/origin/pr/{pr}" for pr in DEFAULT_PR_REFS]
    if args.fetch_pr_refs:
        fetch_pr_refs(DEFAULT_PR_REFS)
    runs = build_runs(repo_root, include_worktree=not args.no_worktree, refs=refs)
    result = predict_log(
        runs=runs,
        target_log_path=Path(args.log).resolve(),
        cutoff_ms=args.cutoff_ms,
        mainstream_only=(args.mode == "mainstream"),
        model_name=args.model,
    )
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict 600s final val_bpb from an early stopping point.")
    parser.add_argument("--repo-root", default=".", help="Path to the git repo root.")
    parser.add_argument("--model", default="extra_trees", choices=["ridge", "random_forest", "extra_trees"])
    parser.add_argument("--fetch-pr-refs", action="store_true", help="Fetch tracked PR refs before analysis.")
    parser.add_argument("--no-worktree", action="store_true", help="Use git refs only, not current worktree logs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Evaluate the predictor on historical logs.")
    summary.add_argument("--cutoffs", nargs="+", type=int, default=[60000, 120000, 180000, 300000])
    summary.set_defaults(func=cmd_summary)

    predict = subparsers.add_parser("predict", help="Predict final val_bpb for a specific log file.")
    predict.add_argument("--log", required=True, help="Path to the log to score.")
    predict.add_argument("--cutoff-ms", required=True, type=int, help="Early stopping cutoff in milliseconds.")
    predict.add_argument("--mode", choices=["mainstream", "all"], default="mainstream")
    predict.set_defaults(func=cmd_predict)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
