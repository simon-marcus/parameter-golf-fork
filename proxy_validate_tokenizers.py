"""
Local proxy-validation harness for tokenizer candidates.

For each candidate tokenizer, this script:
1. ensures decoded proxy train/val text samples exist
2. retokenizes those samples into challenge-style shard files
3. exports tokenizer metadata
4. runs train_gpt.py with a fixed proxy recipe
5. parses and summarizes training/eval metrics
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_DATASET = Path("/Users/simon/Code/parameter-golf-local/data/datasets/fineweb10B_sp1024")
DEFAULT_LOCAL_TOKENIZER = Path("/Users/simon/Code/parameter-golf-local/data/tokenizers/fineweb_1024_bpe.model")
OUT_DIR = ROOT / "proxy_validation"
TRAIN_PYTHON = os.environ.get("PROXY_TRAIN_PYTHON", sys.executable)


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def ensure_text_sample(split: str, output_path: Path, dataset_dir: Path, tokenizer_path: Path, max_tokens: int) -> None:
    if output_path.exists():
        return
    cmd = [
        sys.executable,
        str(ROOT / "data" / "extract_text_sample_from_shards.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--tokenizer-path",
        str(tokenizer_path),
        "--split",
        split,
        "--max-tokens",
        str(max_tokens),
        "--chunk-tokens",
        "2048",
        "--output",
        str(output_path),
    ]
    run_cmd(cmd)


def parse_history_candidates(history_path: Path, top_k: int) -> list[Path]:
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [row for row in rows if row.get("kept")]
    kept.sort(key=lambda row: float(row.get("score", 1e9)))
    seen: set[str] = set()
    out: list[Path] = []
    for row in kept:
        model_path = str(row["metrics"]["model_path"])
        if model_path in seen:
            continue
        seen.add(model_path)
        out.append(Path(model_path))
        if len(out) >= top_k:
            break
    return out


def export_sentencepiece_meta(model_path: Path) -> Path:
    meta_path = model_path.with_suffix(".meta.npz")
    cmd = [sys.executable, str(ROOT / "data" / "export_sentencepiece_meta.py"), str(model_path), "--validate"]
    run_cmd(cmd)
    return meta_path


def export_tokenmonster_meta(vocab_ref: str, name: str) -> Path:
    meta_dir = OUT_DIR / "tokenizers"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{sanitize_name(name)}.meta.npz"
    cmd = [sys.executable, str(ROOT / "data" / "export_tokenmonster_meta.py"), vocab_ref, "--output", str(meta_path)]
    run_cmd(cmd)
    return meta_path


def build_proxy_dataset(
    name: str,
    family: str,
    tokenizer_ref: str,
    train_text: Path,
    val_text: Path,
    *,
    train_chunks: int,
    val_chunks: int,
) -> Path:
    output_dir = OUT_DIR / "datasets" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "data" / "build_proxy_tokenizer_dataset.py"),
        "--train-text",
        str(train_text),
        "--val-text",
        str(val_text),
        "--family",
        family,
        "--tokenizer-path",
        tokenizer_ref,
        "--output-dir",
        str(output_dir),
        "--train-max-chunks",
        str(train_chunks),
        "--val-max-chunks",
        str(val_chunks),
    ]
    run_cmd(cmd)
    return output_dir


def parse_train_log(log_text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    patterns = {
        "model_params": r"model_params:(\d+)",
        "train_time_ms": r"stopping_early: wallclock_cap train_time:(\d+)ms",
        "prequant_val_bpb": r"step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)",
        "postquant_val_bpb": r"final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)",
        "artifact_bytes": r"Total submission size int8\+zlib: (\d+) bytes",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, log_text)
        if not matches:
            continue
        value = matches[-1]
        if key in {"model_params", "train_time_ms", "artifact_bytes"}:
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def run_proxy_train(
    name: str,
    dataset_dir: Path,
    tokenizer_path: str,
    meta_path: Path,
    *,
    vocab_size: int,
    max_wallclock_seconds: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    run_dir = OUT_DIR / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    env = os.environ.copy()
    env.update(
        {
            "DATA_PATH": str(dataset_dir),
            "TOKENIZER_PATH": tokenizer_path,
            "TOKENIZER_META_PATH": str(meta_path),
            "VOCAB_SIZE": str(vocab_size),
            "RUN_ID": f"proxy_{name}",
            "SEED": str(seed),
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
            "ITERATIONS": str(iterations),
            "VAL_LOSS_EVERY": "0",
            "TRAIN_LOG_EVERY": "50",
            "NUM_LAYERS": env.get("PROXY_NUM_LAYERS", "4"),
            "MODEL_DIM": env.get("PROXY_MODEL_DIM", "256"),
            "NUM_HEADS": env.get("PROXY_NUM_HEADS", "4"),
            "NUM_KV_HEADS": env.get("PROXY_NUM_KV_HEADS", "2"),
            "MLP_MULT": env.get("PROXY_MLP_MULT", "2"),
            "TRAIN_SEQ_LEN": env.get("PROXY_TRAIN_SEQ_LEN", "512"),
            "TRAIN_BATCH_TOKENS": env.get("PROXY_TRAIN_BATCH_TOKENS", "65536"),
            "VAL_BATCH_SIZE": env.get("PROXY_VAL_BATCH_SIZE", "65536"),
        }
    )
    result = subprocess.run(
        [TRAIN_PYTHON, str(ROOT / "train_gpt.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    log_text = result.stdout + result.stderr
    log_path.write_text(log_text, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Proxy training failed for {name}. See {log_path}")
    parsed = parse_train_log(log_text)
    parsed["log_path"] = str(log_path)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Proxy-validate tokenizer candidates with a fixed training recipe")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_LOCAL_DATASET))
    parser.add_argument("--baseline-tokenizer", default=str(DEFAULT_LOCAL_TOKENIZER))
    parser.add_argument("--history-path", default=str(ROOT / "autoresearch" / "tokenizer_discovery" / "history.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-model", action="append", default=[])
    parser.add_argument("--tokenmonster-vocab", action="append", default=[])
    parser.add_argument("--train-sample-tokens", type=int, default=2_000_000)
    parser.add_argument("--val-sample-tokens", type=int, default=500_000)
    parser.add_argument("--train-max-chunks", type=int, default=400)
    parser.add_argument("--val-max-chunks", type=int, default=120)
    parser.add_argument("--max-wallclock-seconds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-json", default=str(OUT_DIR / "summary.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    baseline_tokenizer = Path(args.baseline_tokenizer).expanduser().resolve()
    history_path = Path(args.history_path).expanduser().resolve()
    train_text = OUT_DIR / "train_text_sample.jsonl"
    val_text = OUT_DIR / "val_text_sample.jsonl"

    ensure_text_sample("train", train_text, dataset_dir, baseline_tokenizer, args.train_sample_tokens)
    ensure_text_sample("val", val_text, dataset_dir, baseline_tokenizer, args.val_sample_tokens)

    candidate_models = [Path(p).expanduser().resolve() for p in args.candidate_model]
    if history_path.exists():
        candidate_models.extend(parse_history_candidates(history_path, args.top_k))
    baseline_name = "baseline_sp1024"
    all_candidates: list[dict[str, object]] = [
        {
            "name": baseline_name,
            "family": "sentencepiece",
            "tokenizer_ref": str(baseline_tokenizer),
        }
    ]
    seen = {("sentencepiece", str(baseline_tokenizer))}
    for model_path in candidate_models:
        key = ("sentencepiece", str(model_path))
        if key in seen:
            continue
        seen.add(key)
        label = f"{model_path.parent.name}_{model_path.stem}"
        all_candidates.append(
            {
                "name": label,
                "family": "sentencepiece",
                "tokenizer_ref": str(model_path),
            }
        )
    for vocab_ref in args.tokenmonster_vocab:
        key = ("tokenmonster", vocab_ref)
        if key in seen:
            continue
        seen.add(key)
        all_candidates.append(
            {
                "name": sanitize_name(vocab_ref),
                "family": "tokenmonster",
                "tokenizer_ref": vocab_ref,
            }
        )

    summary_rows: list[dict[str, object]] = []
    for candidate in all_candidates:
        name = str(candidate["name"])
        family = str(candidate["family"])
        tokenizer_ref = str(candidate["tokenizer_ref"])
        if family == "sentencepiece":
            meta_path = export_sentencepiece_meta(Path(tokenizer_ref))
            vocab_size = int(json.loads(run_cmd([
                sys.executable,
                "-c",
                (
                    "import json, sentencepiece as spm, sys; "
                    "sp=spm.SentencePieceProcessor(model_file=sys.argv[1]); "
                    "print(json.dumps({'vocab_size': int(sp.vocab_size())}))"
                ),
                tokenizer_ref,
            ]))["vocab_size"])
            tokenizer_path_for_train = tokenizer_ref
        elif family == "tokenmonster":
            meta_path = export_tokenmonster_meta(tokenizer_ref, name)
            vocab_size = int(json.loads(run_cmd([
                sys.executable,
                "-c",
                (
                    "import json, tokenmonster, sys; "
                    "v=tokenmonster.load(sys.argv[1]); "
                    "print(json.dumps({'vocab_size': int(v.vocab_size)}))"
                ),
                tokenizer_ref,
            ]))["vocab_size"])
            tokenizer_path_for_train = tokenizer_ref
        else:
            raise ValueError(f"Unsupported tokenizer family: {family}")
        proxy_dataset = build_proxy_dataset(
            name,
            family,
            tokenizer_ref,
            train_text,
            val_text,
            train_chunks=args.train_max_chunks,
            val_chunks=args.val_max_chunks,
        )
        if args.prepare_only:
            metrics = {
                "prepared_only": True,
                "log_path": None,
            }
        else:
            metrics = run_proxy_train(
                name,
                proxy_dataset,
                tokenizer_path_for_train,
                meta_path,
                vocab_size=vocab_size,
                max_wallclock_seconds=args.max_wallclock_seconds,
                iterations=args.iterations,
                seed=args.seed,
            )
        summary_rows.append(
            {
                "name": name,
                "family": family,
                "tokenizer_path": tokenizer_ref,
                "tokenizer_meta_path": str(meta_path),
                "proxy_dataset": str(proxy_dataset),
                "vocab_size": vocab_size,
                "train_python": TRAIN_PYTHON,
                **metrics,
            }
        )

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
