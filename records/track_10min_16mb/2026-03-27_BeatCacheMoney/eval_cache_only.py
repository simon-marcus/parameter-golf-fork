#!/usr/bin/env python3
"""Standalone cache-only eval — no GPU, no PyTorch, no model needed.

Loads pre-computed Pass 1 arrays (model_p, entropy, positions, val_tokens)
and runs n-gram/phrase cache build + calibration + rescore.

Usage:
    python3 eval_cache_only.py --pass1 /path/to/pass1_arrays.npz [options]

    # Or via env vars:
    PASS1_PATH=pass1_arrays.npz NGRAM_NUM_BUCKETS=4194304 python3 eval_cache_only.py

Designed for fast CPU-only iteration (~60s per experiment).
"""
from __future__ import annotations
import argparse
import math
import os
import time
import numpy as np


# === HASH PRIMES ===

_NGRAM_PRIMES = np.array([
    36313, 27191, 51647, 81929, 131071, 174763, 233017, 283721,
    347237, 411527, 479909, 557927, 646333, 746773, 862319, 992353,
    1143449, 1301081, 1497311, 1721381,
], dtype=np.int64)

_PHRASE_PRIMES = np.array([
    36313, 27191, 51647, 81929, 131071, 174763, 233017, 283721,
    347237, 411527, 479909, 557927, 646333, 746773, 862319, 992353,
    1143449, 1301081, 1497311, 1721381, 1984703, 2280179, 2621471, 2995931,
    3444107, 3969403, 4560379, 5242877, 5989997, 6888191, 7938127, 9120757,
    10485767, 11980039, 13776427, 15876269, 18241517, 20971529, 23960023, 27552851,
    31752553, 36483043, 41943049, 47920069, 55105663, 63505117, 72966089, 83886103,
    95840137, 110211271, 127010227, 145932179, 167772161, 191680267, 220422547, 254020477,
    291864389, 335544323, 383360519, 440845147, 508040969, 583728791, 671088667, 766721999,
], dtype=np.int64)

_ORDER_MULTS = np.array([
    0.30, 0.30, 0.97, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
], dtype=np.float32)


# === N-GRAM CACHE ===

class NgramCache:
    def __init__(self, min_order: int = 2, max_order: int = 12, num_buckets: int = 4_194_304):
        self.min_order = min_order
        self.max_order = max_order
        self.num_orders = max_order - min_order + 1
        self.num_buckets = num_buckets
        self.bucket_mask = np.int64(num_buckets - 1)
        self.ctx_tables = [np.zeros(num_buckets, dtype=np.int32) for _ in range(self.num_orders)]
        self.full_tables = [np.zeros(num_buckets, dtype=np.int32) for _ in range(self.num_orders)]

    def _compute_hashes(self, tokens_np, start, end, order_idx):
        n = self.min_order + order_idx
        valid_start = max(start, n - 1)
        N = end - valid_start
        if N <= 0:
            return None, None, valid_start
        h = np.zeros(N, dtype=np.int64)
        for k in range(n - 1):
            offset = valid_start - (n - 1) + k
            h ^= tokens_np[offset:offset + N].astype(np.int64) * _NGRAM_PRIMES[k % len(_NGRAM_PRIMES)]
        ctx_h = h & self.bucket_mask
        target_prime = _NGRAM_PRIMES[min(n - 1, len(_NGRAM_PRIMES) - 1)]
        full_h = (h ^ (tokens_np[valid_start:end].astype(np.int64) * target_prime)) & self.bucket_mask
        return ctx_h, full_h, valid_start

    def _bincount_add(self, table, indices):
        counts = np.bincount(indices.astype(np.intp), minlength=self.num_buckets)
        table += counts[:self.num_buckets].astype(table.dtype)

    def build_full(self, tokens_np):
        for oi in range(self.num_orders):
            ctx_h, full_h, _ = self._compute_hashes(tokens_np, 0, len(tokens_np), oi)
            if ctx_h is None:
                continue
            self._bincount_add(self.ctx_tables[oi], ctx_h)
            self._bincount_add(self.full_tables[oi], full_h)

    def score_positions(self, tokens_np, positions, min_count=2, leave_one_out=False):
        N = len(positions)
        ngram_prob = np.zeros(N, dtype=np.float32)
        matched_order = np.full(N, -1, dtype=np.int32)
        matched = np.zeros(N, dtype=bool)
        if N == 0:
            return ngram_prob, matched_order
        positions = positions.astype(np.int64, copy=False)
        for oi in range(self.num_orders - 1, -1, -1):
            n = self.min_order + oi
            ctx_h_all, full_h_all, valid_start = self._compute_hashes(tokens_np, 0, len(tokens_np), oi)
            if ctx_h_all is None:
                continue
            remaining_idx = np.where(~matched)[0]
            if remaining_idx.size == 0:
                break
            pos_sub = positions[remaining_idx]
            valid_mask = pos_sub >= valid_start
            if not np.any(valid_mask):
                continue
            valid_idx = remaining_idx[valid_mask]
            lookup = (pos_sub[valid_mask] - valid_start).astype(np.int64)
            ctx_h = ctx_h_all[lookup]
            full_h = full_h_all[lookup]
            ctx_counts = self.ctx_tables[oi][ctx_h].astype(np.int64)
            full_counts = self.full_tables[oi][full_h].astype(np.int64)
            if leave_one_out:
                ctx_counts = np.maximum(ctx_counts - 1, 0)
                full_counts = np.maximum(full_counts - 1, 0)
            full_counts = np.minimum(full_counts, ctx_counts)
            eligible = (ctx_counts >= min_count) & (full_counts > 0)
            if not np.any(eligible):
                continue
            out_idx = valid_idx[eligible]
            prob = full_counts[eligible].astype(np.float32) / np.maximum(ctx_counts[eligible].astype(np.float32), 1.0)
            ngram_prob[out_idx] = prob
            matched_order[out_idx] = n
            matched[out_idx] = True
        return ngram_prob, matched_order


# === PHRASE CACHE ===

class PhraseCache:
    def __init__(self, probe_lengths: list[int], num_buckets: int = 4_194_304):
        self.probe_lengths = sorted({int(v) for v in probe_lengths if int(v) > 1}, reverse=True)
        self.num_buckets = num_buckets
        self.bucket_mask = np.int64(num_buckets - 1)
        self.ctx_tables = {L: np.zeros(num_buckets, dtype=np.int32) for L in self.probe_lengths}
        self.full_tables = {L: np.zeros(num_buckets, dtype=np.int32) for L in self.probe_lengths}

    def _bincount_add(self, table, indices):
        counts = np.bincount(indices.astype(np.intp), minlength=self.num_buckets)
        table += counts[:self.num_buckets].astype(table.dtype)

    def build_full(self, tokens_np):
        for length in self.probe_lengths:
            valid_start = length
            N = len(tokens_np) - valid_start
            if N <= 0:
                continue
            h = np.zeros(N, dtype=np.int64)
            for k in range(length):
                offset = valid_start - length + k
                h ^= tokens_np[offset:offset + N].astype(np.int64) * _PHRASE_PRIMES[k % len(_PHRASE_PRIMES)]
            ctx_h = h & self.bucket_mask
            target_prime = _PHRASE_PRIMES[length % len(_PHRASE_PRIMES)]
            full_h = (h ^ (tokens_np[valid_start:len(tokens_np)].astype(np.int64) * target_prime)) & self.bucket_mask
            self._bincount_add(self.ctx_tables[length], ctx_h)
            self._bincount_add(self.full_tables[length], full_h)

    def score_positions(self, tokens_np, positions, min_count=1, leave_one_out=False):
        N = len(positions)
        phrase_prob = np.zeros(N, dtype=np.float32)
        matched_len = np.zeros(N, dtype=np.int32)
        matched = np.zeros(N, dtype=bool)
        if N == 0:
            return phrase_prob, matched_len
        positions = positions.astype(np.int64, copy=False)
        for length in self.probe_lengths:
            remaining_idx = np.where(~matched)[0]
            if remaining_idx.size == 0:
                break
            pos_sub = positions[remaining_idx]
            valid_mask = pos_sub >= length
            if not np.any(valid_mask):
                continue
            valid_idx = remaining_idx[valid_mask]
            pos_valid = pos_sub[valid_mask]
            h = np.zeros(len(pos_valid), dtype=np.int64)
            for k in range(length):
                h ^= tokens_np[pos_valid - length + k].astype(np.int64) * _PHRASE_PRIMES[k % len(_PHRASE_PRIMES)]
            ctx_h = h & self.bucket_mask
            target_prime = _PHRASE_PRIMES[length % len(_PHRASE_PRIMES)]
            full_h = (h ^ (tokens_np[pos_valid].astype(np.int64) * target_prime)) & self.bucket_mask
            ctx_counts = self.ctx_tables[length][ctx_h].astype(np.int64)
            full_counts = self.full_tables[length][full_h].astype(np.int64)
            if leave_one_out:
                ctx_counts = np.maximum(ctx_counts - 1, 0)
                full_counts = np.maximum(full_counts - 1, 0)
            full_counts = np.minimum(full_counts, ctx_counts)
            eligible = (ctx_counts >= min_count) & (full_counts > 0)
            if not np.any(eligible):
                continue
            out_idx = valid_idx[eligible]
            prob = full_counts[eligible].astype(np.float32) / np.maximum(ctx_counts[eligible].astype(np.float32), 1.0)
            phrase_prob[out_idx] = prob
            matched_len[out_idx] = length
            matched[out_idx] = True
        return phrase_prob, matched_len


# === BLEND + CALIBRATE ===

def blend_cache_probs(
    p_model, entropy, ngram_prob, matched_order,
    phrase_prob, matched_len, phrase_probe_lengths,
    ngram_alpha_max, ngram_alpha_min, ngram_entropy_center, ngram_entropy_scale,
    phrase_alpha_min, phrase_alpha_max, phrase_entropy_center, phrase_entropy_scale,
    ngram_min_order,
):
    p_blend = p_model.copy()
    matched = matched_order >= 0
    if np.any(matched):
        order_idx = (matched_order[matched] - ngram_min_order).astype(np.int32)
        centers = ngram_entropy_center - 0.25 * order_idx.astype(np.float32)
        sig = 1.0 / (1.0 + np.exp(-ngram_entropy_scale * (entropy[matched] - centers)))
        raw_alpha = ngram_alpha_min + (ngram_alpha_max - ngram_alpha_min) * sig
        mults = _ORDER_MULTS[np.minimum(order_idx, len(_ORDER_MULTS) - 1)]
        alpha = np.clip(raw_alpha * mults, 0.0, 0.98)
        p_blend[matched] = (1.0 - alpha) * p_blend[matched] + alpha * ngram_prob[matched]

    if phrase_prob is not None and matched_len is not None:
        phrase_matched = matched_len > 0
        if np.any(phrase_matched) and phrase_probe_lengths:
            len_floor = min(phrase_probe_lengths)
            len_ceil = max(phrase_probe_lengths)
            denom = max(len_ceil - len_floor, 1)
            len_frac = (matched_len[phrase_matched].astype(np.float32) - len_floor) / denom
            sig = 1.0 / (1.0 + np.exp(
                -phrase_entropy_scale * (entropy[phrase_matched] - phrase_entropy_center)
            ))
            alpha = phrase_alpha_min + (phrase_alpha_max - phrase_alpha_min) * len_frac
            alpha = np.clip(alpha * (0.5 + 0.5 * sig), 0.0, 0.999)
            p_blend[phrase_matched] = (
                (1.0 - alpha) * p_blend[phrase_matched] + alpha * phrase_prob[phrase_matched]
            )

    return np.clip(p_blend, 1e-10, 1.0)


def calibrate(
    p_model, entropy, ngram_prob, matched_order,
    phrase_prob, matched_len, phrase_probe_lengths,
    cfg,
):
    alpha_grid = [float(x) for x in cfg["alpha_max_grid"].split(",")]
    center_grid = [float(x) for x in cfg["entropy_center_grid"].split(",")]
    phrase_alpha_grid = [float(x) for x in cfg["phrase_alpha_max_grid"].split(",")]
    if phrase_prob is None:
        phrase_alpha_grid = [cfg["phrase_alpha_max"]]

    best = (cfg["ngram_alpha_max"], cfg["ngram_entropy_center"], cfg["phrase_alpha_max"])
    best_nll = float("inf")
    for am in alpha_grid:
        for ec in center_grid:
            for pa in phrase_alpha_grid:
                p = blend_cache_probs(
                    p_model, entropy, ngram_prob, matched_order,
                    phrase_prob, matched_len, phrase_probe_lengths,
                    am, cfg["ngram_alpha_min"], ec, cfg["ngram_entropy_scale"],
                    cfg["phrase_alpha_min"], pa,
                    cfg["phrase_entropy_center"], cfg["phrase_entropy_scale"],
                    cfg["ngram_min_order"],
                )
                nll = float((-np.log(p)).mean())
                if nll < best_nll:
                    best_nll = nll
                    best = (am, ec, pa)

    print(f"calibration: grid={len(alpha_grid)}x{len(center_grid)}x{len(phrase_alpha_grid)} "
          f"best_alpha_max={best[0]:.3f} best_center={best[1]:.1f} "
          f"best_phrase_max={best[2]:.3f} mean_nll={best_nll:.6f}")
    return best


# === MAIN ===

def main():
    p = argparse.ArgumentParser(description="Cache-only eval (CPU, no GPU)")
    p.add_argument("--pass1", default=os.environ.get("PASS1_PATH", "pass1_arrays.npz"))
    p.add_argument("--ngram-min-order", type=int, default=int(os.environ.get("NGRAM_MIN_ORDER", "2")))
    p.add_argument("--ngram-max-order", type=int, default=int(os.environ.get("NGRAM_MAX_ORDER", "12")))
    p.add_argument("--ngram-num-buckets", type=int, default=int(os.environ.get("NGRAM_NUM_BUCKETS", "4194304")))
    p.add_argument("--ngram-min-count", type=int, default=int(os.environ.get("NGRAM_MIN_COUNT", "2")))
    p.add_argument("--ngram-alpha-min", type=float, default=float(os.environ.get("NGRAM_ALPHA_MIN", "0.05")))
    p.add_argument("--ngram-alpha-max", type=float, default=float(os.environ.get("NGRAM_ALPHA_MAX", "0.80")))
    p.add_argument("--ngram-entropy-center", type=float, default=float(os.environ.get("NGRAM_ENTROPY_CENTER", "3.0")))
    p.add_argument("--ngram-entropy-scale", type=float, default=float(os.environ.get("NGRAM_ENTROPY_SCALE", "2.0")))
    p.add_argument("--leave-one-out", type=int, default=int(os.environ.get("NGRAM_LEAVE_ONE_OUT", "1")))
    p.add_argument("--phrase-enabled", type=int, default=int(os.environ.get("PHRASE_ENABLED", "0")))
    p.add_argument("--phrase-probe-lengths", default=os.environ.get("PHRASE_PROBE_LENGTHS", "64,56,48,36,28,20,16"))
    p.add_argument("--phrase-num-buckets", type=int, default=int(os.environ.get("PHRASE_NUM_BUCKETS", "4194304")))
    p.add_argument("--phrase-min-count", type=int, default=int(os.environ.get("PHRASE_MIN_COUNT", "1")))
    p.add_argument("--phrase-alpha-min", type=float, default=float(os.environ.get("PHRASE_ALPHA_MIN", "0.88")))
    p.add_argument("--phrase-alpha-max", type=float, default=float(os.environ.get("PHRASE_ALPHA_MAX", "0.995")))
    p.add_argument("--phrase-entropy-center", type=float, default=float(os.environ.get("PHRASE_ENTROPY_CENTER", "2.5")))
    p.add_argument("--phrase-entropy-scale", type=float, default=float(os.environ.get("PHRASE_ENTROPY_SCALE", "2.0")))
    p.add_argument("--calibration-frac", type=float, default=float(os.environ.get("NGRAM_CALIBRATION_FRAC", "0.05")))
    p.add_argument("--calibration-alpha-grid", default=os.environ.get("NGRAM_CALIBRATION_ALPHA_MAX_GRID", "0.70,0.80,0.90,0.95,0.99"))
    p.add_argument("--calibration-center-grid", default=os.environ.get("NGRAM_CALIBRATION_ENTROPY_CENTER_GRID", "2.0,2.5,3.0,3.5"))
    p.add_argument("--calibration-phrase-grid", default=os.environ.get("PHRASE_CALIBRATION_ALPHA_MAX_GRID", "0.980,0.990,0.995,0.999"))
    args = p.parse_args()

    t0 = time.perf_counter()

    # Load Pass 1 arrays
    print(f"loading {args.pass1}...")
    d = np.load(args.pass1)
    model_p = d["model_p"]
    entropy = d["entropy"]
    token_bytes = d["token_bytes"]
    positions = d["positions"]
    pass1_bpb = float(d["pass1_bpb"])
    tokens_np = d["val_tokens"]
    print(f"loaded: {len(positions)} scored positions, {len(tokens_np)} val tokens, pass1_bpb={pass1_bpb:.6f}")

    # Build n-gram cache
    t1 = time.perf_counter()
    print(f"building n-gram cache orders={args.ngram_min_order}-{args.ngram_max_order} buckets={args.ngram_num_buckets}")
    cache = NgramCache(args.ngram_min_order, args.ngram_max_order, args.ngram_num_buckets)
    cache.build_full(tokens_np)
    t2 = time.perf_counter()
    print(f"n-gram cache built in {t2-t1:.1f}s")

    # Build phrase cache
    phrase_cache = None
    phrase_prob = None
    matched_len = None
    phrase_probe_lengths = []
    if args.phrase_enabled:
        phrase_probe_lengths = [int(x) for x in args.phrase_probe_lengths.split(",") if x.strip()]
        print(f"building phrase cache lengths={phrase_probe_lengths} buckets={args.phrase_num_buckets}")
        phrase_cache = PhraseCache(phrase_probe_lengths, args.phrase_num_buckets)
        phrase_cache.build_full(tokens_np)
        t3 = time.perf_counter()
        print(f"phrase cache built in {t3-t2:.1f}s")

    # Score n-gram
    ngram_prob, matched_order = cache.score_positions(
        tokens_np, positions,
        min_count=args.ngram_min_count,
        leave_one_out=bool(args.leave_one_out),
    )
    n_ng = int((matched_order >= 0).sum())
    print(f"ngram matched: {n_ng}/{len(positions)} ({100*n_ng/max(len(positions),1):.1f}%)")

    # Score phrase
    n_ph = 0
    if phrase_cache is not None:
        phrase_prob, matched_len = phrase_cache.score_positions(
            tokens_np, positions,
            min_count=args.phrase_min_count,
            leave_one_out=bool(args.leave_one_out),
        )
        n_ph = int((matched_len > 0).sum())
        print(f"phrase matched: {n_ph}/{len(positions)} ({100*n_ph/max(len(positions),1):.1f}%)")

    # Calibrate
    N = len(positions)
    cal_n = max(100, int(N * args.calibration_frac))
    cfg = {
        "ngram_alpha_max": args.ngram_alpha_max,
        "ngram_alpha_min": args.ngram_alpha_min,
        "ngram_entropy_center": args.ngram_entropy_center,
        "ngram_entropy_scale": args.ngram_entropy_scale,
        "phrase_alpha_min": args.phrase_alpha_min,
        "phrase_alpha_max": args.phrase_alpha_max,
        "phrase_entropy_center": args.phrase_entropy_center,
        "phrase_entropy_scale": args.phrase_entropy_scale,
        "ngram_min_order": args.ngram_min_order,
        "alpha_max_grid": args.calibration_alpha_grid,
        "entropy_center_grid": args.calibration_center_grid,
        "phrase_alpha_max_grid": args.calibration_phrase_grid,
    }
    best_am, best_ec, best_pa = calibrate(
        model_p[:cal_n], entropy[:cal_n],
        ngram_prob[:cal_n], matched_order[:cal_n],
        phrase_prob[:cal_n] if phrase_prob is not None else None,
        matched_len[:cal_n] if matched_len is not None else None,
        phrase_probe_lengths, cfg,
    )

    # Final rescore with calibrated params
    p_blend = blend_cache_probs(
        model_p, entropy, ngram_prob, matched_order,
        phrase_prob, matched_len, phrase_probe_lengths,
        best_am, args.ngram_alpha_min, best_ec, args.ngram_entropy_scale,
        args.phrase_alpha_min, best_pa,
        args.phrase_entropy_center, args.phrase_entropy_scale,
        args.ngram_min_order,
    )

    nll = -np.log(p_blend).astype(np.float64)
    val_loss = nll.sum() / len(nll)
    val_bpb = val_loss / math.log(2.0) * (float(len(nll)) / token_bytes.sum())

    t_end = time.perf_counter()
    print(f"val_bpb={val_bpb:.8f} improvement={pass1_bpb - val_bpb:.6f} "
          f"cal_alpha_max={best_am:.3f} cal_center={best_ec:.1f} cal_phrase_max={best_pa:.3f}")
    print(f"total time: {t_end-t0:.1f}s")
    # Machine-readable output line (for autoresearch parsing)
    print(f"RESULT val_bpb={val_bpb:.8f}")


if __name__ == "__main__":
    main()
