"""
Focused inner-loop probe for JEPA target embedding research.

This is intentionally much cheaper than full LM training:
- consumes byte260 shards directly
- trains only a tiny patch-level JEPA toy model
- evaluates the target/predictor path on held-out patch pairs

The mutation surface should stay narrow:
- TargetPatchEncoder
- optionally the paired Predictor
"""

from __future__ import annotations

import glob
import math
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_byte260")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 260))
    model_dim = int(os.environ.get("PROBE_MODEL_DIM", 128))
    patch_size = int(os.environ.get("PATCH_SIZE", 8))
    probe_steps = int(os.environ.get("PROBE_STEPS", 400))
    batch_size = int(os.environ.get("PROBE_BATCH_SIZE", 256))
    eval_batches = int(os.environ.get("PROBE_EVAL_BATCHES", 24))
    lr = float(os.environ.get("PROBE_LR", 3e-3))
    target_ema = float(os.environ.get("TARGET_EMA", 0.995))
    jepa_std_target = float(os.environ.get("JEPA_STD_TARGET", 0.5))
    jepa_var_weight = float(os.environ.get("JEPA_VAR_WEIGHT", 0.02))
    log_every = int(os.environ.get("PROBE_LOG_EVERY", 50))
    use_synthetic_data = bool(int(os.environ.get("PROBE_USE_SYNTHETIC_DATA", "0")))
    synthetic_tokens = int(os.environ.get("PROBE_SYNTHETIC_TOKENS", 2_000_000))


def pick_device() -> torch.device:
    forced = os.environ.get("DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.int64, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class SyntheticTokenStream:
    def __init__(self, vocab_size: int, total_tokens: int, seed: int):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        self.tokens = torch.randint(0, vocab_size, (total_tokens,), generator=g, dtype=torch.int64)
        self.pos = 0

    def take(self, n: int) -> Tensor:
        if self.pos + n > self.tokens.numel():
            self.pos = 0
        out = self.tokens[self.pos : self.pos + n]
        self.pos += n
        return out


class CurrentPatchEncoder(nn.Module):
    def __init__(self, model_dim: int, patch_size: int):
        super().__init__()
        self.local_pos = nn.Parameter(torch.zeros(patch_size, model_dim, dtype=torch.float32))
        self.mix = nn.Linear(model_dim, model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.normal_(self.local_pos, mean=0.0, std=0.01)

    def forward(self, patch_byte_emb: Tensor) -> Tensor:
        x = patch_byte_emb + self.local_pos[None, None, :, :].to(dtype=patch_byte_emb.dtype)
        x = F.rms_norm(x, (x.size(-1),))
        x = torch.relu(self.mix(x))
        x = x.mean(dim=2)
        x = self.out(x)
        return F.rms_norm(x, (x.size(-1),))


class TargetPatchEncoder(nn.Module):
    """
    Primary mutation surface for target-embedding autoresearch.

    Keep changes narrowly focused here unless a proposal explicitly requires
    a matched update to Predictor.
    """

    def __init__(self, model_dim: int, patch_size: int):
        super().__init__()
        self.local_pos = nn.Parameter(torch.zeros(patch_size, model_dim, dtype=torch.float32))
        self.in_proj = nn.Linear(model_dim, model_dim, bias=False)
        # Learned attention pooling: project each byte to a scalar weight
        self.attn_pool = nn.Linear(model_dim, 1, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.normal_(self.local_pos, mean=0.0, std=0.01)

    def forward(self, patch_byte_emb: Tensor) -> Tensor:
        # patch_byte_emb: (B, num_patches, patch_size, model_dim)
        x = patch_byte_emb + self.local_pos[None, None, :, :].to(dtype=patch_byte_emb.dtype)
        x = F.rms_norm(x, (x.size(-1),))
        x = torch.relu(self.in_proj(x))
        # Attention-weighted pooling over byte dimension
        attn_logits = self.attn_pool(x).squeeze(-1)  # (B, P, S)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, P, S)
        x = (x * attn_weights.unsqueeze(-1)).sum(dim=2)  # (B, P, D)
        x = self.out_proj(x)
        return F.rms_norm(x, (x.size(-1),))


class Predictor(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, model_dim, bias=False)
        self.fc2 = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = F.rms_norm(x, (x.size(-1),))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x.square())
        return F.rms_norm(x, (x.size(-1),))


class ProbeModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.patch_size = args.patch_size
        self.jepa_std_target = args.jepa_std_target
        self.jepa_var_weight = args.jepa_var_weight
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.online_encoder = CurrentPatchEncoder(args.model_dim, args.patch_size)
        self.target_online_encoder = TargetPatchEncoder(args.model_dim, args.patch_size)
        self.predictor = Predictor(args.model_dim)
        self.target_tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.target_encoder = TargetPatchEncoder(args.model_dim, args.patch_size)

    @torch.no_grad()
    def sync_targets(self) -> None:
        self.target_tok_emb.weight.copy_(self.tok_emb.weight.detach().float())
        self.target_encoder.load_state_dict(self.target_online_encoder.state_dict(), strict=True)

    @torch.no_grad()
    def update_targets_ema(self, decay: float) -> None:
        self.target_tok_emb.weight.mul_(decay).add_(self.tok_emb.weight.detach().float(), alpha=1.0 - decay)
        for target_param, online_param in zip(self.target_encoder.parameters(), self.target_online_encoder.parameters(), strict=True):
            target_param.data.mul_(decay).add_(online_param.detach().float(), alpha=1.0 - decay)

    def encode_current(self, patches: Tensor) -> Tensor:
        return self.online_encoder(self.tok_emb(patches))

    def encode_next_online(self, patches: Tensor) -> Tensor:
        return self.target_online_encoder(self.tok_emb(patches))

    def encode_next_target(self, patches: Tensor) -> Tensor:
        with torch.no_grad():
            return self.target_encoder(self.target_tok_emb(patches))

    def loss_on_pairs(self, current_patches: Tensor, next_patches: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        current_latent = self.encode_current(current_patches)
        _ = self.encode_next_online(next_patches)  # keeps target-online path active and trainable
        pred = self.predictor(current_latent)
        target = self.encode_next_target(next_patches).float()
        pred_f = pred.float()
        pred_n = F.normalize(pred_f, dim=-1)
        target_n = F.normalize(target, dim=-1)
        mse = F.mse_loss(pred_n, target_n, reduction="mean")
        cosine = (pred_n * target_n).sum(dim=-1).mean()
        pred_std = pred_f.std(dim=(0, 1), correction=0)
        target_std = target.std(dim=(0, 1), correction=0)
        var_loss = torch.relu(self.jepa_std_target - pred_std).mean()
        total = mse + self.jepa_var_weight * var_loss
        aux = {
            "mse": mse.detach(),
            "cosine": cosine.detach(),
            "pred_std": pred_std.mean().detach(),
            "target_std": target_std.mean().detach(),
            "var_loss": var_loss.detach(),
        }
        return total, aux


def make_patch_pairs(stream, batch_size: int, patch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    chunk = stream.take(batch_size * patch_size * 2)
    patches = chunk.reshape(batch_size, 2, patch_size)
    current = patches[:, 0, :].to(device=device, dtype=torch.long)
    nxt = patches[:, 1, :].to(device=device, dtype=torch.long)
    return current.unsqueeze(1), nxt.unsqueeze(1)


@torch.no_grad()
def evaluate_probe(model: ProbeModel, stream, args: Hyperparameters, device: torch.device) -> dict[str, float]:
    model.eval()
    mse_sum = 0.0
    cos_sum = 0.0
    retrieval_sum = 0.0
    pred_std_sum = 0.0
    target_std_sum = 0.0
    shuffle_gap_sum = 0.0
    count = 0
    for _ in range(args.eval_batches):
        current_patches, next_patches = make_patch_pairs(stream, args.batch_size, args.patch_size, device)
        current_latent = model.encode_current(current_patches)
        pred = model.predictor(current_latent).float().reshape(-1, args.model_dim)
        target = model.encode_next_target(next_patches).float().reshape(-1, args.model_dim)
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        mse = F.mse_loss(pred_n, target_n, reduction="mean").item()
        cosine = (pred_n * target_n).sum(dim=-1).mean().item()
        sim = pred_n @ target_n.T
        retrieval = (sim.argmax(dim=1) == torch.arange(sim.size(0), device=sim.device)).float().mean().item()
        pred_std = pred.std(dim=0, correction=0).mean().item()
        target_std = target.std(dim=0, correction=0).mean().item()
        shuffled = next_patches.clone()
        perm = torch.randperm(args.patch_size, device=device)
        shuffled = shuffled[:, :, perm]
        target_shuf = model.encode_next_target(shuffled).float().reshape(-1, args.model_dim)
        target_shuf_n = F.normalize(target_shuf, dim=-1)
        shuffle_gap = (1.0 - (target_n * target_shuf_n).sum(dim=-1).mean().item())
        mse_sum += mse
        cos_sum += cosine
        retrieval_sum += retrieval
        pred_std_sum += pred_std
        target_std_sum += target_std
        shuffle_gap_sum += shuffle_gap
        count += 1
    model.train()
    mse = mse_sum / count
    cosine = cos_sum / count
    retrieval = retrieval_sum / count
    pred_std = pred_std_sum / count
    target_std = target_std_sum / count
    shuffle_gap = shuffle_gap_sum / count
    collapse_penalty = max(args.jepa_std_target - pred_std, 0.0)
    probe_score = mse + 0.25 * (1.0 - retrieval) + 0.10 * collapse_penalty
    return {
        "val_loss": mse,
        "probe_score": probe_score,
        "val_cosine": cosine,
        "retrieval_at_1": retrieval,
        "pred_std": pred_std,
        "target_std": target_std,
        "shuffle_gap": shuffle_gap,
        "collapse_penalty": collapse_penalty,
    }


def main() -> None:
    args = Hyperparameters()
    device = pick_device()
    logfile = None
    if os.environ.get("RANK", "0") == "0":
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str) -> None:
        if os.environ.get("RANK", "0") != "0":
            return
        print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.use_synthetic_data:
        train_stream = SyntheticTokenStream(args.vocab_size, args.synthetic_tokens, args.seed)
        val_stream = SyntheticTokenStream(args.vocab_size, max(args.synthetic_tokens // 4, args.batch_size * args.patch_size * 4), args.seed + 1)
        train_files = 0
        val_files = 0
    else:
        train_stream = TokenStream(args.train_files)
        val_stream = TokenStream(args.val_files)
        train_files = len(list(Path(args.data_path).glob("fineweb_train_*.bin")))
        val_files = len(list(Path(args.data_path).glob("fineweb_val_*.bin")))

    model = ProbeModel(args).to(device)
    model.sync_targets()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.tok_emb.parameters(), "lr": args.lr},
            {"params": model.online_encoder.parameters(), "lr": args.lr},
            {"params": model.target_online_encoder.parameters(), "lr": args.lr},
            {"params": model.predictor.parameters(), "lr": args.lr},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )

    log0(f"device:{device.type}")
    log0(f"probe_family:jepa_target_embedding_inner_loop patch_size:{args.patch_size} model_dim:{args.model_dim}")
    log0(f"data_path:{args.data_path} synthetic:{args.use_synthetic_data} train_shards:{train_files} val_shards:{val_files}")
    log0(f"probe_steps:{args.probe_steps} batch_size:{args.batch_size} eval_batches:{args.eval_batches} lr:{args.lr}")

    t0 = time.perf_counter()
    for step in range(1, args.probe_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        current_patches, next_patches = make_patch_pairs(train_stream, args.batch_size, args.patch_size, device)
        loss, aux = model.loss_on_pairs(current_patches, next_patches)
        loss.backward()
        optimizer.step()
        model.update_targets_ema(args.target_ema)
        if step <= 10 or step % args.log_every == 0:
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            patch_pairs_seen = step * args.batch_size
            pairs_per_s = patch_pairs_seen / max(elapsed_ms / 1000.0, 1e-9)
            log0(
                f"step:{step}/{args.probe_steps} train_loss:{loss.item():.6f} "
                f"mse:{aux['mse'].item():.6f} cosine:{aux['cosine'].item():.6f} "
                f"pred_std:{aux['pred_std'].item():.6f} target_std:{aux['target_std'].item():.6f} "
                f"pairs_per_s:{pairs_per_s:.1f}"
            )

    metrics = evaluate_probe(model, val_stream, args, device)
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log0(
        f"DIAGNOSTIC probe_summary val_cosine:{metrics['val_cosine']:.8f} "
        f"retrieval_at_1:{metrics['retrieval_at_1']:.8f} pred_std:{metrics['pred_std']:.8f} "
        f"target_std:{metrics['target_std']:.8f} shuffle_gap:{metrics['shuffle_gap']:.8f} "
        f"collapse_penalty:{metrics['collapse_penalty']:.8f}"
    )
    if device.type == "cuda":
        log0(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
        )
    log0(
        f"final_int8_zlib_roundtrip val_loss:{metrics['val_loss']:.4f} val_bpb:{metrics['probe_score']:.4f} "
        f"eval_time:{elapsed_ms:.0f}ms"
    )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss:{metrics['val_loss']:.8f} "
        f"val_bpb:{metrics['probe_score']:.8f}"
    )


if __name__ == "__main__":
    main()
