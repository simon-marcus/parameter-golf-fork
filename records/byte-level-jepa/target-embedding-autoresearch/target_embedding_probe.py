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
    probe_steps = int(os.environ.get("PROBE_STEPS", 750))
    batch_size = int(os.environ.get("PROBE_BATCH_SIZE", 256))
    eval_batches = int(os.environ.get("PROBE_EVAL_BATCHES", 24))
    lr = float(os.environ.get("PROBE_LR", 3e-3))
    target_ema_start = float(os.environ.get("TARGET_EMA_START", 0.996))
    target_ema_end = float(os.environ.get("TARGET_EMA_END", 0.999))
    jepa_std_target = float(os.environ.get("JEPA_STD_TARGET", 0.5))
    jepa_var_weight = float(os.environ.get("JEPA_VAR_WEIGHT", 0.02))
    jepa_cov_weight = float(os.environ.get("JEPA_COV_WEIGHT", 0.01))
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
        self.gate = nn.Linear(model_dim, model_dim, bias=False)
        # Depthwise bidirectional conv to capture local byte bigram/trigram patterns,
        # matching the pattern proven effective in TargetPatchEncoder (experiment #20).
        # Symmetric padding since the current encoder also sees the full patch.
        self.local_conv = nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2, groups=model_dim, bias=False)
        # Intra-patch self-attention (4 heads, bidirectional) — mirrors the pattern
        # that improved TargetPatchEncoder in #20. Enriches current-patch representations
        # fed to the Predictor by capturing non-local byte interactions (e.g. byte 0
        # attending to byte 7) that the k=5 conv cannot reach in patch_size=8.
        self.sa_num_heads = 4
        self.sa_head_dim = model_dim // self.sa_num_heads
        self.sa_q = nn.Linear(model_dim, model_dim, bias=False)
        self.sa_k = nn.Linear(model_dim, model_dim, bias=False)
        self.sa_v = nn.Linear(model_dim, model_dim, bias=False)
        self.sa_out = nn.Linear(model_dim, model_dim, bias=False)
        self.sa_scale = self.sa_head_dim ** -0.5
        # Learned relative position bias for SA (range -(patch_size-1) to +(patch_size-1))
        self.sa_rel_pos_bias = nn.Parameter(torch.zeros(self.sa_num_heads, 2 * patch_size - 1))
        # Attention pooling: 4 heads, matching target encoder style
        self.num_pool_heads = 4
        self.attn_pool = nn.Linear(model_dim, self.num_pool_heads, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.normal_(self.local_pos, mean=0.0, std=0.01)

    def forward(self, patch_byte_emb: Tensor) -> Tensor:
        # patch_byte_emb: (B, num_patches, patch_size, model_dim)
        x = patch_byte_emb + self.local_pos[None, None, :, :].to(dtype=patch_byte_emb.dtype)
        x = F.rms_norm(x, (x.size(-1),))
        x = F.silu(self.gate(x)) * self.mix(x)  # SwiGLU gating
        # Apply depthwise bidirectional conv over byte dimension with residual
        B, P, S, D = x.shape
        x_flat = x.reshape(B * P, S, D).transpose(1, 2)  # (B*P, D, S)
        conv_out = self.local_conv(x_flat)  # bidirectional: symmetric padding
        x = x + conv_out.transpose(1, 2).reshape(B, P, S, D)  # residual
        x = F.rms_norm(x, (x.size(-1),))
        # Intra-patch multi-head self-attention (4 heads, 8x8 per head — very cheap)
        B, P, S, D = x.shape
        H = self.sa_num_heads
        Dh = self.sa_head_dim
        q = self.sa_q(x).view(B, P, S, H, Dh).permute(0, 1, 3, 2, 4)  # (B, P, H, S, Dh)
        k = self.sa_k(x).view(B, P, S, H, Dh).permute(0, 1, 3, 2, 4)
        v = self.sa_v(x).view(B, P, S, H, Dh).permute(0, 1, 3, 2, 4)
        sa_logits = torch.einsum('bphsd,bphtd->bphst', q, k) * self.sa_scale  # (B, P, H, S, S)
        # Add learned relative position bias
        positions = torch.arange(S, device=sa_logits.device)
        rel_idx = positions.unsqueeze(1) - positions.unsqueeze(0) + (S - 1)  # (S, S)
        sa_logits = sa_logits + self.sa_rel_pos_bias[:, rel_idx].to(dtype=sa_logits.dtype)
        sa_weights = F.softmax(sa_logits, dim=-1)
        sa_out = torch.einsum('bphst,bphtd->bphsd', sa_weights, v)  # (B, P, H, S, Dh)
        sa_out = sa_out.permute(0, 1, 3, 2, 4).reshape(B, P, S, D)  # (B, P, S, D)
        x = x + self.sa_out(sa_out)  # residual
        x = F.rms_norm(x, (x.size(-1),))
        # Multi-head attention pooling over byte dimension
        head_dim = D // self.num_pool_heads
        attn_logits = self.attn_pool(x)  # (B, P, S, H)
        attn_weights = F.softmax(attn_logits.permute(0, 1, 3, 2), dim=-1)  # (B, P, H, S)
        x_heads = x.view(B, P, S, self.num_pool_heads, head_dim).permute(0, 1, 3, 2, 4)  # (B, P, H, S, Dh)
        x = (attn_weights.unsqueeze(-1) * x_heads).sum(dim=3).reshape(B, P, D)  # (B, P, D)
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
        self.gate_proj = nn.Linear(model_dim, model_dim, bias=False)
        # Depthwise bidirectional conv to capture local byte bigram/trigram patterns.
        # Uses symmetric (centered) padding since target encoder sees the complete patch —
        # no causal constraint needed, and bidirectional context gives richer features.
        self.local_conv = nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2, groups=model_dim, bias=False)
        # Multi-head self-attention (4 heads) over byte positions within the patch.
        # Complements the local conv by allowing non-local byte interactions (e.g.
        # byte 0 attending to byte 7) which conv with k=5 cannot capture in patch_size=8.
        # Multiple heads capture diverse interaction patterns (e.g. one head for
        # repeated bytes, another for bracket-like pairs, etc.)
        self.sa_num_heads = 4
        self.sa_head_dim = model_dim // self.sa_num_heads
        self.sa_q = nn.Linear(model_dim, model_dim, bias=False)
        self.sa_k = nn.Linear(model_dim, model_dim, bias=False)
        self.sa_v = nn.Linear(model_dim, model_dim, bias=False)
        self.sa_out = nn.Linear(model_dim, model_dim, bias=False)
        self.sa_scale = self.sa_head_dim ** -0.5
        # Learned relative position bias for self-attention: encodes distance
        # between byte positions (range -(patch_size-1) to +(patch_size-1)).
        # This gives the SA explicit positional structure so it can distinguish
        # e.g. adjacent bytes from distant ones, complementing the conv's local bias.
        self.sa_rel_pos_bias = nn.Parameter(torch.zeros(self.sa_num_heads, 2 * patch_size - 1))
        # Second SwiGLU block after self-attention: composes SA outputs nonlinearly
        # before the irreversible pooling step. SA enriches representations with
        # non-local byte interactions; this block lets the model learn complex
        # combinations of those interactions before they are aggregated away.
        self.sa_post_gate = nn.Linear(model_dim, model_dim, bias=False)
        self.sa_post_up = nn.Linear(model_dim, model_dim, bias=False)
        self.sa_post_down = nn.Linear(model_dim, model_dim, bias=False)
        # Multi-head attention pooling: each head focuses on different byte patterns
        self.num_pool_heads = 4
        self.attn_pool = nn.Linear(model_dim, self.num_pool_heads, bias=False)
        # Learned temperature for attention sharpness (initialized to 1.0 = no-op)
        self.attn_temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))
        # Post-pooling MLP: adds nonlinear composition after aggregation
        self.post_mlp_up = nn.Linear(model_dim, model_dim, bias=False)
        self.post_mlp_down = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.normal_(self.local_pos, mean=0.0, std=0.01)

    def forward(self, patch_byte_emb: Tensor) -> Tensor:
        # patch_byte_emb: (B, num_patches, patch_size, model_dim)
        x = patch_byte_emb + self.local_pos[None, None, :, :].to(dtype=patch_byte_emb.dtype)
        x = F.rms_norm(x, (x.size(-1),))
        x = F.silu(self.gate_proj(x)) * self.in_proj(x)  # SwiGLU-style gated projection
        # Apply depthwise bidirectional conv over byte dimension with residual
        B, P, S, D = x.shape
        x_flat = x.reshape(B * P, S, D).transpose(1, 2)  # (B*P, D, S)
        conv_out = self.local_conv(x_flat)  # bidirectional: symmetric padding, no trim needed
        x = x + conv_out.transpose(1, 2).reshape(B, P, S, D)  # residual
        # Normalize after conv residual for stable attention inputs
        x = F.rms_norm(x, (x.size(-1),))
        # Multi-head self-attention over byte positions (4 heads, 8x8 per head — very cheap)
        B, P, S, D = x.shape
        H = self.sa_num_heads
        Dh = self.sa_head_dim
        q = self.sa_q(x).view(B, P, S, H, Dh).permute(0, 1, 3, 2, 4)  # (B, P, H, S, Dh)
        k = self.sa_k(x).view(B, P, S, H, Dh).permute(0, 1, 3, 2, 4)
        v = self.sa_v(x).view(B, P, S, H, Dh).permute(0, 1, 3, 2, 4)
        sa_logits = torch.einsum('bphsd,bphtd->bphst', q, k) * self.sa_scale  # (B, P, H, S, S)
        # Add learned relative position bias: index by (i - j + patch_size - 1)
        positions = torch.arange(S, device=sa_logits.device)
        rel_idx = positions.unsqueeze(1) - positions.unsqueeze(0) + (S - 1)  # (S, S), values in [0, 2S-2]
        sa_logits = sa_logits + self.sa_rel_pos_bias[:, rel_idx].to(dtype=sa_logits.dtype)  # broadcast (H, S, S)
        sa_weights = F.softmax(sa_logits, dim=-1)
        sa_out = torch.einsum('bphst,bphtd->bphsd', sa_weights, v)  # (B, P, H, S, Dh)
        sa_out = sa_out.permute(0, 1, 3, 2, 4).reshape(B, P, S, D)  # (B, P, S, D)
        x = x + self.sa_out(sa_out)  # residual
        x = F.rms_norm(x, (x.size(-1),))
        # Second SwiGLU: compose SA-enriched features before pooling
        x = x + self.sa_post_down(F.silu(self.sa_post_gate(x)) * self.sa_post_up(x))
        x = F.rms_norm(x, (x.size(-1),))
        # Multi-head attention-weighted pooling over byte dimension
        head_dim = D // self.num_pool_heads
        attn_logits = self.attn_pool(x) / self.attn_temperature.clamp(min=0.01).to(dtype=x.dtype)  # (B, P, S, H)
        attn_weights = F.softmax(attn_logits.permute(0, 1, 3, 2), dim=-1)  # (B, P, H, S)
        x_heads = x.view(B, P, S, self.num_pool_heads, head_dim).permute(0, 1, 3, 2, 4)  # (B, P, H, S, Dh)
        x = (attn_weights.unsqueeze(-1) * x_heads).sum(dim=3).reshape(B, P, D)  # (B, P, D)
        # Post-pooling nonlinear composition with residual
        x = x + self.post_mlp_down(F.silu(self.post_mlp_up(F.rms_norm(x, (x.size(-1),)))))
        x = self.out_proj(x)
        return F.rms_norm(x, (x.size(-1),))


class Predictor(nn.Module):
    def __init__(self, model_dim: int, max_patches: int = 16):
        super().__init__()
        # Learnable patch-position embeddings: gives cross-patch causal attention
        # explicit ordinal position information (patch 0 vs patch 3), enabling it
        # to learn position-dependent prediction strategies (e.g. first patch
        # prediction is harder than later ones that have more context)
        self.patch_pos_emb = nn.Parameter(torch.zeros(max_patches, model_dim))
        nn.init.normal_(self.patch_pos_emb, mean=0.0, std=0.02)
        # Three residual SwiGLU blocks with 2x expansion (model_dim → 2*model_dim → model_dim)
        # for richer current→target mapping capacity
        hidden_dim = model_dim * 2
        self.gate_a = nn.Linear(model_dim, hidden_dim, bias=False)
        self.up_a = nn.Linear(model_dim, hidden_dim, bias=False)
        self.down_a = nn.Linear(hidden_dim, model_dim, bias=False)
        self.gate_b = nn.Linear(model_dim, hidden_dim, bias=False)
        self.up_b = nn.Linear(model_dim, hidden_dim, bias=False)
        self.down_b = nn.Linear(hidden_dim, model_dim, bias=False)
        self.gate_c = nn.Linear(model_dim, hidden_dim, bias=False)
        self.up_c = nn.Linear(model_dim, hidden_dim, bias=False)
        self.down_c = nn.Linear(hidden_dim, model_dim, bias=False)
        self.drop = nn.Dropout(0.005)
        # Stochastic depth: skip probability increases with depth (deeper = more regularized)
        self.drop_path_rates = [0.005, 0.01, 0.015]
        # Layer-scale (CaiT): learned per-channel scaling on residual branches,
        # initialized small (0.1) to stabilize early training and improve generalization
        self.ls_a = nn.Parameter(torch.full((model_dim,), 0.1))
        self.ls_b = nn.Parameter(torch.full((model_dim,), 0.1))
        self.ls_c = nn.Parameter(torch.full((model_dim,), 0.1))
        # Cross-patch causal self-attention: lets each patch prediction attend to
        # earlier patches' representations. Since patches are consecutive text,
        # knowing what was predicted for patch i helps predict patch i+1.
        self.cross_num_heads = 4
        self.cross_head_dim = model_dim // self.cross_num_heads
        self.cross_q = nn.Linear(model_dim, model_dim, bias=False)
        self.cross_k = nn.Linear(model_dim, model_dim, bias=False)
        self.cross_v = nn.Linear(model_dim, model_dim, bias=False)
        self.cross_out = nn.Linear(model_dim, model_dim, bias=False)
        self.cross_scale = self.cross_head_dim ** -0.5
        self.ls_cross = nn.Parameter(torch.full((model_dim,), 0.1))
        # Second cross-patch causal self-attention: a second round of cross-patch
        # information exchange after all three SwiGLU blocks. The first cross-attn
        # (after block B) captures initial sequential dependencies; this second one
        # refines predictions using the richer post-block-C representations.
        self.cross2_q = nn.Linear(model_dim, model_dim, bias=False)
        self.cross2_k = nn.Linear(model_dim, model_dim, bias=False)
        self.cross2_v = nn.Linear(model_dim, model_dim, bias=False)
        self.cross2_out = nn.Linear(model_dim, model_dim, bias=False)
        self.ls_cross2 = nn.Parameter(torch.full((model_dim,), 0.1))
        # Learned per-feature gating to emphasize/suppress discriminative features for improved retrieval
        self.feature_gate = nn.Linear(model_dim, model_dim, bias=False)

    def _drop_path(self, x: Tensor, drop_rate: float) -> Tensor:
        """Stochastic depth: randomly zero entire residual branch during training."""
        if not self.training or drop_rate == 0.0:
            return x
        keep = 1.0 - drop_rate
        # Per-sample binary mask (drop entire sample's residual, not per-element)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep  # scale to preserve expected value

    def forward(self, x: Tensor) -> Tensor:
        # Add patch-position embeddings so cross-patch attention knows ordinal position
        P = x.size(1)
        x = x + self.patch_pos_emb[:P].unsqueeze(0).to(dtype=x.dtype)
        # Block 1: SwiGLU with residual + dropout + stochastic depth + layer-scale
        residual = x
        x = F.rms_norm(x, (x.size(-1),))
        x = self._drop_path(self.ls_a * self.drop(self.down_a(F.silu(self.gate_a(x)) * self.up_a(x))), self.drop_path_rates[0]) + residual
        # Block 2: SwiGLU with residual + dropout + stochastic depth + layer-scale
        residual = x
        x = F.rms_norm(x, (x.size(-1),))
        x = self._drop_path(self.ls_b * self.drop(self.down_b(F.silu(self.gate_b(x)) * self.up_b(x))), self.drop_path_rates[1]) + residual
        # Cross-patch causal self-attention after 2 SwiGLU blocks (enough per-patch
        # processing to produce meaningful representations before cross-patch sharing)
        # x shape: (B, P, D) where P = num_pairs (consecutive patches)
        residual = x
        xn = F.rms_norm(x, (x.size(-1),))
        B, P, D = xn.shape
        H = self.cross_num_heads
        Dh = self.cross_head_dim
        q = self.cross_q(xn).view(B, P, H, Dh).permute(0, 2, 1, 3)  # (B, H, P, Dh)
        k = self.cross_k(xn).view(B, P, H, Dh).permute(0, 2, 1, 3)
        v = self.cross_v(xn).view(B, P, H, Dh).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.cross_scale  # (B, H, P, P)
        # Causal mask: each patch can only attend to itself and earlier patches
        causal_mask = torch.triu(torch.ones(P, P, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask[None, None, :, :], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        cross_out = (attn @ v).permute(0, 2, 1, 3).reshape(B, P, D)  # (B, P, D)
        x = self._drop_path(self.ls_cross * self.cross_out(cross_out), 0.01) + residual
        # Block 3: SwiGLU with residual + dropout + stochastic depth + layer-scale
        residual = x
        x = F.rms_norm(x, (x.size(-1),))
        x = self._drop_path(self.ls_c * self.drop(self.down_c(F.silu(self.gate_c(x)) * self.up_c(x))), self.drop_path_rates[2]) + residual
        # Second cross-patch causal self-attention: refines predictions using
        # the richer post-block-C representations
        residual = x
        xn = F.rms_norm(x, (x.size(-1),))
        B, P, D = xn.shape
        H = self.cross_num_heads
        Dh = self.cross_head_dim
        q2 = self.cross2_q(xn).view(B, P, H, Dh).permute(0, 2, 1, 3)
        k2 = self.cross2_k(xn).view(B, P, H, Dh).permute(0, 2, 1, 3)
        v2 = self.cross2_v(xn).view(B, P, H, Dh).permute(0, 2, 1, 3)
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.cross_scale
        causal_mask2 = torch.triu(torch.ones(P, P, device=x.device, dtype=torch.bool), diagonal=1)
        attn2 = attn2.masked_fill(causal_mask2[None, None, :, :], float('-inf'))
        attn2 = F.softmax(attn2, dim=-1)
        cross2_out = (attn2 @ v2).permute(0, 2, 1, 3).reshape(B, P, D)
        x = self._drop_path(self.ls_cross2 * self.cross2_out(cross2_out), 0.01) + residual
        # Learned per-feature gating: allows model to emphasize/suppress features adaptively
        # for improved discriminative power in retrieval ranking.
        # Return as unnormalized delta: caller adds this to current_latent and normalizes,
        # grounding predictions in the current patch for residual prediction.
        x = F.rms_norm(x, (x.size(-1),))
        return x * torch.sigmoid(self.feature_gate(x))


class ProbeModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.patch_size = args.patch_size
        self.vocab_size = args.vocab_size
        self.jepa_std_target = args.jepa_std_target
        self.jepa_var_weight = args.jepa_var_weight
        self.jepa_cov_weight = args.jepa_cov_weight
        # Learnable log-temperature for InfoNCE (CLIP-style), initialized to log(1/0.07) ≈ 2.66
        self.log_infonce_temp = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))
        # Separate learnable temperature for repr_infonce: this loss operates on raw
        # normalized embeddings (representation space) while the projection-space InfoNCE
        # losses use projected embeddings with different similarity distributions.
        # A shared temperature forces a compromise; a separate one lets repr_infonce
        # (which directly optimizes the eval retrieval metric) find its own optimal sharpness.
        # Initialized slightly cooler (log(1/0.05) ≈ 3.0) since repr-space similarities
        # tend to be higher and benefit from sharper discrimination.
        self.log_repr_temp = nn.Parameter(torch.tensor(math.log(1.0 / 0.05), dtype=torch.float32))
        # Projection head for InfoNCE (SimCLR/BYOL-style): separates contrastive
        # optimization space from representation space, improving representation quality.
        # Wider hidden dim (2x) gives the contrastive loss a richer intermediate space
        # for learning discriminative features, sending better gradients back to encoders.
        proj_dim = args.model_dim
        proj_hidden = args.model_dim * 2
        self.infonce_proj = nn.Sequential(
            nn.Linear(args.model_dim, proj_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(proj_hidden, proj_dim, bias=False),
        )
        # Byte reconstruction head: decodes predicted embedding back to next-patch bytes.
        # Forces pred_f to carry content-specific byte information, making predictor
        # outputs more distinctive across patches (different bytes → different embeddings),
        # directly attacking the retrieval_at_1 bottleneck without touching the alignment path.
        self.pred_token_head = nn.Linear(args.model_dim, args.vocab_size * args.patch_size, bias=False)
        # MoCo-style momentum queue: stores recent target projections as extra negatives
        # for InfoNCE. Increases effective negative set from B*P (~1024) to B*P+Q (~3072)
        # without proportional compute cost, improving contrastive discrimination.
        self.queue_size = 2048
        self.register_buffer('target_queue', F.normalize(torch.randn(self.queue_size, proj_dim), dim=-1))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        # Second MoCo queue for raw normalized target embeddings (not projections).
        # repr_infonce directly optimizes the eval retrieval metric (pred_n @ target_n.T)
        # but only has B*P negatives. This queue adds 2048 extra negatives specifically
        # for repr_infonce, improving retrieval discrimination with harder negatives.
        self.register_buffer('repr_queue', F.normalize(torch.randn(self.queue_size, args.model_dim), dim=-1))
        self.register_buffer('repr_queue_ptr', torch.zeros(1, dtype=torch.long))
        # Learnable prototype vectors for ProtoNCE loss.
        # K=8 prototypes model the global cluster structure of target embeddings.
        # ProtoNCE upweights negatives whose target shares the same prototype cluster
        # as the positive, forcing sharper discrimination within confusion-prone regions
        # of the hypersphere. Unlike hw_infonce (raw pairwise target similarity, noisy,
        # batch-local), ProtoNCE uses a learned structured basis that captures systematic
        # confusion modes and generalizes across batches.
        self.num_prototypes = 8
        self.prototypes = nn.Parameter(F.normalize(torch.randn(self.num_prototypes, args.model_dim), dim=-1))
        # Masked-byte identity prediction head: decodes online target encoder output
        # back to the original byte at a randomly masked position. Unlike token_loss
        # (which decodes the PREDICTOR output → full patch bytes, training only the
        # predictor path) and masked_denoising_loss (which matches masked online encoder
        # to clean EMA target, training representation matching), this creates a direct
        # absolute byte-identity gradient on target_online_encoder itself. The encoder
        # must build representations that retain fine-grained byte content — the same
        # content specificity that makes retrieval succeed. Since target_online_encoder
        # feeds the EMA target encoder via EMA updates, this improves retrieval by
        # making target embeddings more byte-discriminative over time.
        self.mb_head = nn.Linear(args.model_dim, args.vocab_size, bias=False)
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.online_encoder = CurrentPatchEncoder(args.model_dim, args.patch_size)
        self.target_online_encoder = TargetPatchEncoder(args.model_dim, args.patch_size)
        self.predictor = Predictor(args.model_dim)
        self.target_tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.target_encoder = TargetPatchEncoder(args.model_dim, args.patch_size)

    @torch.no_grad()
    def _enqueue(self, keys: Tensor) -> None:
        """FIFO enqueue of projected target embeddings into momentum queue."""
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr.item())
        if ptr + batch_size <= self.queue_size:
            self.target_queue[ptr:ptr + batch_size] = keys
        else:
            overflow = (ptr + batch_size) - self.queue_size
            first = batch_size - overflow
            self.target_queue[ptr:ptr + first] = keys[:first]
            self.target_queue[:overflow] = keys[first:]
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    @torch.no_grad()
    def _enqueue_repr(self, keys: Tensor) -> None:
        """FIFO enqueue of normalized target embeddings into repr queue."""
        batch_size = keys.size(0)
        ptr = int(self.repr_queue_ptr.item())
        if ptr + batch_size <= self.queue_size:
            self.repr_queue[ptr:ptr + batch_size] = keys
        else:
            overflow = (ptr + batch_size) - self.queue_size
            first = batch_size - overflow
            self.repr_queue[ptr:ptr + first] = keys[:first]
            self.repr_queue[:overflow] = keys[first:]
        self.repr_queue_ptr[0] = (ptr + batch_size) % self.queue_size

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
        # Online target encoding: provides actual gradients to target_online_encoder
        # (previously discarded, leaving target_online_encoder at random init)
        online_next = self.encode_next_online(next_patches).float()
        online_next_n = F.normalize(online_next, dim=-1)
        # Byte-mutation near-duplicate repulsion: force the online target encoder to
        # produce distinct embeddings for patches that differ by only a single byte.
        # Motivation: retrieval_at_1=0.089 with val_cosine=0.722 — the encoder conflates
        # near-identical patches (e.g. common words differing by one letter/space).
        # online_infonce contrasts random patch pairs (easy negatives); this specifically
        # contrasts 1-byte-apart patches (the hardest negatives for retrieval).
        # Flows back to the EMA target encoder via EMA updates, making target embeddings
        # more discriminative in the fine-grained neighborhood where argmax fails.
        B_dc, P_dc, S_dc = next_patches.shape
        with torch.no_grad():
            mut_pos = torch.randint(S_dc, (B_dc, P_dc), device=next_patches.device)
            mut_val = torch.randint(self.vocab_size, (B_dc, P_dc), device=next_patches.device)
            decoy_patches = next_patches.clone()
            decoy_patches.scatter_(2, mut_pos.unsqueeze(-1), mut_val.unsqueeze(-1))
        online_decoy = self.encode_next_online(decoy_patches).float()
        online_decoy_n = F.normalize(online_decoy, dim=-1)
        # Margin: cosine sim between original and 1-byte-mutant should be < 0.80.
        # Threshold is generous (only requires measurable sensitivity to byte changes),
        # but acts specifically in the high-similarity regime where retrieval breaks.
        byte_mut_repulsion = F.relu((online_next_n * online_decoy_n).sum(dim=-1) - 0.80).mean()
        # Masked-byte identity prediction: encode next_patches with ONE byte zeroed out
        # per patch, then predict that byte's identity from the patch embedding.
        # This forces target_online_encoder to build representations that retain byte identity
        # (gradients flow directly here, unlike token_loss which only trains the predictor).
        # One masked position per patch keeps the task easy enough to learn quickly while
        # forcing the encoder to represent the full byte sequence (not just aggregate statistics).
        B_mb, P_mb, S_mb = next_patches.shape
        with torch.no_grad():
            mb_pos = torch.randint(S_mb, (B_mb, P_mb), device=next_patches.device)  # (B, P)
            mb_true = next_patches.gather(2, mb_pos.unsqueeze(-1)).squeeze(-1)  # (B, P) true byte ids
            next_mb = next_patches.clone()
            next_mb.scatter_(2, mb_pos.unsqueeze(-1), 0)  # zero out one byte per patch
        online_mb = self.encode_next_online(next_mb).float()  # (B, P, D) — gradients flow to encoder
        mb_logits = self.mb_head(online_mb)  # (B, P, vocab_size)
        masked_byte_loss = F.cross_entropy(mb_logits.reshape(-1, self.vocab_size), mb_true.reshape(-1).long())
        # Masked-patch denoising for online target encoder: randomly zero-mask ~25% of
        # bytes in next_patches, encode with target_online_encoder, and train the result
        # to match the clean EMA target embedding. The EMA target encoder sees clean
        # patches; the online encoder must recover the same embedding despite partial
        # information. This creates a richer denoising gradient for target_online_encoder
        # (which over time improves the EMA target encoder via EMA updates), making the
        # target embeddings more content-robust and harder to confuse across patches.
        # Distinct from contrastive losses: an absolute per-sample denoising signal.
        B_dn, P_dn, S_dn = next_patches.shape
        byte_mask = torch.rand(B_dn, P_dn, S_dn, device=next_patches.device) < 0.25
        next_patches_masked = next_patches.masked_fill(byte_mask, 0)  # 0 = null byte
        online_next_masked = self.encode_next_online(next_patches_masked).float()
        # Residual prediction: predictor outputs an unnormalized delta; we add it to the
        # current embedding and renormalize. Different patches have diverse current bases
        # so predicted targets inherit that diversity — directly attacking the retrieval
        # bottleneck where absolute prediction collapses toward a mean direction.
        pred_delta = self.predictor(current_latent)
        target = self.encode_next_target(next_patches).float()
        pred_f = current_latent.float() + pred_delta.float()
        pred_n = F.normalize(pred_f, dim=-1)
        target_n = F.normalize(target, dim=-1)
        # Denoising MSE: online encoder on masked patches must match clean EMA target
        masked_denoising_loss = F.mse_loss(
            F.normalize(online_next_masked, dim=-1), target_n.detach()
        )
        mse = F.mse_loss(pred_n, target_n, reduction="mean")
        cosine = (pred_n * target_n).sum(dim=-1).mean()
        # Project through InfoNCE head (SimCLR-style: contrastive loss in projection
        # space, MSE in representation space — avoids contrastive loss distorting repr)
        pred_proj = F.normalize(self.infonce_proj(pred_f), dim=-1)
        target_proj = F.normalize(self.infonce_proj(target.detach()), dim=-1)
        online_proj = F.normalize(self.infonce_proj(online_next.float()), dim=-1)
        # InfoNCE contrastive loss against EMA target (in projection space)
        # with MoCo-style momentum queue providing extra negatives
        pred_proj_flat = pred_proj.reshape(-1, pred_proj.size(-1))  # (B*P, D)
        target_proj_flat = target_proj.reshape(-1, target_proj.size(-1))  # (B*P, D)
        temp_scale = self.log_infonce_temp.exp().clamp(max=100.0)
        # Concatenate batch targets with queue for more negatives
        queue_neg = self.target_queue.clone().detach().to(dtype=target_proj_flat.dtype)  # (Q, D)
        all_targets = torch.cat([target_proj_flat, queue_neg], dim=0)  # (B*P+Q, D)
        logits = pred_proj_flat @ all_targets.T  # (B*P, B*P+Q)
        logits = logits * temp_scale
        labels = torch.arange(pred_proj_flat.size(0), device=logits.device)  # positives are first B*P columns
        infonce_fwd = F.cross_entropy(logits, labels, label_smoothing=0.01)
        # Symmetric (CLIP-style) reverse direction: target→pred within batch only
        # (queue entries have no corresponding pred, so reverse is batch-only)
        rev_logits = target_proj_flat @ pred_proj_flat.T  # (B*P, B*P)
        rev_logits = rev_logits * temp_scale
        infonce_rev = F.cross_entropy(rev_logits, labels, label_smoothing=0.01)
        infonce = 0.5 * infonce_fwd + 0.5 * infonce_rev
        # Update queue with current batch targets
        self._enqueue(target_proj_flat.detach())
        # InfoNCE against online target (in projection space)
        online_proj_flat = online_proj.reshape(-1, online_proj.size(-1))
        online_logits = pred_proj_flat @ online_proj_flat.T
        online_logits = online_logits * temp_scale
        online_infonce_fwd = F.cross_entropy(online_logits, labels, label_smoothing=0.01)
        # Symmetric reverse direction for online InfoNCE
        online_rev_logits = online_proj_flat @ pred_proj_flat.T
        online_rev_logits = online_rev_logits * temp_scale
        online_infonce_rev = F.cross_entropy(online_rev_logits, labels, label_smoothing=0.01)
        online_infonce = 0.5 * online_infonce_fwd + 0.5 * online_infonce_rev
        # Direct representation-space InfoNCE: eval retrieval uses pred_n @ target_n.T
        # (raw normalized embeddings, not projections), so this loss directly optimizes
        # the retrieval metric we measure, closing the train-eval mismatch.
        # Uses MoCo-style repr_queue for extra negatives (matching projection-space infonce).
        # Uses separate temperature (log_repr_temp) since repr-space similarities have
        # different distribution than projection-space similarities.
        repr_temp_scale = self.log_repr_temp.exp().clamp(max=100.0)
        pred_n_flat = pred_n.reshape(-1, pred_n.size(-1))  # (B*P, D)
        target_n_flat = target_n.reshape(-1, target_n.size(-1))  # (B*P, D)
        repr_queue_neg = self.repr_queue.clone().detach().to(dtype=target_n_flat.dtype)  # (Q, D)
        all_repr_targets = torch.cat([target_n_flat, repr_queue_neg], dim=0)  # (B*P+Q, D)
        repr_logits = pred_n_flat @ all_repr_targets.T  # (B*P, B*P+Q)
        repr_logits = repr_logits * repr_temp_scale
        # Standard cross-entropy for repr_infonce (no label smoothing — preserves
        # retrieval confidence). Focal loss was tried but down-weighting easy positives
        # destabilized well-learned associations without improving retrieval accuracy.
        repr_infonce_fwd = F.cross_entropy(repr_logits, labels)
        # Symmetric reverse direction for repr InfoNCE (batch-only, no queue in reverse)
        repr_rev_logits = target_n_flat @ pred_n_flat.T  # (B*P, B*P)
        repr_rev_logits = repr_rev_logits * repr_temp_scale
        repr_infonce_rev = F.cross_entropy(repr_rev_logits, labels)
        repr_infonce = 0.5 * repr_infonce_fwd + 0.5 * repr_infonce_rev
        # Target-similarity-reweighted InfoNCE: upweight negatives whose target is
        # similar to the positive target in the batch. When two targets are close on the
        # hypersphere (common for consecutive patches with shared context), retrieval fails
        # because their predictor outputs also end up close. By boosting the logit weight
        # of those "confusable" negatives in the denominator (adding beta*sim_ij), we force
        # the predictor to push its output clearly past the whole cluster of similar patches,
        # directly attacking retrieval_at_1 without adding any parameters.
        repr_logits_batch = pred_n_flat @ target_n_flat.T * repr_temp_scale  # (N, N) batch-only
        with torch.no_grad():
            tgt_tgt_sim = target_n_flat @ target_n_flat.T  # (N, N)
            eye_b = torch.eye(pred_n_flat.size(0), device=tgt_tgt_sim.device, dtype=torch.bool)
            # beta=3: exp(3*0.9)~exp(2.7)≈15x upweight for near-duplicate targets; exp(3*0)=1x for orthogonal
            beta_hw = 3.0
            hw_correction = beta_hw * tgt_tgt_sim.masked_fill(eye_b, 0.0)
        hw_logits = repr_logits_batch + hw_correction  # push hard negatives up in denominator
        hw_infonce = F.cross_entropy(hw_logits, labels)
        # Update repr queue with current batch's normalized targets
        self._enqueue_repr(target_n_flat.detach())
        # Relational Knowledge Distillation (RKD) loss: force predictor pairwise
        # similarity structure to match target pairwise similarity structure.
        # Motivation: val_cosine=0.72 but retrieval_at_1=0.089 signals that all
        # predictor outputs concentrate in the same hyperspherical region and are
        # equally similar to all targets — discrimination fails not because alignment
        # is poor but because pairwise GEOMETRY doesn't match. RKD enforces:
        # p_rel[i,j] ≈ t_rel[i,j], i.e., if target_i and target_j are similar, then
        # pred_i and pred_j should also be similar, and if they are different, pred_i
        # and pred_j should be different. This structural constraint means the correct
        # target_i naturally becomes the nearest neighbor of pred_i in the full space.
        # Unlike InfoNCE (pushes positive above negatives), uniformity (spreads both
        # modalities uniformly), and Barlow Twins (cross-dimension alignment), RKD
        # matches the relational structure between the two representation spaces —
        # the missing signal for the retrieval bottleneck.
        N_rkd = min(pred_n_flat.size(0), 256)
        p_rkd = pred_n_flat[:N_rkd]
        t_rkd = target_n_flat[:N_rkd].detach()  # targets are fixed structure, grad flows only to predictor
        p_rel = p_rkd @ p_rkd.T  # (N_rkd, N_rkd) predictor pairwise cosine similarity
        t_rel = t_rkd @ t_rkd.T  # (N_rkd, N_rkd) target pairwise cosine similarity
        rkd_loss = F.mse_loss(p_rel, t_rel)
        # ProtoNCE: prototype-guided hard negative upweighting for retrieval.
        # K=8 learnable prototypes learn the global cluster structure of target embeddings.
        # Targets assigned to the same prototype cluster are most likely to be confused
        # (they occupy nearby hyperspherical regions). By upweighting their logits in the
        # InfoNCE denominator, we force the predictor to separate predictions within the
        # same confused cluster — directly attacking retrieval_at_1.
        # Compared to hw_infonce (batch-local pairwise target similarity): ProtoNCE uses
        # a learned global basis that captures systematic confusion modes, not per-batch noise.
        proto_n = F.normalize(self.prototypes, dim=-1)  # (K, D) normalized prototypes
        target_cluster_logits = target_n_flat @ proto_n.T  # (N, K) cosine sim to prototypes
        target_cluster_soft = F.softmax(target_cluster_logits * 8.0, dim=-1)  # (N, K) soft assignments
        # Cluster confusion: C[i,j] = sum_k soft_i[k] * soft_j[k] (shared cluster membership)
        cluster_confusion = target_cluster_soft @ target_cluster_soft.T  # (N, N)
        # Upweight logits for confusable negatives (skip diagonal = positive pair)
        eye_n_f = eye_b.float()  # reuse eye_b (defined above in hw_infonce block)
        # Remove .detach() so gradients flow to self.prototypes: the soft assignments
        # depend on proto_n (via target_cluster_logits), so prototypes can learn which
        # directions capture systematic confusion structure. Without this, prototypes
        # stay at random init forever and the whole mechanism is wasted.
        proto_logits = repr_logits_batch + 2.0 * (cluster_confusion * (1.0 - eye_n_f))
        proto_infonce = F.cross_entropy(proto_logits, labels)
        # Prototype diversity: prevent prototypes from collapsing to same direction
        proto_self_sim = proto_n @ proto_n.T  # (K, K)
        eye_k = torch.eye(self.num_prototypes, device=proto_n.device)
        proto_div_loss = (proto_self_sim * (1.0 - eye_k)).pow(2).sum() / (self.num_prototypes * (self.num_prototypes - 1))
        # Uniformity loss (Wang & Isola 2020): push embeddings apart on the unit hypersphere.
        # For unit vectors, -2||z_i - z_j||^2 = -4 + 4*sim, so the uniformity loss becomes
        # log(mean_{i≠j}[exp(4*sim_{ij})]): minimized when sim→0 (embeddings spread uniformly).
        # Applied to both predictor outputs and online target encoder outputs (which feed the
        # EMA target encoder). Directly targets the retrieval_at_1 bottleneck by ensuring
        # each embedding has a more distinct position on the sphere.
        N_u = min(pred_n_flat.size(0), 512)
        p_u = pred_n_flat[:N_u]  # predictor outputs — gradient flows here
        o_u = online_next_n.reshape(-1, online_next_n.size(-1))[:N_u]  # online target encoder — gradient flows here
        eye_u = torch.eye(N_u, device=p_u.device, dtype=p_u.dtype)
        p_sim_u = p_u @ p_u.T  # (N_u, N_u)
        o_sim_u = o_u @ o_u.T
        # Off-diagonal mean: exclude self-similarity (diagonal = 1 always, not informative)
        off_diag = 1.0 - eye_u
        p_unif = (p_sim_u.mul(4).exp() * off_diag).sum() / (N_u * (N_u - 1))
        o_unif = (o_sim_u.mul(4).exp() * off_diag).sum() / (N_u * (N_u - 1))
        uniform_loss = 0.5 * p_unif.log() + 0.5 * o_unif.log()
        # Byte reconstruction auxiliary loss: decode predictor output back to next-patch tokens.
        # This is an absolute content signal — unlike contrastive losses that only push/pull
        # relative to other batch items, reconstruction forces pred_f to be specific to the
        # exact bytes of next_patches. Two patches that look similar under cosine sim but have
        # different bytes will now receive distinct gradient signals, improving retrieval.
        token_logits = self.pred_token_head(pred_f.reshape(-1, pred_f.size(-1)))  # (B*P, vocab*patch_size)
        token_logits = token_logits.view(-1, self.patch_size, self.vocab_size)  # (B*P, patch_size, vocab)
        token_targets = next_patches.reshape(-1, self.patch_size).long()  # (B*P, patch_size)
        token_loss = F.cross_entropy(token_logits.reshape(-1, self.vocab_size), token_targets.reshape(-1))
        pred_std = pred_f.std(dim=(0, 1), correction=0)
        target_std = target.std(dim=(0, 1), correction=0)
        var_loss = torch.relu(self.jepa_std_target - pred_std).mean()
        # VICReg-style covariance regularization: penalize off-diagonal covariance
        # to decorrelate embedding dimensions and encourage richer representations
        pred_cent = pred_f.reshape(-1, pred_f.size(-1))  # (B*P, D)
        pred_centered = pred_cent - pred_cent.mean(dim=0, keepdim=True)
        N = pred_centered.size(0)
        cov_matrix = (pred_centered.T @ pred_centered) / max(N - 1, 1)
        # Zero out the diagonal (we only penalize off-diagonal correlations)
        cov_off_diag = cov_matrix - torch.diag(cov_matrix.diag())
        cov_loss = (cov_off_diag ** 2).sum() / pred_f.size(-1)
        # Barlow Twins cross-correlation loss: push pred-target cross-correlation
        # matrix toward identity. On-diagonal → 1 (alignment), off-diagonal → 0
        # (redundancy reduction). Complements MSE (per-pair alignment) and InfoNCE
        # (batch discriminability) by explicitly decorrelating across the two spaces.
        pred_bt = pred_n.reshape(-1, pred_n.size(-1))
        target_bt = target_n.reshape(-1, target_n.size(-1))
        pred_z = (pred_bt - pred_bt.mean(0)) / (pred_bt.std(0) + 1e-5)
        target_z = (target_bt - target_bt.mean(0)) / (target_bt.std(0) + 1e-5)
        cc = (pred_z.T @ target_z) / pred_z.size(0)  # (D, D) cross-correlation
        bt_on_diag = ((cc.diag() - 1) ** 2).sum() / cc.size(0)
        bt_off_diag = (cc - torch.diag(cc.diag())).pow(2).sum() / cc.size(0)
        bt_loss = bt_on_diag + 0.005 * bt_off_diag
        # Hard negative triplet loss: for each predictor output, find the most
        # similar (hardest) negative target in the batch and enforce a margin
        # separating it from the positive. InfoNCE spreads gradient over all negatives
        # uniformly — this focuses on the single most confusing negative per anchor,
        # which directly attacks the retrieval_at_1 bottleneck (argmax fails when the
        # hardest negative is closer than the positive). Margin 0.2 leaves room for
        # the positive (mean cos ~0.72) to sit safely above hard negatives (~0.5).
        N_hn = pred_n_flat.size(0)
        sim_hn_batch = pred_n_flat @ target_n_flat.T  # (N, N) full cosine similarity matrix
        # Mask diagonal so we don't count the positive as a hard negative
        diag_mask_hn = torch.eye(N_hn, device=sim_hn_batch.device, dtype=torch.bool)
        sim_hn_batch_masked = sim_hn_batch.masked_fill(diag_mask_hn, -1e9)
        # Augment hard negative pool with repr_queue entries (past-batch normalized targets).
        # repr_queue_neg was computed before _enqueue_repr, so it contains only targets from
        # prior batches — no current-batch contamination. Expanding from ~N (~1536) to ~N+Q
        # (~3584) forces the predictor to be discriminative against much harder, more diverse
        # negatives, directly attacking the retrieval_at_1 bottleneck.
        sim_hn_queue = pred_n_flat @ repr_queue_neg.T  # (N, Q)
        all_neg_sim_hn = torch.cat([sim_hn_batch_masked, sim_hn_queue], dim=1)  # (N, N+Q)
        hardest_neg_sim = all_neg_sim_hn.max(dim=1).values  # (N,) hardest negative across batch+queue
        pos_sim_hn = sim_hn_batch.diagonal()  # (N,) positive cosine similarity
        hn_margin = 0.2
        hard_neg_loss = F.relu(hardest_neg_sim - pos_sim_hn + hn_margin).mean()
        # ListNet ranking loss: directly optimize P(correct target is ranked #1).
        # All other ranking losses here are proxies: InfoNCE optimizes a softmax cross-entropy
        # over all items uniformly, hard_neg_loss enforces a margin against the top-1 negative,
        # hw_infonce upweights confusable negatives — but none directly matches the retrieval@1
        # metric (argmax of pred_n @ target_n.T == ground truth). ListNet (Cao et al. 2007)
        # models the Plackett-Luce probability that item i is ranked first given similarity
        # scores: P(i ranked 1st) = exp(s_i) / sum_j exp(s_j). The loss is:
        #   L = -sum_i y_i * log P(i ranked 1st)
        # where y_i is the ground-truth relevance (1 for positive, 0 for negatives).
        # This is equivalent to: -log(exp(s_pos) / sum_j exp(s_j)) = -s_pos + log(sum_j exp(s_j))
        # which is exactly the standard cross-entropy on the similarity logits — *but* using
        # the raw representation-space similarities (pred_n @ target_n.T), NOT a projection head.
        # This creates a direct, metric-aligned gradient on pred_n without the train/eval mismatch
        # introduced by projection heads. Unlike repr_infonce (also batch cross-entropy on pred_n @
        # target_n.T), ListNet here uses a batch-local (no queue) sharper temperature to concentrate
        # gradient energy on the hardest per-sample ranking decisions, avoiding queue dilution.
        # Temperature τ=0.05 (20x sharpening) → near-zero gradient for items already ranked #1,
        # strong gradient where argmax is wrong — exactly matching how retrieval_at_1 is computed.
        listnet_temp = 0.05
        listnet_logits = sim_hn_batch / listnet_temp  # (N, N) batch-only, no queue
        listnet_loss = F.cross_entropy(listnet_logits, torch.arange(N_hn, device=listnet_logits.device))
        # Synthetic midpoint hard negative loss: the critical failure mode for retrieval
        # is when a predictor output lands at the "confusion boundary" between two similar
        # targets. Unlike hard_neg_loss (uses the hardest REAL negative from batch+queue),
        # this constructs a SYNTHETIC negative at the spherical midpoint between each target
        # and its nearest neighbor in the batch — the exact decision-surface point where
        # argmax flips from the correct target to the wrong one. By enforcing a margin triplet
        # pushing pred_i clearly toward target_i and past this midpoint, we train the predictor
        # to navigate the worst-case confusion geometry rather than just the hardest real negative.
        # Reuses tgt_tgt_sim and eye_b (already computed in hw_infonce block) for efficiency.
        with torch.no_grad():
            tgt_sim_for_nn = tgt_tgt_sim.masked_fill(eye_b, -1e9)
            nn_idx_shn = tgt_sim_for_nn.argmax(dim=1)  # (N,) index of nearest-neighbor target
            t_nn_shn = target_n_flat[nn_idx_shn]  # (N, D) nearest-neighbor target embedding
            # Spherical midpoint: normalize(t_i + t_nn) gives the midpoint direction on sphere
            t_synth = F.normalize(target_n_flat + t_nn_shn, dim=-1)  # (N, D)
        synth_neg_sim = (pred_n_flat * t_synth).sum(dim=-1)  # (N,) cosine to synthetic negative
        # Margin triplet: push pred past the cluster-boundary midpoint toward the real target
        synth_hard_neg_loss = F.relu(synth_neg_sim - pos_sim_hn + 0.10).mean()
        # Within-sequence discriminative InfoNCE: forces the predictor to distinguish
        # between adjacent patches in the SAME text sequence. Adjacent patches (e.g.,
        # bytes 8-15 vs 16-23 of the same document) are the hardest negatives for
        # retrieval because they share local context and byte patterns. The standard
        # per-batch InfoNCE dilutes this hard signal with easier cross-sequence negatives.
        # By constructing a per-sequence (B, P, P) similarity matrix and treating it as
        # a P-way position-classification problem (P=6 pairs per sequence), we give the
        # model an isolated, amplified gradient specifically for intra-sequence discrimination
        # — the exact bottleneck causing low retrieval_at_1.
        pred_seq = pred_n  # (B, P, D) per-batch-item, per-position normalized predictions
        target_seq = target_n  # (B, P, D) per-batch-item, per-position normalized targets
        seq_sim = torch.bmm(pred_seq, target_seq.transpose(1, 2))  # (B, P, P) within-sequence similarity
        seq_logits = seq_sim * repr_temp_scale  # temperature-scaled (same repr space as repr_infonce)
        B_seq, P_seq = seq_logits.shape[:2]
        seq_logits_flat = seq_logits.reshape(B_seq * P_seq, P_seq)  # (B*P, P) P-way classification
        seq_labels = torch.arange(P_seq, device=seq_logits.device).repeat(B_seq)  # (B*P,) correct index = diagonal
        within_seq_infonce = F.cross_entropy(seq_logits_flat, seq_labels)
        # Centered Kernel Alignment (CKA) loss: maximize the linear CKA between
        # predictor and target representation spaces. Unlike RKD (which matches N×N
        # pairwise cosine similarity matrices entry-by-entry), CKA is invariant to
        # orthogonal rotation and isotropic scaling — it measures whether predictor
        # captures the SAME variance structure as targets, not the same exact distances.
        # CKA = <K_pred, K_tgt>_F / (||K_pred||_F * ||K_tgt||_F) where K = X_c @ X_c.T.
        # When predictor outputs collapse to a low-rank subspace (all predictions along
        # a few shared directions), K_pred is near rank-1 while K_tgt has full rank —
        # CKA will be low and loss = 1 - CKA will be large, pushing for richer spread.
        # This directly attacks the "flat similarity distribution" that kills retrieval_at_1.
        # Distinct from RKD: RKD enforces exact pairwise similarities; CKA enforces
        # alignment of the full second-order geometry (invariant to within-space rotations).
        N_cka = min(pred_n_flat.size(0), 256)
        pred_cka = pred_n_flat[:N_cka]
        tgt_cka = target_n_flat[:N_cka].detach()
        pred_c = pred_cka - pred_cka.mean(0, keepdim=True)  # center rows
        tgt_c = tgt_cka - tgt_cka.mean(0, keepdim=True)
        K_pred = pred_c @ pred_c.T  # (N_cka, N_cka) linear kernel
        K_tgt = tgt_c @ tgt_c.T    # (N_cka, N_cka) linear kernel
        cka_num = (K_pred * K_tgt).sum()
        cka_denom = K_pred.norm() * K_tgt.norm() + 1e-8
        cka_loss = 1.0 - cka_num / cka_denom  # minimize → maximize structural alignment
        # Spectral spread matching: force the singular value distribution of predictor
        # outputs to match that of the target embeddings. Unlike RKD (aligns pairwise
        # cosine similarity matrix entry-by-entry) and CKA (aligns full Gram matrix
        # including directional structure), SVD-based spectral matching is rotation-invariant:
        # it only requires the predictor to allocate variance across principal directions
        # in the same proportions as targets, without forcing specific axes.
        # If targets have a rich spectrum (several large + many smaller singular values),
        # the predictor must match that profile — preventing collapse to a low-rank subspace
        # where all predictions lie along 1-2 shared directions. This directly attacks
        # the retrieval_at_1 bottleneck (flat similarity distribution) via a signal that
        # is orthogonal to all existing pair-wise and sample-wise losses.
        N_sp = min(pred_n_flat.size(0), 128)
        K_sv = min(16, N_sp // 4)
        pred_sp = pred_n_flat[:N_sp]
        tgt_sp = target_n_flat[:N_sp].detach()
        _, pred_sv, _ = torch.svd_lowrank(pred_sp, q=K_sv)  # top-K singular values, (K_sv,)
        _, tgt_sv, _ = torch.svd_lowrank(tgt_sp, q=K_sv)
        # Normalize to probability distribution over spectral energy
        pred_sv_norm = pred_sv / (pred_sv.sum() + 1e-8)
        tgt_sv_norm = tgt_sv / (tgt_sv.sum() + 1e-8)
        spectral_loss = F.mse_loss(pred_sv_norm, tgt_sv_norm.detach())
        total = 0.35 * mse + 0.20 * infonce + 0.10 * online_infonce + 0.15 * repr_infonce + self.jepa_var_weight * var_loss + self.jepa_cov_weight * cov_loss + 0.05 * bt_loss + 0.05 * uniform_loss + 0.15 * hard_neg_loss + 0.05 * token_loss + 0.05 * hw_infonce + 0.10 * within_seq_infonce + 0.10 * masked_denoising_loss + 0.05 * proto_infonce + 0.005 * proto_div_loss + 0.10 * rkd_loss + 0.10 * cka_loss + 0.10 * synth_hard_neg_loss + 0.05 * spectral_loss + 0.10 * byte_mut_repulsion + 0.10 * masked_byte_loss + 0.15 * listnet_loss
        aux = {
            "mse": mse.detach(),
            "cosine": cosine.detach(),
            "infonce": infonce.detach(),
            "online_infonce": online_infonce.detach(),
            "repr_infonce": repr_infonce.detach(),
            "hw_infonce": hw_infonce.detach(),
            "pred_std": pred_std.mean().detach(),
            "target_std": target_std.mean().detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "bt_loss": bt_loss.detach(),
            "uniform_loss": uniform_loss.detach(),
            "hard_neg_loss": hard_neg_loss.detach(),
            "token_loss": token_loss.detach(),
            "within_seq_infonce": within_seq_infonce.detach(),
            "masked_denoising_loss": masked_denoising_loss.detach(),
            "proto_infonce": proto_infonce.detach(),
            "proto_div_loss": proto_div_loss.detach(),
            "rkd_loss": rkd_loss.detach(),
            "cka_loss": cka_loss.detach(),
            "synth_hard_neg_loss": synth_hard_neg_loss.detach(),
            "spectral_loss": spectral_loss.detach(),
            "byte_mut_repulsion": byte_mut_repulsion.detach(),
            "masked_byte_loss": masked_byte_loss.detach(),
            "listnet_loss": listnet_loss.detach(),
        }
        return total, aux


def make_patch_pairs(stream, batch_size: int, patch_size: int, device: torch.device, num_pairs: int = 6) -> tuple[Tensor, Tensor]:
    # Create num_pairs consecutive patch pairs per batch item.
    # Each item needs (num_pairs+1) patches to form num_pairs overlapping adjacent pairs.
    # This increases InfoNCE contrastive set from B to B*num_pairs (e.g. 256->1024).
    num_patches = num_pairs + 1
    chunk = stream.take(batch_size * num_patches * patch_size)
    all_patches = chunk.reshape(batch_size, num_patches, patch_size)
    current = all_patches[:, :num_pairs, :].to(device=device, dtype=torch.long)  # (B, num_pairs, PS)
    nxt = all_patches[:, 1:, :].to(device=device, dtype=torch.long)  # (B, num_pairs, PS)
    return current, nxt


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
        # Residual prediction: predictor outputs delta; add to current before normalizing
        pred_delta = model.predictor(current_latent).float()
        pred = (current_latent.float() + pred_delta).reshape(-1, args.model_dim)
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
            {"params": model.predictor.parameters(), "lr": args.lr * 1.5},
            {"params": model.infonce_proj.parameters(), "lr": args.lr},
            {"params": [model.log_infonce_temp], "lr": args.lr * 0.1},  # slower lr for temperature
            {"params": [model.log_repr_temp], "lr": args.lr * 0.1},  # separate repr temperature
            {"params": model.pred_token_head.parameters(), "lr": args.lr},  # byte reconstruction head
            {"params": [model.prototypes], "lr": args.lr},  # ProtoNCE prototype vectors — were missing from optimizer, leaving them frozen at random init
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    log0(f"device:{device.type}")
    log0(f"probe_family:jepa_target_embedding_inner_loop patch_size:{args.patch_size} model_dim:{args.model_dim}")
    log0(f"data_path:{args.data_path} synthetic:{args.use_synthetic_data} train_shards:{train_files} val_shards:{val_files}")
    log0(f"probe_steps:{args.probe_steps} batch_size:{args.batch_size} eval_batches:{args.eval_batches} lr:{args.lr}")

    warmup_steps = max(1, args.probe_steps // 10)  # 10% warmup
    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        # cosine decay from 1.0 to 0.05 over remaining steps
        progress = (step - warmup_steps) / max(1, args.probe_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    t0 = time.perf_counter()
    for step in range(1, args.probe_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        current_patches, next_patches = make_patch_pairs(train_stream, args.batch_size, args.patch_size, device)
        loss, aux = model.loss_on_pairs(current_patches, next_patches)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        # Cosine EMA schedule: ramp from ema_start to ema_end (BYOL/DINO style)
        ema_progress = step / args.probe_steps
        ema_decay = args.target_ema_end - (args.target_ema_end - args.target_ema_start) * (1.0 + math.cos(math.pi * ema_progress)) / 2.0
        model.update_targets_ema(ema_decay)
        if step <= 10 or step % args.log_every == 0:
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            patch_pairs_seen = step * args.batch_size
            pairs_per_s = patch_pairs_seen / max(elapsed_ms / 1000.0, 1e-9)
            log0(
                f"step:{step}/{args.probe_steps} train_loss:{loss.item():.6f} "
                f"mse:{aux['mse'].item():.6f} infonce:{aux['infonce'].item():.6f} "
                f"online_infonce:{aux['online_infonce'].item():.6f} repr_infonce:{aux['repr_infonce'].item():.6f} cosine:{aux['cosine'].item():.6f} "
                f"pred_std:{aux['pred_std'].item():.6f} target_std:{aux['target_std'].item():.6f} "
                f"cov_loss:{aux['cov_loss'].item():.6f} pairs_per_s:{pairs_per_s:.1f}"
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
