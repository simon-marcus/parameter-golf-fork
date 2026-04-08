"""
Minimal-diff JEPA isolation lane built on top of the 2026-03-17 NaiveBaseline.

This keeps the March 17 baseline architecture, optimizer split, and export path,
but switches to byte-level input by default and adds the smallest possible JEPA
auxiliary loss: predict the next byte-patch embedding target directly from the
current patch state, with a stop-grad EMA target embedding.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


BASELINE_PATH = Path(__file__).resolve().parent / "records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py"
BASELINE_SPEC = importlib.util.spec_from_file_location("naivebaseline_20260317", BASELINE_PATH)
if BASELINE_SPEC is None or BASELINE_SPEC.loader is None:
    raise RuntimeError(f"Unable to import baseline module from {BASELINE_PATH}")
naivebaseline = importlib.util.module_from_spec(BASELINE_SPEC)
BASELINE_SPEC.loader.exec_module(naivebaseline)


class Hyperparameters:
    # Match the March 17 baseline defaults except where the JEPA isolation lane
    # needs byte-level input and JEPA-specific knobs.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_byte260")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_kind = os.environ.get("TOKENIZER_KIND", "byte").strip().lower()
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 260 if tokenizer_kind == "byte" else 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    activation_neg_slope = float(os.environ.get("ACTIVATION_NEG_SLOPE", 0.5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    xsa_all = bool(int(os.environ.get("XSA_ALL", "0")))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    patch_size = int(os.environ.get("PATCH_SIZE", 8))
    jepa_loss_weight = float(os.environ.get("JEPA_LOSS_WEIGHT", 0.2))
    jepa_var_weight = float(os.environ.get("JEPA_VAR_WEIGHT", 0.02))
    jepa_std_target = float(os.environ.get("JEPA_STD_TARGET", 0.5))
    target_ema = float(os.environ.get("TARGET_EMA", 0.995))
    target_encoder_mode = os.environ.get("TARGET_ENCODER_MODE", "patch_trainable").strip().lower()
    target_encoder_ema = float(os.environ.get("TARGET_ENCODER_EMA", os.environ.get("TARGET_EMA", "0.995")))
    model_ema_enabled = bool(int(os.environ.get("MODEL_EMA_ENABLED", "1")))
    model_ema_decay = float(os.environ.get("MODEL_EMA_DECAY", 0.997))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "0")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    use_compile = bool(int(os.environ.get("USE_COMPILE", "0")))


def build_byte_luts(vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.ones((vocab_size,), dtype=torch.int16, device=device),
        torch.zeros((vocab_size,), dtype=torch.bool, device=device),
        torch.zeros((vocab_size,), dtype=torch.bool, device=device),
    )


def load_metric_luts(args: Hyperparameters, device: torch.device) -> tuple[tuple[Tensor, Tensor, Tensor], str]:
    if args.tokenizer_kind == "byte":
        return build_byte_luts(args.vocab_size, device), "builtin-byte"
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"SentencePiece tokenizer must end with .model, got {args.tokenizer_path}")
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required for TOKENIZER_KIND=sentencepiece") from exc
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    return naivebaseline.build_sentencepiece_luts(sp, args.vocab_size, device), args.tokenizer_path


class LeakyReluSquaredMLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, negative_slope: float):
        super().__init__()
        hidden = mlp_mult * dim
        self.negative_slope = negative_slope
        self.fc = naivebaseline.CastedLinear(dim, hidden, bias=False)
        self.proj = naivebaseline.CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=self.negative_slope)
        return self.proj(x.square())


class XSACausalSelfAttention(naivebaseline.CausalSelfAttention):
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        # Baseline attention uses [B, H, T, D]; adapt the XSA projection subtraction
        # from the stronger leader stack without introducing new parameters.
        bsz, num_heads, seqlen, head_dim = y.shape
        num_kv_heads = v.size(1)
        group = num_heads // num_kv_heads
        y_bt = y.transpose(1, 2)  # [B, T, H, D]
        v_bt = v.transpose(1, 2)  # [B, T, Hkv, D]
        y_g = y_bt.reshape(bsz, seqlen, num_kv_heads, group, head_dim)
        vn = F.normalize(v_bt, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(bsz, seqlen, num_heads, head_dim).transpose(1, 2).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = naivebaseline.apply_rotary_emb(q, cos, sin)
        k = naivebaseline.apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = self._xsa_efficient(y, v)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class TargetPatchEncoder(nn.Module):
    def __init__(self, model_dim: int, patch_size: int):
        super().__init__()
        if model_dim % 4 != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by 4 for TargetPatchEncoder")
        self.local_pos = nn.Parameter(torch.zeros(patch_size, model_dim, dtype=torch.float32))
        self.in_proj = naivebaseline.CastedLinear(model_dim, model_dim, bias=False)
        self.gate_proj = naivebaseline.CastedLinear(model_dim, model_dim, bias=False)
        self.local_conv = nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=4, groups=model_dim, bias=False)
        self.intra_num_heads = 4
        self.intra_head_dim = model_dim // self.intra_num_heads
        self.intra_qkv = naivebaseline.CastedLinear(model_dim, 3 * model_dim, bias=False)
        self.intra_out = naivebaseline.CastedLinear(model_dim, model_dim, bias=False)
        self.num_pool_heads = 4
        self.attn_pool = naivebaseline.CastedLinear(model_dim, self.num_pool_heads, bias=False)
        self.attn_temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.post_mlp_up = naivebaseline.CastedLinear(model_dim, model_dim, bias=False)
        self.post_mlp_down = naivebaseline.CastedLinear(model_dim, model_dim, bias=False)
        self.out_proj = naivebaseline.CastedLinear(model_dim, model_dim, bias=False)
        nn.init.normal_(self.local_pos, mean=0.0, std=0.01)

    def forward(self, patch_byte_emb: Tensor) -> Tensor:
        x = patch_byte_emb + self.local_pos[None, None, :, :].to(dtype=patch_byte_emb.dtype)
        x = F.rms_norm(x, (x.size(-1),))
        x = F.silu(self.gate_proj(x)) * self.in_proj(x)
        bsz, num_patches, patch_size, model_dim = x.shape
        x_flat = x.reshape(bsz * num_patches, patch_size, model_dim).transpose(1, 2)
        conv_out = self.local_conv(x_flat)[:, :, :patch_size]
        x = x + conv_out.transpose(1, 2).reshape(bsz, num_patches, patch_size, model_dim)
        x_sa = F.rms_norm(x, (x.size(-1),))
        x_flat_sa = x_sa.reshape(bsz * num_patches, patch_size, model_dim)
        qkv = self.intra_qkv(x_flat_sa).reshape(
            bsz * num_patches,
            patch_size,
            3,
            self.intra_num_heads,
            self.intra_head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.intra_head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        sa_out = (attn @ v).transpose(1, 2).reshape(bsz * num_patches, patch_size, model_dim)
        x = x + self.intra_out(sa_out).reshape(bsz, num_patches, patch_size, model_dim)
        x = F.rms_norm(x, (x.size(-1),))
        head_dim = model_dim // self.num_pool_heads
        temp = self.attn_temperature.clamp(min=0.01).to(dtype=x.dtype)
        attn_logits = self.attn_pool(x) / temp
        attn_weights = F.softmax(attn_logits.permute(0, 1, 3, 2), dim=-1)
        x_heads = x.view(bsz, num_patches, patch_size, self.num_pool_heads, head_dim).permute(0, 1, 3, 2, 4)
        x = (attn_weights.unsqueeze(-1) * x_heads).sum(dim=3).reshape(bsz, num_patches, model_dim)
        x = x + self.post_mlp_down(F.silu(self.post_mlp_up(F.rms_norm(x, (x.size(-1),)))))
        x = self.out_proj(x)
        return F.rms_norm(x, (x.size(-1),))


class GPTBaselineJEPA(naivebaseline.GPT):
    def __init__(
        self,
        *,
        patch_size: int,
        jepa_var_weight: float,
        jepa_std_target: float,
        target_encoder_mode: str,
        activation_neg_slope: float,
        xsa_all: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if patch_size <= 1:
            raise ValueError(f"PATCH_SIZE must be >= 2, got {patch_size}")
        self.patch_size = patch_size
        self.jepa_var_weight = jepa_var_weight
        self.jepa_std_target = jepa_std_target
        self.target_encoder_mode = target_encoder_mode
        self.jepa_in = naivebaseline.CastedLinear(self.tok_emb.embedding_dim, self.tok_emb.embedding_dim, bias=False)
        self.jepa_out = naivebaseline.CastedLinear(self.tok_emb.embedding_dim, self.tok_emb.embedding_dim, bias=False)
        self.target_patch_encoder = (
            TargetPatchEncoder(self.tok_emb.embedding_dim, patch_size)
            if target_encoder_mode in {"patch_trainable", "patch_nograd", "patch_ema"}
            else None
        )
        self.target_patch_encoder_ema = (
            TargetPatchEncoder(self.tok_emb.embedding_dim, patch_size) if target_encoder_mode == "patch_ema" else None
        )
        self.activation_neg_slope = activation_neg_slope
        for block in self.blocks:
            block.mlp = LeakyReluSquaredMLP(self.tok_emb.embedding_dim, kwargs["mlp_mult"], self.activation_neg_slope)
        if xsa_all:
            for i, block in enumerate(self.blocks):
                old_attn = block.attn
                new_attn = XSACausalSelfAttention(
                    self.tok_emb.embedding_dim,
                    kwargs["num_heads"],
                    kwargs["num_kv_heads"],
                    kwargs["rope_base"],
                    kwargs["qk_gain_init"],
                )
                new_attn.load_state_dict(old_attn.state_dict(), strict=True)
                block.attn = new_attn

    def encode_hidden(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def logits_from_hidden(self, hidden: Tensor) -> Tensor:
        x = hidden.reshape(-1, hidden.size(-1))
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def predict_patch_latents(self, patch_states: Tensor) -> Tensor:
        x = F.rms_norm(patch_states, (patch_states.size(-1),))
        x = torch.relu(self.jepa_in(x))
        return self.jepa_out(x.square())

    def compute_jepa_loss(
        self,
        input_ids: Tensor,
        hidden: Tensor,
        target_tok_emb: nn.Embedding,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if hidden.size(1) % self.patch_size != 0:
            raise ValueError(
                f"sequence length {hidden.size(1)} must be divisible by PATCH_SIZE={self.patch_size}"
            )
        patch_states = hidden[:, self.patch_size - 1 :: self.patch_size, :]
        if patch_states.size(1) < 2:
            zero = hidden.new_zeros(())
            return zero, zero, zero, zero
        pred = self.predict_patch_latents(patch_states[:, :-1, :]).float()
        target_patches = input_ids.reshape(input_ids.size(0), -1, self.patch_size)[:, 1:, :]
        pred_n = F.normalize(pred, dim=-1)
        if self.target_encoder_mode == "meanpool":
            with torch.no_grad():
                target = target_tok_emb(target_patches).mean(dim=2)
                target = F.rms_norm(target.float(), (target.size(-1),))
            target_n = F.normalize(target, dim=-1)
            jepa_loss = F.mse_loss(pred_n, target_n, reduction="mean")
        else:
            if self.target_patch_encoder is None:
                raise RuntimeError(f"target_patch_encoder is required for mode={self.target_encoder_mode}")
            target_patch_emb = target_tok_emb(target_patches).to(dtype=hidden.dtype)
            if self.target_encoder_mode == "patch_nograd":
                with torch.no_grad():
                    target = self.target_patch_encoder(target_patch_emb).float()
                    target = F.rms_norm(target, (target.size(-1),))
                target_n = F.normalize(target, dim=-1)
                jepa_loss = F.mse_loss(pred_n, target_n, reduction="mean")
            elif self.target_encoder_mode == "patch_ema":
                if self.target_patch_encoder_ema is None:
                    raise RuntimeError("target_patch_encoder_ema is required for TARGET_ENCODER_MODE=patch_ema")
                online_target = self.target_patch_encoder(target_patch_emb).float()
                online_target = F.rms_norm(online_target, (online_target.size(-1),))
                online_target_n = F.normalize(online_target, dim=-1)
                with torch.no_grad():
                    target = self.target_patch_encoder_ema(target_patch_emb).float()
                    target = F.rms_norm(target, (target.size(-1),))
                target_n = F.normalize(target, dim=-1)
                jepa_loss = 0.5 * F.mse_loss(pred_n, target_n, reduction="mean")
                jepa_loss = jepa_loss + 0.5 * F.mse_loss(pred_n, online_target_n, reduction="mean")
            elif self.target_encoder_mode == "patch_trainable":
                target = self.target_patch_encoder(target_patch_emb).float()
                target = F.rms_norm(target, (target.size(-1),))
                target_n = F.normalize(target, dim=-1)
                jepa_loss = F.mse_loss(pred_n, target_n, reduction="mean")
            else:
                raise ValueError(f"Unknown TARGET_ENCODER_MODE={self.target_encoder_mode}")
        pred_std = pred.std(dim=(0, 1), correction=0)
        target_std = target.float().std(dim=(0, 1), correction=0)
        var_loss = torch.relu(self.jepa_std_target - pred_std).mean()
        return jepa_loss, var_loss, pred_std.mean(), target_std.mean()

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        *,
        compute_jepa: bool = False,
        target_tok_emb: nn.Embedding | None = None,
        jepa_loss_weight: float = 0.0,
        return_aux: bool = False,
    ):
        hidden = self.encode_hidden(input_ids)
        logits = self.logits_from_hidden(hidden)
        targets = target_ids.reshape(-1)
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        total_loss = ce_loss
        jepa_loss = hidden.new_zeros(())
        var_loss = hidden.new_zeros(())
        pred_std_mean = hidden.new_zeros(())
        target_std_mean = hidden.new_zeros(())
        if compute_jepa and target_tok_emb is not None and jepa_loss_weight > 0.0:
            jepa_loss, var_loss, pred_std_mean, target_std_mean = self.compute_jepa_loss(
                input_ids,
                hidden,
                target_tok_emb,
            )
            total_loss = total_loss + jepa_loss_weight * jepa_loss + self.jepa_var_weight * var_loss
        if return_aux:
            return (
                total_loss,
                ce_loss.detach(),
                jepa_loss.detach(),
                var_loss.detach(),
                pred_std_mean.detach(),
                target_std_mean.detach(),
            )
        return total_loss


@torch.no_grad()
def sync_target_embedding(target_tok_emb: nn.Embedding, online_tok_emb: nn.Embedding) -> None:
    target_tok_emb.weight.copy_(online_tok_emb.weight.detach().float())


@torch.no_grad()
def update_target_embedding_ema(target_tok_emb: nn.Embedding, online_tok_emb: nn.Embedding, decay: float) -> None:
    target_tok_emb.weight.mul_(decay).add_(online_tok_emb.weight.detach().float(), alpha=1.0 - decay)


@torch.no_grad()
def sync_module_state(target_module: nn.Module | None, online_module: nn.Module | None) -> None:
    if target_module is None or online_module is None:
        return
    target_module.load_state_dict(online_module.state_dict(), strict=True)


@torch.no_grad()
def update_module_ema(target_module: nn.Module | None, online_module: nn.Module | None, decay: float) -> None:
    if target_module is None or online_module is None:
        return
    for target_param, online_param in zip(target_module.parameters(), online_module.parameters(), strict=True):
        target_param.data.mul_(decay).add_(online_param.detach().float(), alpha=1.0 - decay)
    for target_buf, online_buf in zip(target_module.buffers(), online_module.buffers(), strict=True):
        target_buf.copy_(online_buf.detach())


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    naivebaseline.zeropower_via_newtonschulz5 = torch.compile(naivebaseline.zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = naivebaseline.load_validation_tokens(args.val_files, args.train_seq_len)
    (base_bytes_lut, has_leading_space_lut, is_boundary_token_lut), tokenizer_source = load_metric_luts(args, device)
    log0(f"val_bpb:enabled tokenizer_kind={args.tokenizer_kind} tokenizer_path={args.tokenizer_path} tokenizer_source={tokenizer_source}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    if args.train_seq_len % args.patch_size != 0:
        raise ValueError(
            f"TRAIN_SEQ_LEN={args.train_seq_len} must be divisible by PATCH_SIZE={args.patch_size}"
        )
    valid_target_encoder_modes = {"meanpool", "patch_trainable", "patch_nograd", "patch_ema"}
    if args.target_encoder_mode not in valid_target_encoder_modes:
        raise ValueError(
            f"TARGET_ENCODER_MODE must be one of {sorted(valid_target_encoder_modes)}, got {args.target_encoder_mode}"
        )

    base_model = GPTBaselineJEPA(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        activation_neg_slope=args.activation_neg_slope,
        xsa_all=args.xsa_all,
        patch_size=args.patch_size,
        jepa_var_weight=args.jepa_var_weight,
        jepa_std_target=args.jepa_std_target,
        target_encoder_mode=args.target_encoder_mode,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, naivebaseline.CastedLinear):
            module.float()
    naivebaseline.restore_low_dim_params_to_fp32(base_model)
    sync_module_state(base_model.target_patch_encoder_ema, base_model.target_patch_encoder)
    target_tok_emb = nn.Embedding(args.vocab_size, args.model_dim, device=device, dtype=torch.float32)
    sync_target_embedding(target_tok_emb, base_model.tok_emb)
    target_tok_emb.requires_grad_(False)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if args.use_compile else base_model
    model: nn.Module = (
        DDP(
            compiled_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if distributed
        else compiled_model
    )

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in naivebaseline.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    target_encoder_matrix_params = []
    target_encoder_scalar_params = []
    if base_model.target_patch_encoder is not None:
        target_encoder_matrix_params = [p for p in base_model.target_patch_encoder.parameters() if p.ndim == 2]
        target_encoder_scalar_params = [p for p in base_model.target_patch_encoder.parameters() if p.ndim != 2]
    matrix_params.extend([base_model.jepa_in.weight, base_model.jepa_out.weight, *target_encoder_matrix_params])
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in naivebaseline.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params.extend(target_encoder_scalar_params)
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = naivebaseline.Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"model_family:naivebaseline_jepa patch_size:{args.patch_size} "
        f"jepa_loss_weight:{args.jepa_loss_weight} target_ema:{args.target_ema} "
        f"target_encoder_mode:{args.target_encoder_mode} target_encoder_ema:{args.target_encoder_ema} "
        f"use_compile:{args.use_compile}"
    )
    xsa_layers = list(range(args.num_layers)) if args.xsa_all else []
    log0(f"XSA:all={args.xsa_all} active_layers:{xsa_layers}")
    log0(f"activation_mode:leaky_relu_sq activation_neg_slope:{args.activation_neg_slope}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    train_loader = naivebaseline.DistributedTokenLoader(args.train_files, rank, world_size, device)
    model_ema_state = (
        {name: tensor.detach().float().clone() for name, tensor in base_model.state_dict().items()}
        if args.model_ema_enabled
        else None
    )
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_target_embedding = target_tok_emb.weight.detach().cpu().clone()
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(
                        x,
                        y,
                        compute_jepa=args.jepa_loss_weight > 0.0,
                        target_tok_emb=target_tok_emb,
                        jepa_loss_weight=args.jepa_loss_weight,
                    )
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            update_target_embedding_ema(target_tok_emb, base_model.tok_emb, args.target_ema)
            update_module_ema(base_model.target_patch_encoder_ema, base_model.target_patch_encoder, args.target_encoder_ema)
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        target_tok_emb.weight.copy_(initial_target_embedding.to(device=device, dtype=torch.float32))
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = naivebaseline.DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = naivebaseline.eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        ce_loss_sum = torch.zeros((), device=device)
        jepa_loss_sum = torch.zeros((), device=device)
        jepa_var_sum = torch.zeros((), device=device)
        jepa_pred_std_sum = torch.zeros((), device=device)
        jepa_target_std_sum = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss, ce_loss, jepa_loss, var_loss, pred_std_mean, target_std_mean = model(
                    x,
                    y,
                    compute_jepa=args.jepa_loss_weight > 0.0,
                    target_tok_emb=target_tok_emb,
                    jepa_loss_weight=args.jepa_loss_weight,
                    return_aux=True,
                )
            train_loss += loss.detach()
            ce_loss_sum += ce_loss
            jepa_loss_sum += jepa_loss
            jepa_var_sum += var_loss
            jepa_pred_std_sum += pred_std_mean
            jepa_target_std_sum += target_std_mean
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        ce_loss_sum /= grad_accum_steps
        jepa_loss_sum /= grad_accum_steps
        jepa_var_sum /= grad_accum_steps
        jepa_pred_std_sum /= grad_accum_steps
        jepa_target_std_sum /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        update_target_embedding_ema(target_tok_emb, base_model.tok_emb, args.target_ema)
        update_module_ema(base_model.target_patch_encoder_ema, base_model.target_patch_encoder, args.target_encoder_ema)
        if model_ema_state is not None:
            with torch.no_grad():
                for name, tensor in base_model.state_dict().items():
                    model_ema_state[name].mul_(args.model_ema_decay).add_(
                        tensor.detach().float(), alpha=1.0 - args.model_ema_decay
                    )
        if args.swa_enabled and scale < 0.2 and step > 0 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, tensor in base_model.state_dict().items():
                    swa_state[name] += tensor.detach().cpu()
                swa_count += 1
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"ce_loss:{ce_loss_sum.item():.4f} jepa_loss:{jepa_loss_sum.item():.4f} "
                f"jepa_var:{jepa_var_sum.item():.4f} jepa_pred_std:{jepa_pred_std_sum.item():.4f} "
                f"target_std:{jepa_target_std_sum.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    if args.swa_enabled and swa_state is not None and swa_count > 0:
        log0(f"swa:applying averages count:{swa_count}")
        current_state = base_model.state_dict()
        avg_state = {
            name: (tensor / swa_count).to(dtype=current_state[name].dtype)
            for name, tensor in swa_state.items()
        }
        base_model.load_state_dict(avg_state, strict=True)
    elif model_ema_state is not None:
        log0("ema:applying EMA weights")
        current_state = base_model.state_dict()
        avg_state = {
            name: tensor.to(dtype=current_state[name].dtype)
            for name, tensor in model_ema_state.items()
        }
        base_model.load_state_dict(avg_state, strict=True)

    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = naivebaseline.eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = naivebaseline.quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(naivebaseline.dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = naivebaseline.eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
