"""
Second-generation patch-first byte-level JEPA trainer.

This version replaces the weak flatten/reducer patch representation with:
- a causal local byte encoder that turns each byte patch into a latent token
- the usual global transformer over shifted patch latents
- a causal local decoder conditioned on patch context to emit exact byte logits
- JEPA on future patch latents
"""

from __future__ import annotations

import io
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

import train_jepa_baseline as jbase
import train_jepa_patched as v1

naivebaseline = jbase.naivebaseline


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_byte260")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_kind = os.environ.get("TOKENIZER_KIND", "byte").strip().lower()
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_pure_byte_260.json")
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

    patch_size = int(os.environ.get("PATCH_SIZE", 4))
    local_layers = int(os.environ.get("LOCAL_LAYERS", 2))
    local_heads = int(os.environ.get("LOCAL_HEADS", 4))
    jepa_loss_weight = float(os.environ.get("JEPA_LOSS_WEIGHT", 0.1))
    jepa_var_weight = float(os.environ.get("JEPA_VAR_WEIGHT", 0.02))
    jepa_std_target = float(os.environ.get("JEPA_STD_TARGET", 0.5))
    target_ema = float(os.environ.get("TARGET_EMA", 0.995))
    model_ema_enabled = bool(int(os.environ.get("MODEL_EMA_ENABLED", "1")))
    model_ema_decay = float(os.environ.get("MODEL_EMA_DECAY", 0.997))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "0")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    use_compile = bool(int(os.environ.get("USE_COMPILE", "0")))


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


class TinyCausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = naivebaseline.CastedLinear(dim, 3 * dim, bias=False)
        self.proj = naivebaseline.CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return x + self.proj(y)


class TinyLocalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, negative_slope: float):
        super().__init__()
        self.attn = TinyCausalSelfAttention(dim, num_heads)
        self.mlp = LeakyReluSquaredMLP(dim, mlp_mult, negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class LocalPatchEncoder(nn.Module):
    def __init__(self, model_dim: int, patch_size: int, local_layers: int, local_heads: int, negative_slope: float):
        super().__init__()
        self.patch_size = patch_size
        self.pos_emb = nn.Parameter(torch.zeros(patch_size, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [TinyLocalBlock(model_dim, local_heads, 2, negative_slope) for _ in range(local_layers)]
        )
        self.out_proj = naivebaseline.CastedLinear(2 * model_dim, model_dim, bias=False)

    def forward(self, byte_emb: Tensor) -> Tensor:
        # byte_emb: [B, P, S, D]
        bsz, num_patches, patch_size, dim = byte_emb.shape
        x = byte_emb.reshape(bsz * num_patches, patch_size, dim)
        x = x + self.pos_emb.to(dtype=x.dtype)[None, :, :]
        for block in self.blocks:
            x = block(x)
        last = x[:, -1, :]
        mean = x.mean(dim=1)
        return self.out_proj(torch.cat((last, mean), dim=-1)).reshape(bsz, num_patches, dim)


class LocalPatchDecoder(nn.Module):
    def __init__(self, model_dim: int, patch_size: int, local_layers: int, local_heads: int, negative_slope: float):
        super().__init__()
        self.patch_size = patch_size
        self.pos_emb = nn.Parameter(torch.zeros(patch_size + 1, model_dim, dtype=torch.float32))
        self.in_proj = naivebaseline.CastedLinear(model_dim, model_dim, bias=False)
        self.blocks = nn.ModuleList(
            [TinyLocalBlock(model_dim, local_heads, 2, negative_slope) for _ in range(local_layers)]
        )
        self.fuse = naivebaseline.CastedLinear(2 * model_dim, model_dim, bias=False)

    def forward(self, patch_context: Tensor, tok_emb: nn.Embedding, target_ids: Tensor) -> Tensor:
        # patch_context: [B, P, D], target_ids: [B, P, S]
        bsz, num_patches, dim = patch_context.shape
        flat_target = target_ids.reshape(bsz * num_patches, self.patch_size)
        ctx = patch_context.reshape(bsz * num_patches, dim)
        prev_ids = flat_target[:, :-1]
        prev_emb = tok_emb(prev_ids) if self.patch_size > 1 else ctx.new_zeros((ctx.size(0), 0, dim))
        bos = ctx[:, None, :]
        x = torch.cat((bos, prev_emb), dim=1)
        x = self.in_proj(x) + self.pos_emb[: self.patch_size].to(dtype=x.dtype)[None, :, :]
        for block in self.blocks:
            x = block(x)
        byte_states = x[:, 1:, :] if self.patch_size > 1 else x[:, :0, :]
        first_state = x[:, 0:1, :]
        byte_states = torch.cat((first_state, byte_states), dim=1)
        ctx_expand = ctx[:, None, :].expand(-1, self.patch_size, -1)
        return self.fuse(torch.cat((byte_states, ctx_expand), dim=-1))


class PatchFirstJEPAv2(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        activation_neg_slope: float,
        patch_size: int,
        local_layers: int,
        local_heads: int,
        jepa_var_weight: float,
        jepa_std_target: float,
    ):
        super().__init__()
        if patch_size <= 1:
            raise ValueError(f"PATCH_SIZE must be >= 2, got {patch_size}")
        self.patch_size = patch_size
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.jepa_var_weight = jepa_var_weight
        self.jepa_std_target = jepa_std_target

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.patch_encoder = LocalPatchEncoder(model_dim, patch_size, local_layers, local_heads, activation_neg_slope)
        self.patch_bos = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
        self.decoder = LocalPatchDecoder(model_dim, patch_size, local_layers, local_heads, activation_neg_slope)
        self.jepa_in = naivebaseline.CastedLinear(model_dim, model_dim, bias=False)
        self.jepa_out = naivebaseline.CastedLinear(model_dim, model_dim, bias=False)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                naivebaseline.Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for _ in range(num_layers)
            ]
        )
        for block in self.blocks:
            block.mlp = LeakyReluSquaredMLP(model_dim, mlp_mult, activation_neg_slope)
        self.final_norm = naivebaseline.RMSNorm()
        self.lm_head = None if tie_embeddings else naivebaseline.CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        nn.init.normal_(self.patch_bos, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def encode_patch_tokens(self, input_ids: Tensor) -> Tensor:
        emb = self.tok_emb(input_ids)
        bsz, seq_len, dim = emb.shape
        if seq_len % self.patch_size != 0:
            raise ValueError(f"sequence length {seq_len} must be divisible by PATCH_SIZE={self.patch_size}")
        num_patches = seq_len // self.patch_size
        patch_emb = emb.reshape(bsz, num_patches, self.patch_size, dim)
        patch_tokens = self.patch_encoder(patch_emb)
        patch_tokens = F.rms_norm(patch_tokens, (patch_tokens.size(-1),))
        bos = self.patch_bos.to(dtype=patch_tokens.dtype)[None, None, :].expand(bsz, 1, -1)
        return torch.cat((bos, patch_tokens[:, :-1, :]), dim=1)

    def encode_patch_context(self, input_ids: Tensor) -> Tensor:
        x = self.encode_patch_tokens(input_ids)
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

    def _logits_from_hidden(self, hidden: Tensor) -> Tensor:
        if self.tie_embeddings:
            logits_proj = F.linear(hidden, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(hidden)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def decode_byte_logits(self, patch_hidden: Tensor, target_ids: Tensor) -> Tensor:
        bsz, num_patches, _ = patch_hidden.shape
        target_patches = target_ids.reshape(bsz, num_patches, self.patch_size)
        byte_hidden = self.decoder(F.rms_norm(patch_hidden, (patch_hidden.size(-1),)), self.tok_emb, target_patches)
        return self._logits_from_hidden(byte_hidden.reshape(bsz * num_patches * self.patch_size, -1))

    def predict_patch_latents(self, patch_hidden: Tensor) -> Tensor:
        x = F.rms_norm(patch_hidden, (patch_hidden.size(-1),))
        x = torch.relu(self.jepa_in(x))
        return self.jepa_out(x.square())

    def compute_jepa_loss(
        self,
        target_ids: Tensor,
        patch_hidden: Tensor,
        target_tok_emb: nn.Embedding,
        target_patch_encoder: nn.Module,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if patch_hidden.size(1) < 2:
            zero = patch_hidden.new_zeros(())
            return zero, zero, zero
        pred = self.predict_patch_latents(patch_hidden[:, :-1, :]).float()
        bsz, _, dim = patch_hidden.shape
        next_patches = target_ids.reshape(bsz, -1, self.patch_size)[:, 1:, :]
        with torch.no_grad():
            target_patch_emb = target_tok_emb(next_patches).reshape(bsz, -1, self.patch_size, dim)
            target = target_patch_encoder(target_patch_emb.float())
            target = F.rms_norm(target.float(), (target.size(-1),))
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        jepa_loss = F.mse_loss(pred_n, target_n, reduction="mean")
        pred_std = pred.std(dim=(0, 1), correction=0)
        var_loss = torch.relu(self.jepa_std_target - pred_std).mean()
        return jepa_loss, var_loss, pred_std.mean()

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        *,
        compute_jepa: bool = False,
        target_tok_emb: nn.Embedding | None = None,
        target_patch_encoder: nn.Module | None = None,
        jepa_loss_weight: float = 0.0,
        return_aux: bool = False,
    ):
        patch_hidden = self.encode_patch_context(input_ids)
        logits = self.decode_byte_logits(patch_hidden, target_ids)
        ce_loss = F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")
        total_loss = ce_loss
        jepa_loss = patch_hidden.new_zeros(())
        var_loss = patch_hidden.new_zeros(())
        pred_std_mean = patch_hidden.new_zeros(())
        if compute_jepa and target_tok_emb is not None and target_patch_encoder is not None and jepa_loss_weight > 0.0:
            jepa_loss, var_loss, pred_std_mean = self.compute_jepa_loss(
                target_ids, patch_hidden, target_tok_emb, target_patch_encoder
            )
            total_loss = total_loss + jepa_loss_weight * jepa_loss + self.jepa_var_weight * var_loss
        if return_aux:
            return total_loss, ce_loss.detach(), jepa_loss.detach(), var_loss.detach(), pred_std_mean.detach()
        return total_loss


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
    (base_bytes_lut, has_leading_space_lut, is_boundary_token_lut), tokenizer_source = jbase.load_metric_luts(args, device)
    log0(f"val_bpb:enabled tokenizer_kind={args.tokenizer_kind} tokenizer_path={args.tokenizer_path} tokenizer_source={tokenizer_source}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    if args.train_seq_len % args.patch_size != 0:
        raise ValueError(f"TRAIN_SEQ_LEN={args.train_seq_len} must be divisible by PATCH_SIZE={args.patch_size}")

    base_model = PatchFirstJEPAv2(
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
        patch_size=args.patch_size,
        local_layers=args.local_layers,
        local_heads=args.local_heads,
        jepa_var_weight=args.jepa_var_weight,
        jepa_std_target=args.jepa_std_target,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, naivebaseline.CastedLinear):
            module.float()
    naivebaseline.restore_low_dim_params_to_fp32(base_model)

    target_tok_emb = nn.Embedding(args.vocab_size, args.model_dim, device=device, dtype=torch.float32)
    target_patch_encoder = LocalPatchEncoder(
        args.model_dim,
        args.patch_size,
        args.local_layers,
        args.local_heads,
        args.activation_neg_slope,
    ).to(device=device, dtype=torch.float32)
    v1.sync_target_modules(target_tok_emb, base_model.tok_emb, target_patch_encoder, base_model.patch_encoder)
    target_tok_emb.requires_grad_(False)
    target_patch_encoder.requires_grad_(False)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if args.use_compile else base_model
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        if distributed
        else compiled_model
    )

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in naivebaseline.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    patch_matrix_params = [p for p in list(base_model.patch_encoder.parameters()) + list(base_model.decoder.parameters()) if p.ndim == 2]
    patch_scalar_params = [p for p in list(base_model.patch_encoder.parameters()) + list(base_model.decoder.parameters()) if p.ndim < 2]
    matrix_params.extend(patch_matrix_params + [base_model.jepa_in.weight, base_model.jepa_out.weight])
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in naivebaseline.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params.extend(patch_scalar_params + [base_model.patch_bos])
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
        f"model_family:patch_first_jepa_v2 patch_size:{args.patch_size} local_layers:{args.local_layers} "
        f"local_heads:{args.local_heads} jepa_loss_weight:{args.jepa_loss_weight} "
        f"target_ema:{args.target_ema} use_compile:{args.use_compile}"
    )
    log0("activation_mode:leaky_relu_sq")
    log0("patch_embed_mode:causal_local_encoder")
    log0("decoder_mode:causal_local_decoder")
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
            v1.clear_rotary_caches(base_model)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        ce_loss_sum = torch.zeros((), device=device)
        jepa_loss_sum = torch.zeros((), device=device)
        jepa_var_sum = torch.zeros((), device=device)
        jepa_pred_std_sum = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss, ce_loss, jepa_loss, var_loss, pred_std_mean = model(
                    x,
                    y,
                    compute_jepa=args.jepa_loss_weight > 0.0,
                    target_tok_emb=target_tok_emb,
                    target_patch_encoder=target_patch_encoder,
                    jepa_loss_weight=args.jepa_loss_weight,
                    return_aux=True,
                )
            train_loss += loss.detach()
            ce_loss_sum += ce_loss
            jepa_loss_sum += jepa_loss
            jepa_var_sum += var_loss
            jepa_pred_std_sum += pred_std_mean
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        ce_loss_sum /= grad_accum_steps
        jepa_loss_sum /= grad_accum_steps
        jepa_var_sum /= grad_accum_steps
        jepa_pred_std_sum /= grad_accum_steps

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
        v1.update_target_modules_ema(target_tok_emb, base_model.tok_emb, target_patch_encoder, base_model.patch_encoder, args.target_ema)
        if model_ema_state is not None:
            with torch.no_grad():
                for name, tensor in base_model.state_dict().items():
                    model_ema_state[name].mul_(args.model_ema_decay).add_(tensor.detach().float(), alpha=1.0 - args.model_ema_decay)
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
        should_log_train = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"ce_loss:{ce_loss_sum.item():.4f} jepa_loss:{jepa_loss_sum.item():.4f} "
                f"jepa_var:{jepa_var_sum.item():.4f} jepa_pred_std:{jepa_pred_std_sum.item():.4f} "
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
        avg_state = {name: (tensor / swa_count).to(dtype=current_state[name].dtype) for name, tensor in swa_state.items()}
        base_model.load_state_dict(avg_state, strict=True)
    elif model_ema_state is not None:
        log0("ema:applying EMA weights")
        current_state = base_model.state_dict()
        avg_state = {name: tensor.to(dtype=current_state[name].dtype) for name, tensor in model_ema_state.items()}
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
    v1.clear_rotary_caches(base_model)

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
    v1.clear_rotary_caches(base_model)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
