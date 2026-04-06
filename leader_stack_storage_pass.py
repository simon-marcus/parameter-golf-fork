#!/usr/bin/env python3
from __future__ import annotations

import io
import lzma
import os
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.distributed as dist

import train_gpt_leader_stack_jepa as wrapper


def build_model(args: wrapper.Hyperparameters, device: torch.device, *, mtp_num_heads: int, mtp_loss_weight: float):
    model = wrapper.GPTLeaderStackJEPA(
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
        mtp_num_heads=mtp_num_heads,
        mtp_loss_weight=mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
        activation_mode=args.activation_mode,
        activation_neg_slope=args.activation_neg_slope,
        asymmetric_square_init=args.asymmetric_square_init,
        gated_square_beta_init=args.gated_square_beta_init,
    ).to(device).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for module in model.modules():
        if isinstance(module, wrapper.leaderstack.CastedLinear):
            module.float()
    wrapper.leaderstack.restore_low_dim_params_to_fp32(model)
    return model


def maybe_restore_aliases(state_dict: dict[str, torch.Tensor]) -> None:
    for alias, backing in (
        ("jepa_in.weight", "blocks.0.jepa_in.weight"),
        ("jepa_out.weight", "blocks.0.jepa_out.weight"),
    ):
        if alias not in state_dict and backing in state_dict:
            state_dict[alias] = state_dict[backing]


def main() -> None:
    args = wrapper.Hyperparameters()
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "final_model.pt")
    quant_cats = set(filter(None, os.environ.get("QUANT_CATS", "mlp,attn").split(",")))
    drop_alias_export = bool(int(os.environ.get("DROP_JEPA_ALIAS_EXPORT", "0")))

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    def log0(msg: str) -> None:
        if master_process:
            print(msg, flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"SentencePiece .model required: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = wrapper.leaderstack.load_validation_tokens(args.val_files, val_seq_len, args.val_tokens_limit)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = wrapper.leaderstack.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    model = build_model(args, device, mtp_num_heads=args.mtp_num_heads, mtp_loss_weight=args.mtp_loss_weight)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    export_sd = {k: v for k, v in model.state_dict().items() if "mtp_heads" not in k}
    if drop_alias_export:
        export_sd.pop("jepa_in.weight", None)
        export_sd.pop("jepa_out.weight", None)

    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = wrapper.leaderstack._unbank_state_dict(sd_cpu, args.num_layers)
    quant_result, quant_meta = wrapper.leaderstack.mixed_quantize_int6(unbanked_sd, quant_cats)

    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=6)
    code_bytes = Path(wrapper.BASE_PATH).stat().st_size
    if master_process:
        log0(f"storage_pass checkpoint={checkpoint_path}")
        log0(f"storage_pass quant_cats={sorted(quant_cats)} drop_alias_export={drop_alias_export}")
        log0(f"storage_pass quant_bytes={len(quant_blob)}")
        log0(f"storage_pass code_bytes={code_bytes}")
        log0(f"storage_pass estimated_total={len(quant_blob) + code_bytes}")

    if distributed:
        dist.barrier()

    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")
    deq_unbanked = wrapper.leaderstack.dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = wrapper.leaderstack._rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    maybe_restore_aliases(deq_state)

    eval_model = build_model(args, device, mtp_num_heads=0, mtp_loss_weight=0.0)
    eval_model.load_state_dict(deq_state, strict=True)

    grad_accum_steps = 8 // world_size
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = wrapper.leaderstack.eval_val(
        args,
        eval_model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"storage_pass_int6_roundtrip val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )

    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = wrapper.leaderstack.eval_val_sliding(
            args,
            eval_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"storage_pass_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} "
            f"val_bpb:{sw_val_bpb:.8f} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"storage_pass_final_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
