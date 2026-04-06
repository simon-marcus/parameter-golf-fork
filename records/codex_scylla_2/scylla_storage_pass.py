#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import io
import lzma
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "records" / "codex_scylla_2" / "train_gpt_legal_ttt.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("scylla_ttt_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base module from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def build_model(args, device: torch.device, *, mtp_num_heads: int, mtp_loss_weight: float):
    model = base.GPT(
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
        if isinstance(module, base.CastedLinear):
            module.float()
    base.restore_low_dim_params_to_fp32(model)
    return model


def find_alias_groups(state_dict: dict[str, torch.Tensor]) -> list[list[str]]:
    groups: dict[tuple[int, tuple[int, ...], tuple[int, ...], str], list[str]] = defaultdict(list)
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        untyped = tensor.untyped_storage()
        key = (untyped.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
        groups[key].append(name)
    return [sorted(names) for names in groups.values() if len(names) > 1]


def maybe_drop_keys(export_sd: dict[str, torch.Tensor], drop_keys: set[str]) -> None:
    for key in sorted(drop_keys):
        export_sd.pop(key, None)


def maybe_drop_prefixes(export_sd: dict[str, torch.Tensor], drop_prefixes: tuple[str, ...]) -> list[str]:
    dropped: list[str] = []
    if not drop_prefixes:
        return dropped
    for key in sorted(list(export_sd.keys())):
        if any(key.startswith(prefix) for prefix in drop_prefixes):
            export_sd.pop(key, None)
            dropped.append(key)
    return dropped


def maybe_zero_bigram_tensors(state_dict: dict[str, torch.Tensor]) -> list[str]:
    touched: list[str] = []
    for key, value in state_dict.items():
        if key == "bigram.scale":
            value.zero_()
            touched.append(key)
        elif key.startswith("bigram.") and key.endswith(".weight"):
            value.zero_()
            touched.append(key)
    return touched


def maybe_restore_zero_bigram_from_model(
    state_dict: dict[str, torch.Tensor], model: torch.nn.Module
) -> list[str]:
    restored: list[str] = []
    model_sd = model.state_dict()
    for key in ("bigram.scale", "bigram.embed.weight", "bigram.proj.weight"):
        if key in model_sd and key not in state_dict:
            state_dict[key] = torch.zeros_like(model_sd[key].detach().cpu())
            restored.append(key)
    return restored


def main() -> None:
    args = base.Hyperparameters()
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "final_model.pt")
    quant_cats = set(filter(None, os.environ.get("QUANT_CATS", "mlp,attn").split(",")))
    drop_keys = set(filter(None, os.environ.get("DROP_EXPORT_KEYS", "").split(",")))
    drop_prefixes = tuple(filter(None, os.environ.get("DROP_EXPORT_PREFIXES", "").split(",")))
    lzma_preset = int(os.environ.get("LZMA_PRESET", "6"))
    zero_bigram = bool(int(os.environ.get("ZERO_BIGRAM", "0")))
    run_legal_ttt = bool(int(os.environ.get("RUN_LEGAL_TTT", "1")))

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

    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = base.load_validation_tokens(args.val_files, val_seq_len, args.val_tokens_limit)
    (base_bytes_lut, has_leading_space_lut, is_boundary_token_lut), meta = base.load_tokenizer_luts(
        args.tokenizer_path,
        args.tokenizer_meta_path,
        args.vocab_size,
        device,
    )

    model = build_model(args, device, mtp_num_heads=args.mtp_num_heads, mtp_loss_weight=args.mtp_loss_weight)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    full_state_dict = model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    alias_groups = find_alias_groups(export_sd)
    maybe_drop_keys(export_sd, drop_keys)
    dropped_prefix_keys = maybe_drop_prefixes(export_sd, drop_prefixes)

    if master_process:
        log0(f"storage_pass checkpoint={checkpoint_path}")
        log0(f"storage_pass quant_cats={sorted(quant_cats)}")
        log0(f"storage_pass tokenizer_kind={meta.get('tokenizer_kind')}")
        log0(f"storage_pass drop_keys={sorted(drop_keys)}")
        log0(f"storage_pass drop_prefixes={list(drop_prefixes)}")
        log0(f"storage_pass zero_bigram={zero_bigram}")
        log0(f"storage_pass lzma_preset={lzma_preset}")
        log0(f"storage_pass alias_groups={len(alias_groups)}")
        for names in alias_groups[:20]:
            log0(f"storage_pass alias_group: {names}")
        if dropped_prefix_keys:
            log0(f"storage_pass dropped_prefix_keys={dropped_prefix_keys}")

    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = base._unbank_state_dict(sd_cpu, args.num_layers)
    quant_result, quant_meta = base.mixed_quantize_int6(unbanked_sd, quant_cats)

    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=lzma_preset)
    code_bytes = BASE_PATH.stat().st_size
    if master_process:
        log0(f"storage_pass quant_bytes={len(quant_blob)}")
        log0(f"storage_pass code_bytes={code_bytes}")
        log0(f"storage_pass estimated_total={len(quant_blob) + code_bytes}")

    if distributed:
        dist.barrier()

    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")
    deq_unbanked = base.dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = base._rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    restored_bigram: list[str] = []
    if zero_bigram:
        restored_bigram = maybe_restore_zero_bigram_from_model(deq_state, model)
        zeroed = maybe_zero_bigram_tensors(deq_state)
        if restored_bigram:
            log0(f"storage_pass restored_bigram_tensors={restored_bigram}")
        log0(f"storage_pass zeroed_bigram_tensors={zeroed}")

    eval_model = build_model(args, device, mtp_num_heads=0, mtp_loss_weight=0.0)
    eval_model.load_state_dict(deq_state, strict=True)

    grad_accum_steps = 8 // world_size
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = base.eval_val(
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
        sw_val_loss, sw_val_bpb = base.eval_val_sliding(
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
            f"storage_pass_int8_sliding_window_exact val_loss:{sw_val_loss:.8f} "
            f"val_bpb:{sw_val_bpb:.8f} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )

    if run_legal_ttt:
        ttt_model = model if args.ttt_use_prequant else eval_model
        log0(f"storage_pass_legal_ttt_mode:{'prequant' if args.ttt_use_prequant else 'postquant'}")
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = base.eval_val_sliding_ttt(
            args,
            ttt_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            log0=log0,
        )
        torch.cuda.synchronize()
        log0(
            f"storage_pass_legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
