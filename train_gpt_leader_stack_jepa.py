"""
Leader-stack JEPA candidate built on the 2026-03-23 leaky-ReLU / legal-TTT family.

This wrapper imports the March 23 script, keeps its training stack intact, and
adds a JEPA auxiliary loss on top of the leader hidden states. The current
version uses a small learned predictor head, registered under the block stack so
the leader optimizer path actually trains it. TTT is disabled by default for the
first A/B so JEPA can be judged against the pre-TTT sliding-window metric.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor


BASE_PATH = Path(__file__).resolve().parent / "records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py"
SPEC = importlib.util.spec_from_file_location("leaderstack_20260323", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import leader stack module from {BASE_PATH}")
leaderstack = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = leaderstack
SPEC.loader.exec_module(leaderstack)


class Hyperparameters(leaderstack.Hyperparameters):
    patch_size = int(os.environ.get("PATCH_SIZE", 8))
    jepa_loss_weight = float(os.environ.get("JEPA_LOSS_WEIGHT", 0.0))
    jepa_var_weight = float(os.environ.get("JEPA_VAR_WEIGHT", 0.02))
    jepa_std_target = float(os.environ.get("JEPA_STD_TARGET", 0.5))
    # Keep TTT off by default so leader-stack JEPA is judged on the pre-TTT metric first.
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))


class GPTLeaderStackJEPA(leaderstack.GPT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = int(os.environ.get("PATCH_SIZE", 8))
        self.jepa_loss_weight = float(os.environ.get("JEPA_LOSS_WEIGHT", 0.0))
        self.jepa_var_weight = float(os.environ.get("JEPA_VAR_WEIGHT", 0.02))
        self.jepa_std_target = float(os.environ.get("JEPA_STD_TARGET", 0.5))
        # Register the learned predictor under blocks[0] so the leader-stack optimizer
        # picks it up through block_named_params without rewriting leaderstack.main().
        self.blocks[0].jepa_in = leaderstack.CastedLinear(self.tok_emb.embedding_dim, self.tok_emb.embedding_dim, bias=False)
        self.blocks[0].jepa_out = leaderstack.CastedLinear(self.tok_emb.embedding_dim, self.tok_emb.embedding_dim, bias=False)
        self.blocks[0].jepa_out._zero_init = True
        self.jepa_in = self.blocks[0].jepa_in
        self.jepa_out = self.blocks[0].jepa_out

    def encode_hidden(self, input_ids: Tensor) -> Tensor:
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](
                x,
                x0,
                self.qo_bank[i],
                self.kv_bank[i],
                self.kv_bank[n + i],
                self.qo_bank[n + i],
                self.mlp_up_bank[i],
                self.mlp_down_bank[i],
                v_embed=ve,
                v0=v0,
            )
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x, _ = self.blocks[bi](
                x,
                x0,
                self.qo_bank[bi],
                self.kv_bank[bi],
                self.kv_bank[n + bi],
                self.qo_bank[n + bi],
                self.mlp_up_bank[bi],
                self.mlp_down_bank[bi],
                v_embed=ve,
                v0=v0,
            )
        return self.final_norm(x)

    def predict_patch_latents(self, patch_states: Tensor) -> Tensor:
        x = F.rms_norm(patch_states, (patch_states.size(-1),))
        x = torch.relu(self.jepa_in(x))
        return self.jepa_out(x.square())

    def compute_jepa_loss(self, input_ids: Tensor, hidden: Tensor) -> Tensor:
        if hidden.size(1) % self.patch_size != 0:
            raise ValueError(
                f"sequence length {hidden.size(1)} must be divisible by PATCH_SIZE={self.patch_size}"
            )
        patch_states = hidden[:, self.patch_size - 1 :: self.patch_size, :]
        if patch_states.size(1) < 2:
            return hidden.new_zeros(())
        pred = self.predict_patch_latents(patch_states[:, :-1, :]).float()
        target_patches = input_ids.reshape(input_ids.size(0), -1, self.patch_size)[:, 1:, :]
        with torch.no_grad():
            target = self.tok_emb(target_patches).mean(dim=2).float()
            target = F.rms_norm(target, (target.size(-1),))
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        jepa_loss = F.mse_loss(pred_n, target_n, reduction="mean")
        pred_std = pred.std(dim=(0, 1), correction=0)
        var_loss = torch.relu(self.jepa_std_target - pred_std).mean()
        return jepa_loss + self.jepa_var_weight * var_loss

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.encode_hidden(input_ids)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_targets = target_ids[:, k + 1 :].reshape(-1)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        if self.training and self.jepa_loss_weight > 0.0:
            main_loss = main_loss + self.jepa_loss_weight * self.compute_jepa_loss(input_ids, x)
        return main_loss


leaderstack.Hyperparameters = Hyperparameters
leaderstack.GPT = GPTLeaderStackJEPA


if __name__ == "__main__":
    leaderstack.main()
