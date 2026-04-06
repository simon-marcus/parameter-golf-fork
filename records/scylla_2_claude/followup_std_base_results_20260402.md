# Follow-Up Standard-Base Results

Date: 2026-04-02

## Scope

Completed follow-up runs on the standard-tokenizer base `sp_xsa_all`:

- `std_adapter` on pod `xq0m5tbqvn786r`
- `std_causal` on pod `fwsk8itvz65n4l`

## Log Pull Status

- `xq0m5tbqvn786r` logs were pulled locally to:
  `records/scylla_2_claude/pulled_followup_runs/xq0m5tbqvn786r/std_adapter/`
- `fwsk8itvz65n4l` was stopped before the `rsync` completed. The local pull is incomplete.
  The final `std_causal` metrics below were recorded from live remote log inspection
  immediately before shutdown.

## Baseline

- Base config:
  `records/scylla_2_claude/followup_bases/sp_xsa_all.json`
- Stage 1 base score:
  `final_int6_sliding_window_exact val_bpb = 1.35526573`

## Adapter TTT Results

Source logs:

- `records/scylla_2_claude/pulled_followup_runs/xq0m5tbqvn786r/std_adapter/sp1024_sp_xsa_all/top4_r32/seed_1337/train.log`
- `records/scylla_2_claude/pulled_followup_runs/xq0m5tbqvn786r/std_adapter/sp1024_sp_xsa_all/top4_r64/seed_1337/train.log`
- `records/scylla_2_claude/pulled_followup_runs/xq0m5tbqvn786r/std_adapter/sp1024_sp_xsa_all/all_r32/seed_1337/train.log`
- `records/scylla_2_claude/pulled_followup_runs/xq0m5tbqvn786r/std_adapter/sp1024_sp_xsa_all/all_r64/seed_1337/train.log`

Results:

- `top4_r32`: `legal_ttt_exact val_bpb = 1.35669510`
- `top4_r64`: `legal_ttt_exact val_bpb = 1.35571459`
- `all_r32`: `legal_ttt_exact val_bpb = 1.35714084`
- `all_r64`: `legal_ttt_exact val_bpb = 1.35572511`

Best adapter result:

- `top4_r64`: `1.35571459`

Takeaway:

- Adapterized TTT did not beat the `sp_xsa_all` base.
- Best delta vs base: `+0.00044886` BPB worse.

## Legal Causal Cache Results

Recorded from remote logs on `fwsk8itvz65n4l` before shutdown.

Results:

- `o24_light`: `legal_causal_cache_exact val_bpb = 1.38459763`
- `o24_strong`: `legal_causal_cache_exact val_bpb = 1.40857779`
- `o34_light`: `legal_causal_cache_exact val_bpb = 1.37919631`
- `o34_conservative`: `legal_causal_cache_exact val_bpb = 1.36923587`

Best causal-cache result:

- `o34_conservative`: `1.36923587`

Takeaway:

- The exact causal cache variants were all worse than the `sp_xsa_all` base.
- Best delta vs base: `+0.01397014` BPB worse.

## Pod Actions

Both completed pods were stopped after result capture:

- `fwsk8itvz65n4l`
- `xq0m5tbqvn786r`
