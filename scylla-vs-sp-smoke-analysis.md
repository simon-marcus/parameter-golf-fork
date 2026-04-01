## Scylla-v2 vs SP1024 `1xH100` Smoke Analysis

Context:
- Both runs are `1xH100` smoke tests on a reduced `2`-train-shard subset plus full val.
- Both use the same legal score-first TTT stack and seed `1337`.
- Scylla-v2 run log was pulled from the pod.
- Baseline SP1024 run was partially saved locally in [temp-logs.log](/Users/simon/Code/parameter-golf/temp-logs.log).

### Key Results

From [scylla-vs-sp-observations.csv](/Users/simon/Code/parameter-golf/scylla-vs-sp-observations.csv):

- Scylla-v2 `final_int8_zlib_roundtrip_exact`: `3.86079038`
- SP1024 `final_int8_zlib_roundtrip_exact`: `4.68463856`
- Scylla-v2 `legal_ttt_exact`: `2.57423349`
- SP1024 `legal_ttt_exact`: `2.53334434`

### Interpretation

The current smoke evidence suggests different strengths at different stages:

- Scylla-v2 is materially better before TTT.
  - It beats SP1024 clearly on roundtrip/postquant evaluation.
- SP1024 is better after long score-first TTT adaptation.
  - Final legal TTT is better by about `0.04089` BPB (`2.57423 - 2.53334`).

So the current picture is not "Scylla is dead" and not "Scylla is immediately better."
It looks more like:

- Scylla-v2 improves the base model / tokenizer representation.
- SP1024 adapts more effectively under the current TTT regime.

### Saved Baseline TTT Progress

From [temp-logs.log](/Users/simon/Code/parameter-golf/temp-logs.log), baseline SP1024 TTT had reached:

- `ttt_chunk [1511/1893] bpb=2.561776 time=1499.0s`

Then there is a missing interval, and later:

- `ttt_chunk [1841/1893] bpb=2.536590 time=1827.8s`

This is consistent with the final baseline `legal_ttt_exact` beating Scylla-v2 after the later TTT chunks.

### Working Hypothesis

Likely explanation:

- Scylla-v2 gives a stronger starting representation.
- But its current tokenization does not work as well with the existing TTT hyperparameters or adaptation dynamics.

That means the next high-value work is probably not "throw Scylla away."
It is:

1. recover the full baseline and Scylla smoke logs from the pod volume
2. compare TTT trajectories directly
3. check whether Scylla-v2 needs different TTT settings:
   - `TTT_LR`
   - `TTT_EPOCHS`
   - `TTT_CHUNK_TOKENS`
   - freeze depth / adaptation scope

### Immediate Conclusion

Scylla-v2 remains promising.

- It wins pre-TTT quality.
- It currently loses late-stage TTT quality.

So the next decision should be driven by TTT-compatibility analysis, not by raw roundtrip BPB alone.
