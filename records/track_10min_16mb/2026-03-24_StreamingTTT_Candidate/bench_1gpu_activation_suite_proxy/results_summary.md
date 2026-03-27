# 1-GPU Activation Suite Summary

| Case | Runs | Activation | TTT | Param mode | Roundtrip mean | Sliding mean | Legal TTT mean | Legal TTT std | Best legal TTT |
|------|------|------------|-----|------------|----------------|--------------|----------------|---------------|----------------|
| codex_asymmetric_square | 3/3 | asymmetric_square | stream | late_blocks:last_4 | 1.798000 | 1.792348 | 1.787940 | 0.037835 | 1.735814 |
| codex_gated_square | 3/3 | gated_square | stream | late_blocks:last_4 | 1.821240 | 1.814613 | 1.807545 | 0.027838 | 1.771185 |
| codex_leaky_relu_sq | 3/3 | leaky_relu_sq | stream | late_blocks:last_4 | 1.702933 | 1.694098 | 1.690500 | 0.011677 | 1.681367 |
| codex_sign_preserving_square | 3/3 | sign_preserving_square | stream | late_blocks:last_4 | 1.851017 | 1.844734 | 1.835533 | 0.010685 | 1.824514 |
| leader_asymmetric_square | 3/3 | asymmetric_square |  |  | 1.798368 | 1.792538 | 1.730605 | 0.007025 | 1.721115 |
| leader_gated_square | 3/3 | gated_square |  |  | 1.813720 | 1.806840 | 1.728151 | 0.020933 | 1.698766 |
| leader_leaky_relu_sq | 3/3 | leaky_relu_sq |  |  | 1.799023 | 1.793627 | 1.723517 | 0.006732 | 1.714750 |
| leader_sign_preserving_square | 3/3 | sign_preserving_square |  |  | 1.775770 | 1.765027 | 1.691461 | 0.047086 | 1.624997 |
