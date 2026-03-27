# 1-GPU Activation Suite Summary

| Case | Runs | Activation | TTT | Param mode | Roundtrip mean | Sliding mean | Legal TTT mean | Legal TTT std | Best legal TTT |
|------|------|------------|-----|------------|----------------|--------------|----------------|---------------|----------------|
| chunk_2_11 | 3/3 | leaky_relu_sq | chunk | block_range:2-11 | 1.730977 | 1.722965 | 1.708155 | 0.020767 | 1.687663 |
| chunk_3_8 | 3/3 | leaky_relu_sq | chunk | block_range:3-8 | 1.733351 | 1.724901 | 1.718178 | 0.037028 | 1.679703 |
| chunk_4_11 | 3/3 | leaky_relu_sq | chunk | block_range:4-11 | 1.708873 | 1.699495 | 1.693956 | 0.078360 | 1.583371 |
| chunk_4_9 | 3/3 | leaky_relu_sq | chunk | block_range:4-9 | 1.738093 | 1.729894 | 1.726978 | 0.081640 | 1.617316 |
| chunk_5_10 | 3/3 | leaky_relu_sq | chunk | block_range:5-10 | 1.809213 | 1.803492 | 1.801152 | 0.034911 | 1.759012 |
| chunk_6_11 | 3/3 | leaky_relu_sq | chunk | block_range:6-11 | 1.823267 | 1.817241 | 1.813991 | 0.038292 | 1.784728 |
| chunk_full | 3/3 | leaky_relu_sq | chunk | full | 1.762543 | 1.755418 | 1.692211 | 0.028220 | 1.662856 |
| stream_3_8 | 3/3 | leaky_relu_sq | stream | block_range:3-8 | 1.771017 | 1.764194 | 1.740048 | 0.030455 | 1.718330 |
| stream_4_9 | 3/3 | leaky_relu_sq | stream | block_range:4-9 | 1.837560 | 1.832385 | 1.817805 | 0.029840 | 1.796578 |
| stream_5_10 | 3/3 | leaky_relu_sq | stream | block_range:5-10 | 1.840221 | 1.835635 | 1.827319 | 0.020068 | 1.800630 |
| stream_6_11 | 3/3 | leaky_relu_sq | stream | block_range:6-11 | 1.851683 | 1.846920 | 1.838093 | 0.027663 | 1.810028 |
