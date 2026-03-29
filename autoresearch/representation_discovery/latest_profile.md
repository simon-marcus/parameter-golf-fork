# FineWeb Profile: sp1024

## Overview
- tokenizer: `/Users/simon/Code/parameter-golf-local/data/tokenizers/fineweb_1024_bpe.model`
- train tokens sampled: `1000000`
- val tokens sampled: `1000000`
- vocab size: `1024`

## Split Metrics
- `train` bytes/token: `2.4364`
- `train` unique tokens used: `848`
- `train` top class by byte mass: `alpha`
- `val` bytes/token: `2.4070`
- `val` unique tokens used: `861`
- `val` top class by byte mass: `alpha`

## Repetition Profile (val)
- context `1`: repeat_frac=`0.996`, avg_max_next_prob=`0.159`, avg_entropy=`4.959`
- context `2`: repeat_frac=`0.785`, avg_max_next_prob=`0.507`, avg_entropy=`1.895`
- context `4`: repeat_frac=`0.188`, avg_max_next_prob=`0.807`, avg_entropy=`0.565`
- context `8`: repeat_frac=`0.030`, avg_max_next_prob=`0.931`, avg_entropy=`0.211`
- context `16`: repeat_frac=`0.005`, avg_max_next_prob=`0.967`, avg_entropy=`0.095`
- context `32`: repeat_frac=`0.001`, avg_max_next_prob=`0.997`, avg_entropy=`0.007`

## Leading Classes By Byte Mass
- `train`:
  - `alpha` byte_frac=`0.945` token_frac=`0.888` avg_bytes/token=`2.59`
  - `punct` byte_frac=`0.035` token_frac=`0.066` avg_bytes/token=`1.31`
  - `numeric` byte_frac=`0.010` token_frac=`0.025` avg_bytes/token=`1.00`
  - `whitespace` byte_frac=`0.005` token_frac=`0.011` avg_bytes/token=`1.00`
  - `alnum` byte_frac=`0.003` token_frac=`0.006` avg_bytes/token=`1.24`
  - `byte` byte_frac=`0.001` token_frac=`0.003` avg_bytes/token=`1.00`
- `val`:
  - `alpha` byte_frac=`0.942` token_frac=`0.882` avg_bytes/token=`2.57`
  - `punct` byte_frac=`0.037` token_frac=`0.068` avg_bytes/token=`1.30`
  - `numeric` byte_frac=`0.011` token_frac=`0.027` avg_bytes/token=`1.00`
  - `whitespace` byte_frac=`0.005` token_frac=`0.013` avg_bytes/token=`1.00`
  - `alnum` byte_frac=`0.004` token_frac=`0.007` avg_bytes/token=`1.30`
  - `byte` byte_frac=`0.001` token_frac=`0.003` avg_bytes/token=`1.00`
