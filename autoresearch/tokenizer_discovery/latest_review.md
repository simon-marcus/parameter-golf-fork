# Review 0034

- keep: `False`
- headline: 1M input sentences produces identical tokenizer to 500K baseline — no improvement
- frontier_update: SentencePiece BPE training data volume is saturated at 500K sentences for this vocab size; no need to explore higher input_sentence_size values
- next_direction: Test vocab_size=7168 or 8192 to find where artifact cost penalty overtakes compression benefit

The score is exactly 0.3483178826428708, identical to the frontier (experiment 0022). Every metric matches: same tokens_per_byte (0.2913), same unique_tokens_used (6380), same dead_vocab_frac (0.003125). Doubling the training sentences from 500K to 1M produced byte-identical merge quality, meaning 500K sentences already saturates the BPE training for vocab_size=6400.
