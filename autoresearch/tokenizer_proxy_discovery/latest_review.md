# Review 0020

- keep: `False`
- headline: SentencePiece 2M-sentence BPE 1024 scores 4.368, well behind frontier 4.139
- frontier_update: Frontier unchanged at english-1024-balanced-v1 (4.139). This confirms that even well-trained SentencePiece 1024 cannot match TokenMonster 1024 variants. The SentencePiece axis is exhausted for vocab_size=1024.
- next_direction: Evaluate english-1024-clean-v1 and english-1024-consistent-v1 to confirm balanced-v1 is the best TokenMonster 1024 variant before integration.

The calibrated score of 4.368 is 0.23 points worse than the frontier (english-1024-balanced-v1 at 4.139). While this SentencePiece variant has near-zero dead vocab (0.4% vs 14.2% baseline), its higher tokens_per_byte and the sentencepiece:1024 family's weaker calibration coefficient make it uncompetitive. More training sentences did not close the gap with TokenMonster.
