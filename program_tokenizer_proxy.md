# Parameter Golf Proxy-Calibrated Tokenizer Program

## Objective
Discover tokenizer candidates that are likely to improve proxy-trained postquant `val_bpb`, not just cheap tokenizer efficiency.

The screening objective must reflect two empirical lessons:
- larger vocabularies looked good on cheap text metrics but often lost in proxy training
- simpler/smaller tokenizers are more promising than larger ones, especially for TokenMonster

## Calibration Rules
- Trust proxy outcomes over tokenizer-only metrics
- Prefer lower estimated proxy `postquant_val_bpb`
- Heavily penalize vocab growth unless the calibrated score clearly improves
- Treat TokenMonster as promising mainly in the smaller-vocab regime

## Allowed Experiment Types
1. `sentencepiece_train`
   - train a SentencePiece tokenizer on the reconstructed FineWeb sample
2. `tokenmonster_eval`
   - evaluate a known TokenMonster vocabulary reference without retraining it

## Search Priorities
1. Establish a proxy-calibrated baseline from `sp1024`
2. Explore small/simple TokenMonster vocabularies first
   - especially `1024` and `2048`
3. Explore SentencePiece mainly near or below the baseline scale
   - for example `768`, `896`, `1024`, `1152`, `1280`, `1536`, `2048`
4. Prefer one primary change per experiment
5. Do not spend time on larger vocabularies unless the calibrated score still strongly favors them

## Anti-Goals
- Do not assume lower `tokens_per_byte` implies better proxy BPB
- Do not prioritize tokenizers that only win by spending more vocab budget
- Do not mix tokenizer experiments with model-training changes in this lane
