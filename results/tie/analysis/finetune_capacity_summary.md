# Fine-tuning capacity summary: Tiny / Small / Medium (official split)

One official-split fine-tune per model size, each compared against its own
HF-pipeline pretrained baseline.

Holm-Bonferroni family = exactly these **3 official-split FT-vs-HF
tests** (one per size), kept separate from the headline cross-model pairwise family in
`statistics_pairwise_transcript_clean.csv` (that family covers PRETRAINED models only;
the fine-tuned variants run through a different decoding engine, so mixing them in would
confound fine-tuning with an engine change, see `analysis/statistics.py`).

| Size | Params | Pretrained (openai) | HF baseline | Fine-tuned | Δ (paired, speaker-clustered) | 95% CI | p | p (Holm) | n clips | n speakers |
|------|:------:|:--------------------:|:-----------:|:----------:|:-----------------------------:|:------:|:-:|:--------:|:-------:|:----------:|
| Whisper Tiny | 39M | 19.43% | 22.10% | 19.14% | -2.96 pp | [-6.35, +0.13] | 0.065 | 0.195 | 985 | 280 |
| Whisper Small | 244M | 16.05% | 17.38% | 16.21% | -1.17 pp | [-3.97, +1.21] | 0.387 | 0.774 | 985 | 280 |
| Whisper Medium | 769M | 14.76% | 14.42% | 14.61% | +0.20 pp | [-0.46, +1.03] | 0.642 | 0.774 | 985 | 280 |

## Pretrained capacity curve (for context; not a fine-tuning statistic)

Speaker-clustered bootstrap CIs from `analysis/statistics.py:analyze()` (N=985 clips, G=280 speakers, B=2000). Point estimates only , no Holm correction applied or needed here (these are per-model CIs, not pairwise tests).

| Model | Params | Corpus WER | 95% CI |
|-------|:------:|:----------:|:------:|
| Whisper Tiny | 39M | 19.43% | [18.12, 20.79] |
| Whisper Base | 74M | 17.53% | [16.30, 18.80] |
| Whisper Small | 244M | 16.05% | [14.85, 17.34] |
| Whisper Medium | 769M | 14.76% | [13.69, 15.87] |
| Whisper Large-v3 | 1.5B | 15.93% | [14.72, 17.16] |
