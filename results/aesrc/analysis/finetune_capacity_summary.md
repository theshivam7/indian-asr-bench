# Fine-tuning capacity summary — Tiny / Small / Medium (AESRC2020 Indian)

One official-split fine-tune per model size, each compared against its own
HF-pipeline pretrained baseline.

Holm-Bonferroni family = exactly these **3 official-split FT-vs-HF
tests** (one per size) — kept separate from the headline cross-model pairwise family in
`statistics_pairwise_transcript_clean.csv` (that family covers PRETRAINED models only;
the fine-tuned variants run through a different decoding engine, so mixing them in would
confound fine-tuning with an engine change — see `analysis/statistics.py`).

| Size | Params | Pretrained (openai) | HF baseline | Fine-tuned | Δ (paired, speaker-clustered) | 95% CI | p | p (Holm) | n clips | n speakers |
|------|:------:|:--------------------:|:-----------:|:----------:|:-----------------------------:|:------:|:-:|:--------:|:-------:|:----------:|
| Whisper Tiny | 39M | 13.66% | 17.45% | 12.64% | -4.81 pp | [-12.30, +1.71] | 0.163 | 0.163 | 1731 | 481 |
| Whisper Small | 244M | 7.23% | 7.22% | 5.64% | -1.58 pp | [-2.01, -1.15] | 0.001 | 0.003 | 1731 | 481 |
| Whisper Medium | 769M | 5.73% | 5.63% | 4.48% | -1.15 pp | [-1.55, -0.77] | 0.001 | 0.003 | 1731 | 481 |

## Pretrained capacity curve (for context; not a fine-tuning statistic)

Speaker-clustered bootstrap CIs from `analysis/statistics.py:analyze()` (N=1731 clips, G=481 speakers, B=2000). Point estimates only — no Holm correction applied or needed here (these are per-model CIs, not pairwise tests).

| Model | Params | Corpus WER | 95% CI |
|-------|:------:|:----------:|:------:|
| Whisper Tiny | 39M | 13.66% | [12.95, 14.40] |
| Whisper Base | 74M | 9.96% | [9.36, 10.59] |
| Whisper Small | 244M | 7.23% | [6.72, 7.76] |
| Whisper Medium | 769M | 5.73% | [5.25, 6.20] |
| Whisper Large-v3 | 1.5B | 5.20% | [4.75, 5.68] |
