# Statistical significance — Svarah — mode `transcript_clean`

Corpus WER with 95% bootstrap CI: 2000 resamples, seed 42, N=6656 clips, resampled by **clip** (6656 clusters). Headline (chart) models only — the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. No speaker id is exposed for this dataset, so resampling is clip-level; CIs may understate within-speaker correlation (limitation).

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Large | 7.11 | 6.74 | 7.46 | 0.36 |
| Whisper Medium | 7.89 | 7.53 | 8.26 | 0.36 |
| Whisper large-v3-turbo | 8.1 | 7.72 | 8.49 | 0.39 |
| Parakeet-TDT-0.6B | 11.73 | 11.25 | 12.21 | 0.48 |
| Qwen3-ASR-1.7B | 11.82 | 11.28 | 12.4 | 0.56 |
| Whisper Base | 14.53 | 13.99 | 15.04 | 0.52 |
| Parakeet-CTC-1.1B | 15.65 | 15.04 | 16.32 | 0.64 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical clip-level resamples; two-sided p-values with Holm–Bonferroni correction across all 21 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | medium | 6.64 | 6.2 | 7.06 | 0.001 | 0.021 | yes |
| base | large | 7.42 | 6.97 | 7.87 | 0.001 | 0.021 | yes |
| base | large_v3_turbo | 6.43 | 5.97 | 6.88 | 0.001 | 0.021 | yes |
| base | parakeet | 2.8 | 2.3 | 3.3 | 0.001 | 0.021 | yes |
| base | parakeet_ctc | -1.12 | -1.74 | -0.54 | 0.001 | 0.021 | yes |
| base | qwen3 | 2.71 | 2.14 | 3.27 | 0.001 | 0.021 | yes |
| medium | large | 0.79 | 0.48 | 1.09 | 0.001 | 0.021 | yes |
| medium | large_v3_turbo | -0.21 | -0.52 | 0.08 | 0.1789 | 0.3578 | no |
| medium | parakeet | -3.83 | -4.27 | -3.4 | 0.001 | 0.021 | yes |
| medium | parakeet_ctc | -7.75 | -8.35 | -7.2 | 0.001 | 0.021 | yes |
| medium | qwen3 | -3.92 | -4.45 | -3.43 | 0.001 | 0.021 | yes |
| large | large_v3_turbo | -1.0 | -1.33 | -0.68 | 0.001 | 0.021 | yes |
| large | parakeet | -4.62 | -5.05 | -4.19 | 0.001 | 0.021 | yes |
| large | parakeet_ctc | -8.54 | -9.14 | -7.99 | 0.001 | 0.021 | yes |
| large | qwen3 | -4.71 | -5.21 | -4.23 | 0.001 | 0.021 | yes |
| large_v3_turbo | parakeet | -3.62 | -4.06 | -3.18 | 0.001 | 0.021 | yes |
| large_v3_turbo | parakeet_ctc | -7.55 | -8.09 | -7.01 | 0.001 | 0.021 | yes |
| large_v3_turbo | qwen3 | -3.71 | -4.22 | -3.2 | 0.001 | 0.021 | yes |
| parakeet | parakeet_ctc | -3.92 | -4.46 | -3.4 | 0.001 | 0.021 | yes |
| parakeet | qwen3 | -0.09 | -0.61 | 0.41 | 0.7316 | 0.7316 | no |
| parakeet_ctc | qwen3 | 3.83 | 3.49 | 4.2 | 0.001 | 0.021 | yes |
