# Statistical significance — AESRC2020 (Indian) — mode `whisper_norm`

Corpus WER with 95% bootstrap CI: 2000 resamples, seed 42, N=1731 clips, resampled by **speaker** (481 clusters). Headline (chart) models only — the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. Speaker-level resampling accounts for within-speaker correlation (clips from one speaker share accent/channel); clip-level CIs are in the CSV for comparison and are narrower, i.e. anti-conservative.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Large-v3 | 4.78 | 4.36 | 5.2 | 0.42 |
| Qwen3-ASR-1.7B | 4.89 | 4.45 | 5.35 | 0.45 |
| Whisper Medium | 5.41 | 4.95 | 5.87 | 0.46 |
| Whisper large-v3-turbo | 5.56 | 5.1 | 6.0 | 0.45 |
| Parakeet-TDT-0.6B-v2 | 5.93 | 5.5 | 6.38 | 0.44 |
| Whisper Small | 6.96 | 6.43 | 7.53 | 0.55 |
| Parakeet-CTC-1.1B | 7.13 | 6.66 | 7.62 | 0.48 |
| Whisper Base | 9.64 | 9.06 | 10.25 | 0.6 |
| Whisper Tiny | 13.21 | 12.5 | 13.95 | 0.73 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical speaker-level resamples; two-sided p-values with Holm–Bonferroni correction across all 36 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | base | 3.57 | 3.04 | 4.11 | 0.001 | 0.036 | yes |
| tiny | small | 6.25 | 5.65 | 6.86 | 0.001 | 0.036 | yes |
| tiny | medium | 7.79 | 7.19 | 8.46 | 0.001 | 0.036 | yes |
| tiny | large | 8.43 | 7.78 | 9.1 | 0.001 | 0.036 | yes |
| tiny | large_v3_turbo | 7.65 | 7.05 | 8.26 | 0.001 | 0.036 | yes |
| tiny | parakeet | 7.28 | 6.67 | 7.93 | 0.001 | 0.036 | yes |
| tiny | parakeet_ctc | 6.07 | 5.46 | 6.73 | 0.001 | 0.036 | yes |
| tiny | qwen3 | 8.32 | 7.68 | 8.97 | 0.001 | 0.036 | yes |
| base | small | 2.68 | 2.24 | 3.13 | 0.001 | 0.036 | yes |
| base | medium | 4.23 | 3.78 | 4.69 | 0.001 | 0.036 | yes |
| base | large | 4.86 | 4.4 | 5.34 | 0.001 | 0.036 | yes |
| base | large_v3_turbo | 4.09 | 3.64 | 4.56 | 0.001 | 0.036 | yes |
| base | parakeet | 3.71 | 3.24 | 4.21 | 0.001 | 0.036 | yes |
| base | parakeet_ctc | 2.51 | 2.04 | 3.01 | 0.001 | 0.036 | yes |
| base | qwen3 | 4.75 | 4.25 | 5.27 | 0.001 | 0.036 | yes |
| small | medium | 1.54 | 1.17 | 1.94 | 0.001 | 0.036 | yes |
| small | large | 2.18 | 1.79 | 2.6 | 0.001 | 0.036 | yes |
| small | large_v3_turbo | 1.4 | 1.03 | 1.76 | 0.001 | 0.036 | yes |
| small | parakeet | 1.03 | 0.58 | 1.51 | 0.001 | 0.036 | yes |
| small | parakeet_ctc | -0.18 | -0.62 | 0.27 | 0.4398 | 1.0 | no |
| small | qwen3 | 2.07 | 1.65 | 2.51 | 0.001 | 0.036 | yes |
| medium | large | 0.63 | 0.32 | 0.94 | 0.001 | 0.036 | yes |
| medium | large_v3_turbo | -0.14 | -0.47 | 0.17 | 0.3808 | 1.0 | no |
| medium | parakeet | -0.51 | -0.91 | -0.15 | 0.009 | 0.054 | no |
| medium | parakeet_ctc | -1.72 | -2.14 | -1.31 | 0.001 | 0.036 | yes |
| medium | qwen3 | 0.52 | 0.12 | 0.9 | 0.019 | 0.095 | no |
| large | large_v3_turbo | -0.77 | -1.12 | -0.45 | 0.001 | 0.036 | yes |
| large | parakeet | -1.15 | -1.52 | -0.8 | 0.001 | 0.036 | yes |
| large | parakeet_ctc | -2.35 | -2.75 | -1.97 | 0.001 | 0.036 | yes |
| large | qwen3 | -0.11 | -0.47 | 0.25 | 0.5737 | 1.0 | no |
| large_v3_turbo | parakeet | -0.37 | -0.75 | 0.02 | 0.067 | 0.268 | no |
| large_v3_turbo | parakeet_ctc | -1.58 | -1.97 | -1.2 | 0.001 | 0.036 | yes |
| large_v3_turbo | qwen3 | 0.67 | 0.31 | 1.02 | 0.002 | 0.036 | yes |
| parakeet | parakeet_ctc | -1.21 | -1.53 | -0.86 | 0.001 | 0.036 | yes |
| parakeet | qwen3 | 1.04 | 0.67 | 1.4 | 0.001 | 0.036 | yes |
| parakeet_ctc | qwen3 | 2.24 | 1.9 | 2.62 | 0.001 | 0.036 | yes |
