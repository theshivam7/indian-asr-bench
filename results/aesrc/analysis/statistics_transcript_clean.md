# Statistical significance: AESRC2020 (Indian), mode `transcript_clean`

Corpus WER with 95% bootstrap CI: 10000 resamples, seed 42, N=1731 clips, resampled by **speaker** (481 clusters). Headline (chart) models only, the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. Speaker-level resampling accounts for within-speaker correlation (clips from one speaker share accent/channel); clip-level CIs are in the CSV for comparison and are narrower, i.e. anti-conservative.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Large-v3 | 5.2 | 4.75 | 5.68 | 0.46 |
| Qwen3-ASR-1.7B | 5.23 | 4.75 | 5.74 | 0.5 |
| Whisper Medium | 5.73 | 5.25 | 6.22 | 0.49 |
| Whisper large-v3-turbo | 5.81 | 5.36 | 6.26 | 0.45 |
| Parakeet-TDT-0.6B-v2 | 6.26 | 5.81 | 6.71 | 0.45 |
| Whisper Small | 7.23 | 6.7 | 7.76 | 0.53 |
| Parakeet-CTC-1.1B | 7.5 | 7.0 | 8.02 | 0.51 |
| Whisper Base | 9.96 | 9.34 | 10.58 | 0.62 |
| Whisper Tiny | 13.66 | 12.96 | 14.37 | 0.71 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical speaker-level resamples; two-sided p-values with Holm–Bonferroni correction across all 36 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | base | 3.7 | 3.14 | 4.27 | 0.0002 | 0.0072 | yes |
| tiny | small | 6.44 | 5.83 | 7.05 | 0.0002 | 0.0072 | yes |
| tiny | medium | 7.93 | 7.31 | 8.57 | 0.0002 | 0.0072 | yes |
| tiny | large | 8.47 | 7.82 | 9.11 | 0.0002 | 0.0072 | yes |
| tiny | large_v3_turbo | 7.86 | 7.26 | 8.45 | 0.0002 | 0.0072 | yes |
| tiny | parakeet | 7.41 | 6.79 | 8.04 | 0.0002 | 0.0072 | yes |
| tiny | parakeet_ctc | 6.16 | 5.54 | 6.8 | 0.0002 | 0.0072 | yes |
| tiny | qwen3 | 8.44 | 7.79 | 9.07 | 0.0002 | 0.0072 | yes |
| base | small | 2.73 | 2.28 | 3.21 | 0.0002 | 0.0072 | yes |
| base | medium | 4.23 | 3.73 | 4.73 | 0.0002 | 0.0072 | yes |
| base | large | 4.76 | 4.25 | 5.26 | 0.0002 | 0.0072 | yes |
| base | large_v3_turbo | 4.16 | 3.69 | 4.63 | 0.0002 | 0.0072 | yes |
| base | parakeet | 3.7 | 3.21 | 4.2 | 0.0002 | 0.0072 | yes |
| base | parakeet_ctc | 2.46 | 1.95 | 2.98 | 0.0002 | 0.0072 | yes |
| base | qwen3 | 4.73 | 4.21 | 5.25 | 0.0002 | 0.0072 | yes |
| small | medium | 1.5 | 1.09 | 1.89 | 0.0002 | 0.0072 | yes |
| small | large | 2.03 | 1.63 | 2.43 | 0.0002 | 0.0072 | yes |
| small | large_v3_turbo | 1.42 | 1.07 | 1.78 | 0.0002 | 0.0072 | yes |
| small | parakeet | 0.97 | 0.53 | 1.41 | 0.0002 | 0.0072 | yes |
| small | parakeet_ctc | -0.28 | -0.71 | 0.16 | 0.2158 | 0.6474 | no |
| small | qwen3 | 2.0 | 1.55 | 2.44 | 0.0002 | 0.0072 | yes |
| medium | large | 0.53 | 0.22 | 0.86 | 0.0008 | 0.0072 | yes |
| medium | large_v3_turbo | -0.07 | -0.42 | 0.29 | 0.6763 | 1.0 | no |
| medium | parakeet | -0.53 | -0.94 | -0.12 | 0.0124 | 0.0744 | no |
| medium | parakeet_ctc | -1.77 | -2.21 | -1.33 | 0.0002 | 0.0072 | yes |
| medium | qwen3 | 0.5 | 0.07 | 0.93 | 0.0258 | 0.1032 | no |
| large | large_v3_turbo | -0.61 | -0.93 | -0.26 | 0.0012 | 0.0096 | yes |
| large | parakeet | -1.06 | -1.47 | -0.66 | 0.0002 | 0.0072 | yes |
| large | parakeet_ctc | -2.3 | -2.74 | -1.86 | 0.0002 | 0.0072 | yes |
| large | qwen3 | -0.03 | -0.44 | 0.39 | 0.8943 | 1.0 | no |
| large_v3_turbo | parakeet | -0.45 | -0.82 | -0.09 | 0.0192 | 0.096 | no |
| large_v3_turbo | parakeet_ctc | -1.7 | -2.07 | -1.33 | 0.0002 | 0.0072 | yes |
| large_v3_turbo | qwen3 | 0.58 | 0.21 | 0.94 | 0.0038 | 0.0266 | yes |
| parakeet | parakeet_ctc | -1.24 | -1.59 | -0.9 | 0.0002 | 0.0072 | yes |
| parakeet | qwen3 | 1.03 | 0.64 | 1.41 | 0.0002 | 0.0072 | yes |
| parakeet_ctc | qwen3 | 2.28 | 1.89 | 2.66 | 0.0002 | 0.0072 | yes |
