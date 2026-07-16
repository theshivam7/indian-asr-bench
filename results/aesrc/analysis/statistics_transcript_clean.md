# Statistical significance — AESRC2020 (Indian) — mode `transcript_clean`

Corpus WER with 95% bootstrap CI: 2000 resamples, seed 42, N=1731 clips, resampled by **speaker** (481 clusters). Headline (chart) models only — the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. Speaker-level resampling accounts for within-speaker correlation (clips from one speaker share accent/channel); clip-level CIs are in the CSV for comparison and are narrower, i.e. anti-conservative.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Large-v3 | 5.2 | 4.75 | 5.68 | 0.47 |
| Qwen3-ASR-1.7B | 5.23 | 4.75 | 5.75 | 0.5 |
| Whisper Medium | 5.73 | 5.25 | 6.2 | 0.47 |
| Whisper large-v3-turbo | 5.81 | 5.37 | 6.26 | 0.45 |
| Parakeet-TDT-0.6B-v2 | 6.26 | 5.83 | 6.74 | 0.45 |
| Whisper Small | 7.23 | 6.72 | 7.76 | 0.52 |
| Parakeet-CTC-1.1B | 7.5 | 7.01 | 8.0 | 0.5 |
| Whisper Base | 9.96 | 9.36 | 10.59 | 0.61 |
| Whisper Tiny | 13.66 | 12.95 | 14.4 | 0.72 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical speaker-level resamples; two-sided p-values with Holm–Bonferroni correction across all 36 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | base | 3.7 | 3.15 | 4.27 | 0.001 | 0.036 | yes |
| tiny | small | 6.44 | 5.82 | 7.07 | 0.001 | 0.036 | yes |
| tiny | medium | 7.93 | 7.31 | 8.58 | 0.001 | 0.036 | yes |
| tiny | large | 8.47 | 7.81 | 9.14 | 0.001 | 0.036 | yes |
| tiny | large_v3_turbo | 7.86 | 7.25 | 8.48 | 0.001 | 0.036 | yes |
| tiny | parakeet | 7.41 | 6.78 | 8.03 | 0.001 | 0.036 | yes |
| tiny | parakeet_ctc | 6.16 | 5.54 | 6.79 | 0.001 | 0.036 | yes |
| tiny | qwen3 | 8.44 | 7.77 | 9.09 | 0.001 | 0.036 | yes |
| base | small | 2.73 | 2.29 | 3.2 | 0.001 | 0.036 | yes |
| base | medium | 4.23 | 3.75 | 4.73 | 0.001 | 0.036 | yes |
| base | large | 4.76 | 4.27 | 5.26 | 0.001 | 0.036 | yes |
| base | large_v3_turbo | 4.16 | 3.71 | 4.64 | 0.001 | 0.036 | yes |
| base | parakeet | 3.7 | 3.21 | 4.21 | 0.001 | 0.036 | yes |
| base | parakeet_ctc | 2.46 | 1.96 | 2.97 | 0.001 | 0.036 | yes |
| base | qwen3 | 4.73 | 4.21 | 5.27 | 0.001 | 0.036 | yes |
| small | medium | 1.5 | 1.11 | 1.91 | 0.001 | 0.036 | yes |
| small | large | 2.03 | 1.64 | 2.44 | 0.001 | 0.036 | yes |
| small | large_v3_turbo | 1.42 | 1.06 | 1.78 | 0.001 | 0.036 | yes |
| small | parakeet | 0.97 | 0.53 | 1.4 | 0.001 | 0.036 | yes |
| small | parakeet_ctc | -0.28 | -0.71 | 0.16 | 0.2159 | 0.6477 | no |
| small | qwen3 | 2.0 | 1.55 | 2.46 | 0.001 | 0.036 | yes |
| medium | large | 0.53 | 0.23 | 0.85 | 0.001 | 0.036 | yes |
| medium | large_v3_turbo | -0.07 | -0.42 | 0.29 | 0.6587 | 1.0 | no |
| medium | parakeet | -0.53 | -0.93 | -0.14 | 0.009 | 0.054 | no |
| medium | parakeet_ctc | -1.77 | -2.22 | -1.34 | 0.001 | 0.036 | yes |
| medium | qwen3 | 0.5 | 0.06 | 0.91 | 0.031 | 0.124 | no |
| large | large_v3_turbo | -0.61 | -0.93 | -0.26 | 0.004 | 0.036 | yes |
| large | parakeet | -1.06 | -1.47 | -0.68 | 0.001 | 0.036 | yes |
| large | parakeet_ctc | -2.3 | -2.74 | -1.88 | 0.001 | 0.036 | yes |
| large | qwen3 | -0.03 | -0.42 | 0.38 | 0.9085 | 1.0 | no |
| large_v3_turbo | parakeet | -0.45 | -0.85 | -0.1 | 0.018 | 0.09 | no |
| large_v3_turbo | parakeet_ctc | -1.7 | -2.08 | -1.34 | 0.001 | 0.036 | yes |
| large_v3_turbo | qwen3 | 0.58 | 0.22 | 0.93 | 0.004 | 0.036 | yes |
| parakeet | parakeet_ctc | -1.24 | -1.58 | -0.9 | 0.001 | 0.036 | yes |
| parakeet | qwen3 | 1.03 | 0.65 | 1.41 | 0.001 | 0.036 | yes |
| parakeet_ctc | qwen3 | 2.28 | 1.89 | 2.66 | 0.001 | 0.036 | yes |
