# Statistical significance: TIE_shorts, mode `transcript_clean`

Corpus WER with 95% bootstrap CI: 2000 resamples, seed 42, N=985 clips, resampled by **speaker** (280 clusters). Headline (chart) models only, the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. Speaker-level resampling accounts for within-speaker correlation (clips from one speaker share accent/channel); clip-level CIs are in the CSV for comparison and are narrower, i.e. anti-conservative.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Medium | 14.76 | 13.69 | 15.87 | 1.09 |
| Parakeet-TDT-0.6B-v2 | 15.6 | 14.49 | 16.71 | 1.11 |
| Whisper Large-v3 | 15.93 | 14.72 | 17.16 | 1.22 |
| Whisper Small | 16.05 | 14.85 | 17.34 | 1.24 |
| Parakeet-CTC-1.1B | 16.45 | 15.33 | 17.59 | 1.13 |
| Qwen3-ASR-1.7B | 16.66 | 15.57 | 17.79 | 1.11 |
| Whisper Base | 17.53 | 16.3 | 18.8 | 1.25 |
| Whisper large-v3-turbo | 17.98 | 16.49 | 19.47 | 1.49 |
| Whisper Tiny | 19.43 | 18.12 | 20.79 | 1.33 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical speaker-level resamples; two-sided p-values with Holm–Bonferroni correction across all 36 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | base | 1.91 | 1.37 | 2.45 | 0.001 | 0.036 | yes |
| tiny | small | 3.38 | 2.65 | 4.07 | 0.001 | 0.036 | yes |
| tiny | medium | 4.67 | 3.98 | 5.34 | 0.001 | 0.036 | yes |
| tiny | large | 3.5 | 2.5 | 4.44 | 0.001 | 0.036 | yes |
| tiny | large_v3_turbo | 1.46 | 0.26 | 2.57 | 0.019 | 0.19 | no |
| tiny | parakeet | 3.84 | 3.19 | 4.49 | 0.001 | 0.036 | yes |
| tiny | parakeet_ctc | 2.99 | 2.41 | 3.57 | 0.001 | 0.036 | yes |
| tiny | qwen3 | 2.77 | 2.11 | 3.45 | 0.001 | 0.036 | yes |
| base | small | 1.47 | 0.75 | 2.16 | 0.002 | 0.036 | yes |
| base | medium | 2.77 | 2.1 | 3.41 | 0.001 | 0.036 | yes |
| base | large | 1.59 | 0.71 | 2.46 | 0.001 | 0.036 | yes |
| base | large_v3_turbo | -0.45 | -1.66 | 0.65 | 0.4168 | 1.0 | no |
| base | parakeet | 1.93 | 1.34 | 2.51 | 0.001 | 0.036 | yes |
| base | parakeet_ctc | 1.08 | 0.56 | 1.61 | 0.001 | 0.036 | yes |
| base | qwen3 | 0.86 | 0.29 | 1.45 | 0.004 | 0.052 | no |
| small | medium | 1.29 | 0.75 | 1.89 | 0.001 | 0.036 | yes |
| small | large | 0.12 | -0.64 | 0.88 | 0.7166 | 1.0 | no |
| small | large_v3_turbo | -1.92 | -2.99 | -0.89 | 0.001 | 0.036 | yes |
| small | parakeet | 0.46 | -0.15 | 1.05 | 0.1529 | 1.0 | no |
| small | parakeet_ctc | -0.39 | -1.0 | 0.21 | 0.2039 | 1.0 | no |
| small | qwen3 | -0.61 | -1.28 | 0.08 | 0.098 | 0.882 | no |
| medium | large | -1.17 | -1.94 | -0.46 | 0.002 | 0.036 | yes |
| medium | large_v3_turbo | -3.22 | -4.3 | -2.21 | 0.001 | 0.036 | yes |
| medium | parakeet | -0.84 | -1.37 | -0.31 | 0.004 | 0.052 | no |
| medium | parakeet_ctc | -1.69 | -2.23 | -1.14 | 0.001 | 0.036 | yes |
| medium | qwen3 | -1.9 | -2.5 | -1.3 | 0.001 | 0.036 | yes |
| large | large_v3_turbo | -2.04 | -3.23 | -0.93 | 0.001 | 0.036 | yes |
| large | parakeet | 0.34 | -0.44 | 1.21 | 0.4018 | 1.0 | no |
| large | parakeet_ctc | -0.51 | -1.32 | 0.37 | 0.2389 | 1.0 | no |
| large | qwen3 | -0.73 | -1.55 | 0.14 | 0.1089 | 0.882 | no |
| large_v3_turbo | parakeet | 2.38 | 1.34 | 3.57 | 0.001 | 0.036 | yes |
| large_v3_turbo | parakeet_ctc | 1.53 | 0.5 | 2.67 | 0.003 | 0.042 | yes |
| large_v3_turbo | qwen3 | 1.31 | 0.23 | 2.49 | 0.016 | 0.176 | no |
| parakeet | parakeet_ctc | -0.85 | -1.26 | -0.44 | 0.002 | 0.036 | yes |
| parakeet | qwen3 | -1.07 | -1.66 | -0.53 | 0.001 | 0.036 | yes |
| parakeet_ctc | qwen3 | -0.22 | -0.7 | 0.24 | 0.3608 | 1.0 | no |
