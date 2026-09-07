# Statistical significance: TIE_shorts, mode `whisper_norm`

Corpus WER with 95% bootstrap CI: 10000 resamples, seed 42, N=986 clips, resampled by **speaker** (280 clusters). Headline (chart) models only, the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. Speaker-level resampling accounts for within-speaker correlation (clips from one speaker share accent/channel); clip-level CIs are in the CSV for comparison and are narrower, i.e. anti-conservative.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Medium | 14.48 | 13.45 | 15.55 | 1.05 |
| Parakeet-TDT-0.6B-v2 | 15.17 | 14.1 | 16.28 | 1.09 |
| Qwen3-ASR-1.7B | 15.4 | 14.33 | 16.49 | 1.08 |
| Whisper Large-v3 | 15.76 | 14.56 | 17.0 | 1.22 |
| Whisper Small | 15.8 | 14.66 | 17.01 | 1.18 |
| Parakeet-CTC-1.1B | 16.19 | 15.09 | 17.32 | 1.11 |
| Whisper Base | 17.03 | 15.87 | 18.24 | 1.18 |
| Whisper large-v3-turbo | 17.75 | 16.27 | 19.26 | 1.49 |
| Whisper Tiny | 19.01 | 17.77 | 20.27 | 1.25 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical speaker-level resamples; two-sided p-values with Holm–Bonferroni correction across all 36 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | base | 1.98 | 1.44 | 2.52 | 0.0002 | 0.0072 | yes |
| tiny | small | 3.21 | 2.55 | 3.89 | 0.0002 | 0.0072 | yes |
| tiny | medium | 4.53 | 3.88 | 5.2 | 0.0002 | 0.0072 | yes |
| tiny | large | 3.25 | 2.29 | 4.16 | 0.0002 | 0.0072 | yes |
| tiny | large_v3_turbo | 1.26 | 0.06 | 2.4 | 0.0418 | 0.3762 | no |
| tiny | parakeet | 3.84 | 3.25 | 4.45 | 0.0002 | 0.0072 | yes |
| tiny | parakeet_ctc | 2.82 | 2.25 | 3.4 | 0.0002 | 0.0072 | yes |
| tiny | qwen3 | 3.61 | 2.99 | 4.24 | 0.0002 | 0.0072 | yes |
| base | small | 1.23 | 0.6 | 1.89 | 0.0004 | 0.0072 | yes |
| base | medium | 2.56 | 1.94 | 3.18 | 0.0002 | 0.0072 | yes |
| base | large | 1.28 | 0.41 | 2.11 | 0.0046 | 0.0552 | no |
| base | large_v3_turbo | -0.72 | -1.9 | 0.37 | 0.2068 | 1.0 | no |
| base | parakeet | 1.86 | 1.32 | 2.42 | 0.0002 | 0.0072 | yes |
| base | parakeet_ctc | 0.84 | 0.32 | 1.38 | 0.0022 | 0.0286 | yes |
| base | qwen3 | 1.63 | 1.07 | 2.21 | 0.0002 | 0.0072 | yes |
| small | medium | 1.32 | 0.81 | 1.89 | 0.0002 | 0.0072 | yes |
| small | large | 0.04 | -0.72 | 0.76 | 0.8689 | 1.0 | no |
| small | large_v3_turbo | -1.95 | -3.06 | -0.92 | 0.0002 | 0.0072 | yes |
| small | parakeet | 0.63 | 0.09 | 1.2 | 0.022 | 0.22 | no |
| small | parakeet_ctc | -0.39 | -0.93 | 0.19 | 0.1826 | 1.0 | no |
| small | qwen3 | 0.4 | -0.18 | 1.0 | 0.1814 | 1.0 | no |
| medium | large | -1.28 | -2.05 | -0.56 | 0.0006 | 0.0096 | yes |
| medium | large_v3_turbo | -3.27 | -4.39 | -2.25 | 0.0002 | 0.0072 | yes |
| medium | parakeet | -0.69 | -1.18 | -0.21 | 0.005 | 0.0552 | no |
| medium | parakeet_ctc | -1.71 | -2.21 | -1.21 | 0.0002 | 0.0072 | yes |
| medium | qwen3 | -0.92 | -1.42 | -0.43 | 0.0004 | 0.0072 | yes |
| large | large_v3_turbo | -2.0 | -3.21 | -0.85 | 0.0014 | 0.021 | yes |
| large | parakeet | 0.59 | -0.15 | 1.41 | 0.1202 | 0.9616 | no |
| large | parakeet_ctc | -0.43 | -1.2 | 0.41 | 0.2882 | 1.0 | no |
| large | qwen3 | 0.35 | -0.41 | 1.19 | 0.3782 | 1.0 | no |
| large_v3_turbo | parakeet | 2.58 | 1.56 | 3.69 | 0.0002 | 0.0072 | yes |
| large_v3_turbo | parakeet_ctc | 1.56 | 0.57 | 2.62 | 0.0018 | 0.0252 | yes |
| large_v3_turbo | qwen3 | 2.35 | 1.36 | 3.44 | 0.0002 | 0.0072 | yes |
| parakeet | parakeet_ctc | -1.02 | -1.4 | -0.64 | 0.0002 | 0.0072 | yes |
| parakeet | qwen3 | -0.23 | -0.68 | 0.21 | 0.3028 | 1.0 | no |
| parakeet_ctc | qwen3 | 0.79 | 0.46 | 1.12 | 0.0002 | 0.0072 | yes |
