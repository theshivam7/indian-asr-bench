# Statistical significance — TIE_shorts — mode `whisper_norm`

Corpus WER with 95% bootstrap CI: 2000 resamples, seed 42, N=986 clips, resampled by **speaker** (280 clusters). Headline (chart) models only — the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. Speaker-level resampling accounts for within-speaker correlation (clips from one speaker share accent/channel); clip-level CIs are in the CSV for comparison and are narrower, i.e. anti-conservative.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Medium | 14.48 | 13.41 | 15.54 | 1.06 |
| Parakeet-TDT-0.6B-v2 | 15.17 | 14.04 | 16.29 | 1.12 |
| Qwen3-ASR-1.7B | 15.4 | 14.25 | 16.51 | 1.13 |
| Whisper Large-v3 | 15.76 | 14.52 | 16.98 | 1.23 |
| Whisper Small | 15.8 | 14.63 | 17.07 | 1.22 |
| Parakeet-CTC-1.1B | 16.19 | 15.04 | 17.36 | 1.16 |
| Whisper Base | 17.03 | 15.84 | 18.25 | 1.2 |
| Whisper large-v3-turbo | 17.75 | 16.28 | 19.23 | 1.48 |
| Whisper Tiny | 19.01 | 17.7 | 20.34 | 1.32 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical speaker-level resamples; two-sided p-values with Holm–Bonferroni correction across all 36 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | base | 1.98 | 1.47 | 2.51 | 0.001 | 0.036 | yes |
| tiny | small | 3.21 | 2.57 | 3.85 | 0.001 | 0.036 | yes |
| tiny | medium | 4.53 | 3.86 | 5.21 | 0.001 | 0.036 | yes |
| tiny | large | 3.25 | 2.3 | 4.16 | 0.001 | 0.036 | yes |
| tiny | large_v3_turbo | 1.26 | 0.09 | 2.36 | 0.036 | 0.324 | no |
| tiny | parakeet | 3.84 | 3.24 | 4.46 | 0.001 | 0.036 | yes |
| tiny | parakeet_ctc | 2.82 | 2.24 | 3.41 | 0.001 | 0.036 | yes |
| tiny | qwen3 | 3.61 | 2.99 | 4.26 | 0.001 | 0.036 | yes |
| base | small | 1.23 | 0.59 | 1.85 | 0.001 | 0.036 | yes |
| base | medium | 2.56 | 1.95 | 3.15 | 0.001 | 0.036 | yes |
| base | large | 1.28 | 0.41 | 2.12 | 0.007 | 0.077 | no |
| base | large_v3_turbo | -0.72 | -1.9 | 0.36 | 0.2019 | 1.0 | no |
| base | parakeet | 1.86 | 1.3 | 2.41 | 0.001 | 0.036 | yes |
| base | parakeet_ctc | 0.84 | 0.32 | 1.38 | 0.002 | 0.036 | yes |
| base | qwen3 | 1.63 | 1.06 | 2.19 | 0.001 | 0.036 | yes |
| small | medium | 1.32 | 0.83 | 1.9 | 0.001 | 0.036 | yes |
| small | large | 0.04 | -0.71 | 0.78 | 0.8516 | 1.0 | no |
| small | large_v3_turbo | -1.95 | -3.02 | -0.9 | 0.001 | 0.036 | yes |
| small | parakeet | 0.63 | 0.09 | 1.19 | 0.025 | 0.25 | no |
| small | parakeet_ctc | -0.39 | -0.93 | 0.19 | 0.1849 | 1.0 | no |
| small | qwen3 | 0.4 | -0.17 | 1.01 | 0.1609 | 1.0 | no |
| medium | large | -1.28 | -2.04 | -0.56 | 0.001 | 0.036 | yes |
| medium | large_v3_turbo | -3.27 | -4.37 | -2.22 | 0.001 | 0.036 | yes |
| medium | parakeet | -0.69 | -1.18 | -0.2 | 0.006 | 0.072 | no |
| medium | parakeet_ctc | -1.71 | -2.21 | -1.18 | 0.001 | 0.036 | yes |
| medium | qwen3 | -0.92 | -1.43 | -0.43 | 0.001 | 0.036 | yes |
| large | large_v3_turbo | -2.0 | -3.17 | -0.89 | 0.001 | 0.036 | yes |
| large | parakeet | 0.59 | -0.16 | 1.42 | 0.1269 | 1.0 | no |
| large | parakeet_ctc | -0.43 | -1.18 | 0.43 | 0.2979 | 1.0 | no |
| large | qwen3 | 0.35 | -0.38 | 1.19 | 0.3798 | 1.0 | no |
| large_v3_turbo | parakeet | 2.58 | 1.55 | 3.69 | 0.001 | 0.036 | yes |
| large_v3_turbo | parakeet_ctc | 1.56 | 0.57 | 2.6 | 0.003 | 0.039 | yes |
| large_v3_turbo | qwen3 | 2.35 | 1.34 | 3.42 | 0.001 | 0.036 | yes |
| parakeet | parakeet_ctc | -1.02 | -1.4 | -0.64 | 0.001 | 0.036 | yes |
| parakeet | qwen3 | -0.23 | -0.68 | 0.21 | 0.3058 | 1.0 | no |
| parakeet_ctc | qwen3 | 0.79 | 0.45 | 1.14 | 0.001 | 0.036 | yes |
