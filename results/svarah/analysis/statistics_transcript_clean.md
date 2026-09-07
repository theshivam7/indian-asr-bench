# Statistical significance: Svarah, mode `transcript_clean`

Corpus WER with 95% bootstrap CI: 10000 resamples, seed 42, N=6656 clips, resampled by **recording** (3232 clusters). Headline (chart) models only, the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. No speaker id is exposed for this dataset; resampling clusters on the recording tag embedded in the clip filename (chunks of one recording share accent/channel/session). This is not a full speaker id, one speaker can contribute several recordings, so CIs may still understate within-speaker correlation, but strictly less than clip-level resampling would. Clip-level CIs are in the CSV for comparison.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Large-v3 | 7.11 | 6.63 | 7.61 | 0.49 |
| Whisper Medium | 7.89 | 7.43 | 8.4 | 0.48 |
| Whisper large-v3-turbo | 8.1 | 7.59 | 8.66 | 0.54 |
| Whisper Small | 10.06 | 9.49 | 10.69 | 0.6 |
| Parakeet-TDT-0.6B-v2 | 11.73 | 11.09 | 12.41 | 0.66 |
| Qwen3-ASR-1.7B | 11.82 | 11.11 | 12.56 | 0.72 |
| Whisper Base | 14.53 | 13.81 | 15.29 | 0.74 |
| Parakeet-CTC-1.1B | 15.65 | 14.85 | 16.47 | 0.81 |
| Whisper Tiny | 19.96 | 18.9 | 21.09 | 1.1 |

## Pairwise paired significance

Difference = WER(A) - WER(B) in pp; paired bootstrap on identical recording-level resamples; two-sided p-values with Holm-Bonferroni correction across all 36 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | base | 5.44 | 4.67 | 6.26 | 0.0002 | 0.0072 | yes |
| tiny | small | 9.9 | 9.07 | 10.76 | 0.0002 | 0.0072 | yes |
| tiny | medium | 12.07 | 11.19 | 13.0 | 0.0002 | 0.0072 | yes |
| tiny | large | 12.86 | 11.97 | 13.79 | 0.0002 | 0.0072 | yes |
| tiny | large_v3_turbo | 11.86 | 10.99 | 12.8 | 0.0002 | 0.0072 | yes |
| tiny | parakeet | 8.24 | 7.3 | 9.24 | 0.0002 | 0.0072 | yes |
| tiny | parakeet_ctc | 4.32 | 3.35 | 5.31 | 0.0002 | 0.0072 | yes |
| tiny | qwen3 | 8.15 | 7.21 | 9.14 | 0.0002 | 0.0072 | yes |
| base | small | 4.46 | 3.97 | 4.94 | 0.0002 | 0.0072 | yes |
| base | medium | 6.64 | 6.13 | 7.16 | 0.0002 | 0.0072 | yes |
| base | large | 7.42 | 6.9 | 7.96 | 0.0002 | 0.0072 | yes |
| base | large_v3_turbo | 6.43 | 5.91 | 6.95 | 0.0002 | 0.0072 | yes |
| base | parakeet | 2.8 | 2.17 | 3.44 | 0.0002 | 0.0072 | yes |
| base | parakeet_ctc | -1.12 | -1.82 | -0.43 | 0.0016 | 0.0072 | yes |
| base | qwen3 | 2.71 | 2.08 | 3.34 | 0.0002 | 0.0072 | yes |
| small | medium | 2.17 | 1.78 | 2.58 | 0.0002 | 0.0072 | yes |
| small | large | 2.96 | 2.56 | 3.39 | 0.0002 | 0.0072 | yes |
| small | large_v3_turbo | 1.96 | 1.56 | 2.38 | 0.0002 | 0.0072 | yes |
| small | parakeet | -1.66 | -2.23 | -1.09 | 0.0002 | 0.0072 | yes |
| small | parakeet_ctc | -5.58 | -6.24 | -4.93 | 0.0002 | 0.0072 | yes |
| small | qwen3 | -1.75 | -2.32 | -1.18 | 0.0002 | 0.0072 | yes |
| medium | large | 0.79 | 0.47 | 1.09 | 0.0002 | 0.0072 | yes |
| medium | large_v3_turbo | -0.21 | -0.54 | 0.1 | 0.1906 | 0.3812 | no |
| medium | parakeet | -3.83 | -4.37 | -3.32 | 0.0002 | 0.0072 | yes |
| medium | parakeet_ctc | -7.75 | -8.42 | -7.13 | 0.0002 | 0.0072 | yes |
| medium | qwen3 | -3.92 | -4.48 | -3.38 | 0.0002 | 0.0072 | yes |
| large | large_v3_turbo | -1.0 | -1.34 | -0.67 | 0.0002 | 0.0072 | yes |
| large | parakeet | -4.62 | -5.14 | -4.11 | 0.0002 | 0.0072 | yes |
| large | parakeet_ctc | -8.54 | -9.21 | -7.9 | 0.0002 | 0.0072 | yes |
| large | qwen3 | -4.71 | -5.26 | -4.18 | 0.0002 | 0.0072 | yes |
| large_v3_turbo | parakeet | -3.62 | -4.14 | -3.12 | 0.0002 | 0.0072 | yes |
| large_v3_turbo | parakeet_ctc | -7.55 | -8.17 | -6.95 | 0.0002 | 0.0072 | yes |
| large_v3_turbo | qwen3 | -3.71 | -4.25 | -3.18 | 0.0002 | 0.0072 | yes |
| parakeet | parakeet_ctc | -3.92 | -4.54 | -3.34 | 0.0002 | 0.0072 | yes |
| parakeet | qwen3 | -0.09 | -0.65 | 0.46 | 0.7567 | 0.7567 | no |
| parakeet_ctc | qwen3 | 3.83 | 3.44 | 4.24 | 0.0002 | 0.0072 | yes |
