# Statistical significance: Svarah, mode `whisper_norm`

Corpus WER with 95% bootstrap CI: 10000 resamples, seed 42, N=6656 clips, resampled by **recording** (3232 clusters). Headline (chart) models only, the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. No speaker id is exposed for this dataset; resampling clusters on the recording tag embedded in the clip filename (chunks of one recording share accent/channel/session). This is not a full speaker id, one speaker can contribute several recordings, so CIs may still understate within-speaker correlation, but strictly less than clip-level resampling would. Clip-level CIs are in the CSV for comparison.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Large-v3 | 6.8 | 6.33 | 7.31 | 0.49 |
| Whisper Medium | 7.69 | 7.24 | 8.18 | 0.47 |
| Whisper large-v3-turbo | 7.76 | 7.27 | 8.31 | 0.52 |
| Qwen3-ASR-1.7B | 8.32 | 7.75 | 8.94 | 0.6 |
| Parakeet-TDT-0.6B-v2 | 8.35 | 7.87 | 8.86 | 0.49 |
| Whisper Small | 9.91 | 9.31 | 10.54 | 0.62 |
| Parakeet-CTC-1.1B | 11.18 | 10.6 | 11.82 | 0.61 |
| Whisper Base | 14.37 | 13.64 | 15.15 | 0.76 |
| Whisper Tiny | 19.52 | 18.46 | 20.64 | 1.09 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical recording-level resamples; two-sided p-values with Holm–Bonferroni correction across all 36 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | base | 5.15 | 4.43 | 5.94 | 0.0002 | 0.0072 | yes |
| tiny | small | 9.61 | 8.82 | 10.45 | 0.0002 | 0.0072 | yes |
| tiny | medium | 11.83 | 10.97 | 12.75 | 0.0002 | 0.0072 | yes |
| tiny | large | 12.72 | 11.85 | 13.65 | 0.0002 | 0.0072 | yes |
| tiny | large_v3_turbo | 11.76 | 10.91 | 12.66 | 0.0002 | 0.0072 | yes |
| tiny | parakeet | 11.17 | 10.33 | 12.09 | 0.0002 | 0.0072 | yes |
| tiny | parakeet_ctc | 8.34 | 7.54 | 9.21 | 0.0002 | 0.0072 | yes |
| tiny | qwen3 | 11.2 | 10.35 | 12.13 | 0.0002 | 0.0072 | yes |
| base | small | 4.47 | 3.98 | 4.93 | 0.0002 | 0.0072 | yes |
| base | medium | 6.68 | 6.2 | 7.19 | 0.0002 | 0.0072 | yes |
| base | large | 7.57 | 7.06 | 8.09 | 0.0002 | 0.0072 | yes |
| base | large_v3_turbo | 6.61 | 6.13 | 7.1 | 0.0002 | 0.0072 | yes |
| base | parakeet | 6.03 | 5.54 | 6.53 | 0.0002 | 0.0072 | yes |
| base | parakeet_ctc | 3.19 | 2.72 | 3.67 | 0.0002 | 0.0072 | yes |
| base | qwen3 | 6.05 | 5.54 | 6.6 | 0.0002 | 0.0072 | yes |
| small | medium | 2.22 | 1.86 | 2.61 | 0.0002 | 0.0072 | yes |
| small | large | 3.1 | 2.71 | 3.52 | 0.0002 | 0.0072 | yes |
| small | large_v3_turbo | 2.14 | 1.77 | 2.54 | 0.0002 | 0.0072 | yes |
| small | parakeet | 1.56 | 1.18 | 1.98 | 0.0002 | 0.0072 | yes |
| small | parakeet_ctc | -1.27 | -1.68 | -0.85 | 0.0002 | 0.0072 | yes |
| small | qwen3 | 1.59 | 1.17 | 2.05 | 0.0002 | 0.0072 | yes |
| medium | large | 0.88 | 0.6 | 1.15 | 0.0002 | 0.0072 | yes |
| medium | large_v3_turbo | -0.08 | -0.36 | 0.19 | 0.5969 | 1.0 | no |
| medium | parakeet | -0.66 | -0.93 | -0.38 | 0.0002 | 0.0072 | yes |
| medium | parakeet_ctc | -3.49 | -3.85 | -3.15 | 0.0002 | 0.0072 | yes |
| medium | qwen3 | -0.63 | -0.99 | -0.28 | 0.0004 | 0.0072 | yes |
| large | large_v3_turbo | -0.96 | -1.25 | -0.69 | 0.0002 | 0.0072 | yes |
| large | parakeet | -1.54 | -1.83 | -1.25 | 0.0002 | 0.0072 | yes |
| large | parakeet_ctc | -4.38 | -4.76 | -4.0 | 0.0002 | 0.0072 | yes |
| large | qwen3 | -1.51 | -1.87 | -1.16 | 0.0002 | 0.0072 | yes |
| large_v3_turbo | parakeet | -0.58 | -0.85 | -0.29 | 0.0004 | 0.0072 | yes |
| large_v3_turbo | parakeet_ctc | -3.42 | -3.75 | -3.08 | 0.0002 | 0.0072 | yes |
| large_v3_turbo | qwen3 | -0.55 | -0.89 | -0.2 | 0.0022 | 0.0072 | yes |
| parakeet | parakeet_ctc | -2.83 | -3.13 | -2.55 | 0.0002 | 0.0072 | yes |
| parakeet | qwen3 | 0.03 | -0.28 | 0.32 | 0.8463 | 1.0 | no |
| parakeet_ctc | qwen3 | 2.86 | 2.52 | 3.22 | 0.0002 | 0.0072 | yes |
