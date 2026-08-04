# Statistical significance: Svarah, mode `whisper_norm`

Corpus WER with 95% bootstrap CI: 2000 resamples, seed 42, N=6656 clips, resampled by **recording** (3232 clusters). Headline (chart) models only, the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. No speaker id is exposed for this dataset; resampling clusters on the recording tag embedded in the clip filename (chunks of one recording share accent/channel/session). This is not a full speaker id, one speaker can contribute several recordings, so CIs may still understate within-speaker correlation, but strictly less than clip-level resampling would. Clip-level CIs are in the CSV for comparison.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Large-v3 | 6.8 | 6.33 | 7.28 | 0.48 |
| Whisper Medium | 7.69 | 7.25 | 8.17 | 0.46 |
| Whisper large-v3-turbo | 7.76 | 7.27 | 8.3 | 0.51 |
| Qwen3-ASR-1.7B | 8.32 | 7.73 | 8.93 | 0.6 |
| Parakeet-TDT-0.6B-v2 | 8.35 | 7.87 | 8.84 | 0.49 |
| Whisper Small | 9.91 | 9.32 | 10.54 | 0.61 |
| Parakeet-CTC-1.1B | 11.18 | 10.61 | 11.78 | 0.59 |
| Whisper Base | 14.37 | 13.63 | 15.15 | 0.76 |
| Whisper Tiny | 19.52 | 18.46 | 20.59 | 1.06 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical recording-level resamples; two-sided p-values with Holm–Bonferroni correction across all 36 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | base | 5.15 | 4.43 | 5.91 | 0.001 | 0.036 | yes |
| tiny | small | 9.61 | 8.79 | 10.41 | 0.001 | 0.036 | yes |
| tiny | medium | 11.83 | 10.97 | 12.66 | 0.001 | 0.036 | yes |
| tiny | large | 12.72 | 11.87 | 13.63 | 0.001 | 0.036 | yes |
| tiny | large_v3_turbo | 11.76 | 10.89 | 12.6 | 0.001 | 0.036 | yes |
| tiny | parakeet | 11.17 | 10.34 | 12.04 | 0.001 | 0.036 | yes |
| tiny | parakeet_ctc | 8.34 | 7.51 | 9.16 | 0.001 | 0.036 | yes |
| tiny | qwen3 | 11.2 | 10.36 | 12.08 | 0.001 | 0.036 | yes |
| base | small | 4.47 | 3.98 | 4.93 | 0.001 | 0.036 | yes |
| base | medium | 6.68 | 6.21 | 7.19 | 0.001 | 0.036 | yes |
| base | large | 7.57 | 7.07 | 8.09 | 0.001 | 0.036 | yes |
| base | large_v3_turbo | 6.61 | 6.13 | 7.11 | 0.001 | 0.036 | yes |
| base | parakeet | 6.03 | 5.53 | 6.52 | 0.001 | 0.036 | yes |
| base | parakeet_ctc | 3.19 | 2.71 | 3.66 | 0.001 | 0.036 | yes |
| base | qwen3 | 6.05 | 5.53 | 6.61 | 0.001 | 0.036 | yes |
| small | medium | 2.22 | 1.87 | 2.61 | 0.001 | 0.036 | yes |
| small | large | 3.1 | 2.73 | 3.53 | 0.001 | 0.036 | yes |
| small | large_v3_turbo | 2.14 | 1.78 | 2.54 | 0.001 | 0.036 | yes |
| small | parakeet | 1.56 | 1.19 | 1.98 | 0.001 | 0.036 | yes |
| small | parakeet_ctc | -1.27 | -1.69 | -0.84 | 0.001 | 0.036 | yes |
| small | qwen3 | 1.59 | 1.16 | 2.07 | 0.001 | 0.036 | yes |
| medium | large | 0.88 | 0.6 | 1.15 | 0.001 | 0.036 | yes |
| medium | large_v3_turbo | -0.08 | -0.35 | 0.18 | 0.5977 | 1.0 | no |
| medium | parakeet | -0.66 | -0.94 | -0.39 | 0.001 | 0.036 | yes |
| medium | parakeet_ctc | -3.49 | -3.87 | -3.15 | 0.001 | 0.036 | yes |
| medium | qwen3 | -0.63 | -1.01 | -0.28 | 0.001 | 0.036 | yes |
| large | large_v3_turbo | -0.96 | -1.25 | -0.7 | 0.001 | 0.036 | yes |
| large | parakeet | -1.54 | -1.83 | -1.25 | 0.001 | 0.036 | yes |
| large | parakeet_ctc | -4.38 | -4.76 | -4.01 | 0.001 | 0.036 | yes |
| large | qwen3 | -1.51 | -1.88 | -1.15 | 0.001 | 0.036 | yes |
| large_v3_turbo | parakeet | -0.58 | -0.85 | -0.29 | 0.001 | 0.036 | yes |
| large_v3_turbo | parakeet_ctc | -3.42 | -3.76 | -3.09 | 0.001 | 0.036 | yes |
| large_v3_turbo | qwen3 | -0.55 | -0.89 | -0.19 | 0.004 | 0.036 | yes |
| parakeet | parakeet_ctc | -2.83 | -3.13 | -2.56 | 0.001 | 0.036 | yes |
| parakeet | qwen3 | 0.03 | -0.28 | 0.33 | 0.8166 | 1.0 | no |
| parakeet_ctc | qwen3 | 2.86 | 2.52 | 3.22 | 0.001 | 0.036 | yes |
