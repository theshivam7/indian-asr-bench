# Statistical significance — Svarah — mode `transcript_clean`

Corpus WER with 95% bootstrap CI: 2000 resamples, seed 42, N=6656 clips, resampled by **recording** (3232 clusters). Headline (chart) models only — the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. No speaker id is exposed for this dataset; resampling clusters on the recording tag embedded in the clip filename (chunks of one recording share accent/channel/session). This is not a full speaker id — one speaker can contribute several recordings — so CIs may still understate within-speaker correlation, but strictly less than clip-level resampling would. Clip-level CIs are in the CSV for comparison.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Large-v3 | 7.11 | 6.63 | 7.59 | 0.48 |
| Whisper Medium | 7.89 | 7.45 | 8.39 | 0.47 |
| Whisper large-v3-turbo | 8.1 | 7.59 | 8.67 | 0.54 |
| Parakeet-TDT-0.6B-v2 | 11.73 | 11.1 | 12.39 | 0.65 |
| Qwen3-ASR-1.7B | 11.82 | 11.11 | 12.53 | 0.71 |
| Whisper Base | 14.53 | 13.81 | 15.29 | 0.74 |
| Parakeet-CTC-1.1B | 15.65 | 14.86 | 16.46 | 0.8 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical recording-level resamples; two-sided p-values with Holm–Bonferroni correction across all 21 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | medium | 6.64 | 6.15 | 7.17 | 0.001 | 0.021 | yes |
| base | large | 7.42 | 6.92 | 7.97 | 0.001 | 0.021 | yes |
| base | large_v3_turbo | 6.43 | 5.91 | 6.97 | 0.001 | 0.021 | yes |
| base | parakeet | 2.8 | 2.15 | 3.43 | 0.001 | 0.021 | yes |
| base | parakeet_ctc | -1.12 | -1.83 | -0.43 | 0.002 | 0.021 | yes |
| base | qwen3 | 2.71 | 2.08 | 3.35 | 0.001 | 0.021 | yes |
| medium | large | 0.79 | 0.48 | 1.09 | 0.001 | 0.021 | yes |
| medium | large_v3_turbo | -0.21 | -0.55 | 0.08 | 0.1749 | 0.3498 | no |
| medium | parakeet | -3.83 | -4.37 | -3.33 | 0.001 | 0.021 | yes |
| medium | parakeet_ctc | -7.75 | -8.42 | -7.14 | 0.001 | 0.021 | yes |
| medium | qwen3 | -3.92 | -4.48 | -3.36 | 0.001 | 0.021 | yes |
| large | large_v3_turbo | -1.0 | -1.34 | -0.68 | 0.001 | 0.021 | yes |
| large | parakeet | -4.62 | -5.16 | -4.1 | 0.001 | 0.021 | yes |
| large | parakeet_ctc | -8.54 | -9.22 | -7.92 | 0.001 | 0.021 | yes |
| large | qwen3 | -4.71 | -5.26 | -4.18 | 0.001 | 0.021 | yes |
| large_v3_turbo | parakeet | -3.62 | -4.15 | -3.13 | 0.001 | 0.021 | yes |
| large_v3_turbo | parakeet_ctc | -7.55 | -8.16 | -6.94 | 0.001 | 0.021 | yes |
| large_v3_turbo | qwen3 | -3.71 | -4.26 | -3.15 | 0.001 | 0.021 | yes |
| parakeet | parakeet_ctc | -3.92 | -4.54 | -3.35 | 0.001 | 0.021 | yes |
| parakeet | qwen3 | -0.09 | -0.65 | 0.48 | 0.7656 | 0.7656 | no |
| parakeet_ctc | qwen3 | 3.83 | 3.44 | 4.24 | 0.001 | 0.021 | yes |
