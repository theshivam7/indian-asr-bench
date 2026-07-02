# Statistical significance — TIE_shorts — mode `transcript_clean`

Corpus WER with 95% bootstrap CI: 2000 resamples, seed 42, N=985 clips, resampled by **speaker** (280 clusters). Headline (chart) models only — the fine-tuning study is a separate hypothesis family with its own paired test in `finetune_comparison.md`. Speaker-level resampling accounts for within-speaker correlation (clips from one speaker share accent/channel); clip-level CIs are in the CSV for comparison and are narrower, i.e. anti-conservative.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Medium | 14.76 | 13.69 | 15.87 | 1.09 |
| Parakeet-TDT-0.6B | 15.6 | 14.49 | 16.71 | 1.11 |
| Whisper Large | 15.93 | 14.72 | 17.16 | 1.22 |
| Qwen3-ASR-1.7B | 16.66 | 15.57 | 17.79 | 1.11 |
| Whisper Base | 17.53 | 16.3 | 18.8 | 1.25 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical speaker-level resamples; two-sided p-values with Holm–Bonferroni correction across all 10 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | medium | 2.77 | 2.1 | 3.41 | 0.001 | 0.01 | yes |
| base | large | 1.59 | 0.71 | 2.46 | 0.001 | 0.01 | yes |
| base | parakeet | 1.93 | 1.34 | 2.51 | 0.001 | 0.01 | yes |
| base | qwen3 | 0.86 | 0.29 | 1.45 | 0.004 | 0.016 | yes |
| medium | large | -1.17 | -1.94 | -0.46 | 0.002 | 0.01 | yes |
| medium | parakeet | -0.84 | -1.37 | -0.31 | 0.004 | 0.016 | yes |
| medium | qwen3 | -1.9 | -2.5 | -1.3 | 0.001 | 0.01 | yes |
| large | parakeet | 0.34 | -0.44 | 1.21 | 0.4018 | 0.4018 | no |
| large | qwen3 | -0.73 | -1.55 | 0.14 | 0.1089 | 0.2178 | no |
| parakeet | qwen3 | -1.07 | -1.66 | -0.53 | 0.001 | 0.01 | yes |
