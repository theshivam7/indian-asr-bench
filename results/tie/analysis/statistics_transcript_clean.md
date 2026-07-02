# Statistical significance — TIE_shorts — mode `transcript_clean`

Corpus WER with 95% bootstrap CI: 2000 resamples, seed 42, N=985 clips, resampled by **speaker** (280 clusters). Speaker-level resampling accounts for within-speaker correlation (clips from one speaker share accent/channel); clip-level CIs are in the CSV for comparison and are narrower, i.e. anti-conservative.

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Medium (HF) | 14.42 | 12.85 | 16.19 | 1.67 |
| Whisper Medium (FT) | 14.61 | 12.69 | 16.98 | 2.14 |
| Whisper Medium | 14.76 | 13.69 | 15.87 | 1.09 |
| Parakeet-TDT-0.6B | 15.6 | 14.49 | 16.71 | 1.11 |
| Whisper Large | 15.93 | 14.72 | 17.16 | 1.22 |
| Qwen3-ASR-1.7B | 16.66 | 15.57 | 17.79 | 1.11 |
| Whisper Base | 17.53 | 16.3 | 18.8 | 1.25 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; paired bootstrap on identical speaker-level resamples; two-sided p-values with Holm–Bonferroni correction across all 21 pairs.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | medium | 2.77 | 2.1 | 3.41 | 0.001 | 0.021 | yes |
| base | large | 1.59 | 0.71 | 2.46 | 0.001 | 0.021 | yes |
| base | parakeet | 1.93 | 1.34 | 2.51 | 0.001 | 0.021 | yes |
| base | qwen3 | 0.86 | 0.29 | 1.45 | 0.004 | 0.056 | no |
| base | medium_hf | 3.11 | 1.36 | 4.39 | 0.002 | 0.032 | yes |
| base | medium_ft | 2.91 | 0.47 | 4.63 | 0.025 | 0.275 | no |
| medium | large | -1.17 | -1.94 | -0.46 | 0.002 | 0.032 | yes |
| medium | parakeet | -0.84 | -1.37 | -0.31 | 0.004 | 0.056 | no |
| medium | qwen3 | -1.9 | -2.5 | -1.3 | 0.001 | 0.021 | yes |
| medium | medium_hf | 0.34 | -1.18 | 1.44 | 0.5817 | 1.0 | no |
| medium | medium_ft | 0.15 | -2.14 | 1.72 | 0.7956 | 1.0 | no |
| large | parakeet | 0.34 | -0.44 | 1.21 | 0.4018 | 1.0 | no |
| large | qwen3 | -0.73 | -1.55 | 0.14 | 0.1089 | 0.8712 | no |
| large | medium_hf | 1.51 | 0.01 | 2.77 | 0.051 | 0.51 | no |
| large | medium_ft | 1.32 | -0.85 | 2.98 | 0.2199 | 1.0 | no |
| parakeet | qwen3 | -1.07 | -1.66 | -0.53 | 0.001 | 0.021 | yes |
| parakeet | medium_hf | 1.18 | -0.53 | 2.45 | 0.1689 | 1.0 | no |
| parakeet | medium_ft | 0.98 | -1.49 | 2.69 | 0.3618 | 1.0 | no |
| qwen3 | medium_hf | 2.24 | 0.53 | 3.49 | 0.014 | 0.168 | no |
| qwen3 | medium_ft | 2.05 | -0.38 | 3.73 | 0.082 | 0.738 | no |
| medium_hf | medium_ft | -0.2 | -1.03 | 0.46 | 0.6417 | 1.0 | no |
