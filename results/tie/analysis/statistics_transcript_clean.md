# Statistical significance — TIE_shorts — mode `transcript_clean`

Corpus WER with 95% bootstrap CI (2000 resamples, seed 42, N=985 clips).

| Model | Corpus WER % | CI low | CI high | ±pp |
| --- | --- | --- | --- | --- |
| Whisper Medium (HF) | 14.42 | 12.94 | 16.1 | 1.58 |
| Whisper Medium (FT) | 14.61 | 12.71 | 17.12 | 2.2 |
| Whisper Medium | 14.76 | 13.93 | 15.6 | 0.83 |
| Parakeet-TDT-0.6B | 15.6 | 14.75 | 16.5 | 0.88 |
| Whisper Large | 15.93 | 14.93 | 17.04 | 1.05 |
| Qwen3-ASR-1.7B | 16.66 | 15.81 | 17.58 | 0.89 |
| Whisper Base | 17.53 | 16.63 | 18.49 | 0.93 |

## Pairwise paired significance

Difference = WER(A) − WER(B) in pp; CI and two-sided bootstrap p-value; paired on identical resampled clips.

| model_a | model_b | diff_pp | ci_lo_pp | ci_hi_pp | p_value | significant_0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| base | medium | 2.77 | 2.16 | 3.39 | 0.0 | yes |
| base | large | 1.59 | 0.73 | 2.4 | 0.0 | yes |
| base | parakeet | 1.93 | 1.41 | 2.5 | 0.0 | yes |
| base | qwen3 | 0.86 | 0.32 | 1.4 | 0.002 | yes |
| base | medium_hf | 3.11 | 1.4 | 4.4 | 0.001 | yes |
| base | medium_ft | 2.91 | 0.48 | 4.67 | 0.026 | yes |
| medium | large | -1.17 | -1.98 | -0.47 | 0.002 | yes |
| medium | parakeet | -0.84 | -1.37 | -0.3 | 0.007 | yes |
| medium | qwen3 | -1.9 | -2.46 | -1.33 | 0.0 | yes |
| medium | medium_hf | 0.34 | -1.24 | 1.47 | 0.558 | no |
| medium | medium_ft | 0.15 | -2.13 | 1.76 | 0.748 | no |
| large | parakeet | 0.34 | -0.39 | 1.17 | 0.412 | no |
| large | qwen3 | -0.73 | -1.52 | 0.14 | 0.096 | no |
| large | medium_hf | 1.51 | -0.04 | 2.83 | 0.055 | no |
| large | medium_ft | 1.32 | -0.9 | 3.1 | 0.225 | no |
| parakeet | qwen3 | -1.07 | -1.59 | -0.56 | 0.0 | yes |
| parakeet | medium_hf | 1.18 | -0.56 | 2.43 | 0.144 | no |
| parakeet | medium_ft | 0.98 | -1.5 | 2.69 | 0.343 | no |
| qwen3 | medium_hf | 2.24 | 0.63 | 3.46 | 0.01 | yes |
| qwen3 | medium_ft | 2.05 | -0.33 | 3.73 | 0.081 | no |
| medium_hf | medium_ft | -0.2 | -1.02 | 0.5 | 0.748 | no |
