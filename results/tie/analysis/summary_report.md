# WER Evaluation Summary — TIE_shorts

## Corpus WER (%) by model and mode (+ primary-mode CER)

| model | display | transcript_raw | transcript_clean | hf_raw | hf_clean | whisper_norm | CER_primary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | Whisper Base | 17.91 | 17.53 | 20.24 | 18.07 | 17.03 | 11.35 |
| medium | Whisper Medium | 15.11 | 14.76 | 18.01 | 15.76 | 14.48 | 10.14 |
| large | Whisper Large | 16.31 | 15.93 | 19.14 | 16.94 | 15.76 | 11.2 |
| large_v3_turbo | Whisper large-v3-turbo | N/A | N/A | N/A | N/A | N/A | N/A |
| parakeet | Parakeet-TDT-0.6B | 15.97 | 15.6 | 18.54 | 16.4 | 15.17 | 10.21 |
| parakeet_ctc | Parakeet-CTC-1.1B | N/A | N/A | N/A | N/A | N/A | N/A |
| qwen3 | Qwen3-ASR-1.7B | 18.15 | 16.66 | 17.99 | 17.61 | 15.4 | 10.45 |
| medium_hf | Whisper Medium (HF) | 14.75 | 14.42 | 17.72 | 15.51 | 14.23 | 10.06 |
| medium_ft | Whisper Medium (FT) | 14.71 | 14.61 | 17.7 | 15.7 | 14.31 | 10.25 |
| medium_ft_disjoint | Whisper Medium (FT, speaker-disjoint) | 16.53 | 16.17 | 19.39 | 17.1 | 15.95 | 10.92 |
| medium_ft_disjoint_s43 | Whisper Medium (FT, disjoint, seed 43) | 15.14 | 14.8 | 18.01 | 15.73 | 14.53 | 10.43 |
| medium_ft_disjoint_s44 | Whisper Medium (FT, disjoint, seed 44) | 15.58 | 15.2 | 18.52 | 16.16 | 15.01 | 10.68 |
| medium_ft_sizematch_s42 | Whisper Medium (FT, size-matched ctrl, seed 42) | 14.7 | 14.33 | 17.6 | 15.28 | 14.16 | 10.01 |
| medium_ft_sizematch_s43 | Whisper Medium (FT, size-matched ctrl, seed 43) | 14.82 | 14.4 | 17.57 | 15.36 | 14.44 | 9.61 |
| medium_ft_sizematch_s44 | Whisper Medium (FT, size-matched ctrl, seed 44) | 14.8 | 14.4 | 17.71 | 15.49 | 14.31 | 9.69 |

## Best model per mode

- **transcript_raw**: Whisper Medium (FT, size-matched ctrl, seed 42) (14.70%)
- **transcript_clean**: Whisper Medium (FT, size-matched ctrl, seed 42) (14.33%)
- **hf_raw**: Whisper Medium (FT, size-matched ctrl, seed 43) (17.57%)
- **hf_clean**: Whisper Medium (FT, size-matched ctrl, seed 42) (15.28%)
- **whisper_norm**: Whisper Medium (FT, size-matched ctrl, seed 42) (14.16%)