# WER Evaluation Summary — Svarah

## Corpus WER (%) by model and mode (+ primary-mode CER)

| model | display | transcript_raw | transcript_clean | whisper_norm | CER_primary |
| --- | --- | --- | --- | --- | --- |
| tiny | Whisper Tiny | 20.33 | 19.96 | 19.52 | 11.08 |
| base | Whisper Base | 14.88 | 14.53 | 14.37 | 7.78 |
| small | Whisper Small | 10.4 | 10.06 | 9.91 | 5.29 |
| medium | Whisper Medium | 8.18 | 7.89 | 7.69 | 4.15 |
| large | Whisper Large-v3 | 7.49 | 7.11 | 6.8 | 3.78 |
| large_v3_turbo | Whisper large-v3-turbo | 8.32 | 8.1 | 7.76 | 4.27 |
| parakeet | Parakeet-TDT-0.6B-v2 | 13.03 | 11.73 | 8.35 | 6.3 |
| parakeet_ctc | Parakeet-CTC-1.1B | 17.71 | 15.65 | 11.18 | 8.93 |
| qwen3 | Qwen3-ASR-1.7B | 13.48 | 11.82 | 8.32 | 7.27 |

## Best model per mode

- **transcript_raw**: Whisper Large-v3 (7.49%)
- **transcript_clean**: Whisper Large-v3 (7.11%)
- **whisper_norm**: Whisper Large-v3 (6.80%)