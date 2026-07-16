# WER Summary — AESRC2020 (Indian) — All Models x Modes

## Corpus WER (%) Matrix

| model | transcript_raw | transcript_clean | whisper_norm |
| --- | --- | --- | --- |
| base | 10.27 | 9.96 | 9.64 |
| large | 5.39 | 5.2 | 4.78 |
| large_v3_turbo | 6.13 | 5.81 | 5.56 |
| medium | 6.05 | 5.73 | 5.41 |
| medium_aesrc_ft | 4.37 | 4.48 | 4.18 |
| medium_hf | 5.92 | 5.63 | 5.26 |
| parakeet | 6.19 | 6.26 | 5.93 |
| parakeet_ctc | 7.38 | 7.5 | 7.13 |
| qwen3 | 5.14 | 5.23 | 4.89 |
| small | 7.52 | 7.23 | 6.96 |
| small_aesrc_ft | 5.55 | 5.64 | 5.36 |
| small_hf | 7.49 | 7.22 | 6.91 |
| tiny | 13.91 | 13.66 | 13.21 |
| tiny_aesrc_ft | 12.49 | 12.64 | 9.83 |
| tiny_hf | 17.64 | 17.45 | 16.97 |

## Modes

| Mode | Reference | Normalizer |
|---|---|---|
| `transcript_raw` | gold | minimal |
| `transcript_clean` | gold | custom |
| `whisper_norm` | gold | whisper |
