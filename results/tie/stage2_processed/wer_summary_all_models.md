# WER Summary — TIE_shorts — All Models x Modes

## Corpus WER (%) Matrix

| model | transcript_raw | transcript_clean | hf_raw | hf_clean | whisper_norm |
| --- | --- | --- | --- | --- | --- |
| base | 17.91 | 17.53 | 20.24 | 18.07 | 17.03 |
| large | 16.31 | 15.93 | 19.14 | 16.94 | 15.76 |
| large_v3_turbo | 18.35 | 17.98 | 21.1 | 18.91 | 17.75 |
| medium | 15.11 | 14.76 | 18.01 | 15.76 | 14.48 |
| medium_ft | 14.71 | 14.61 | 17.7 | 15.7 | 14.31 |
| medium_hf | 14.75 | 14.42 | 17.72 | 15.51 | 14.23 |
| parakeet | 15.97 | 15.6 | 18.54 | 16.4 | 15.17 |
| parakeet_ctc | 18.53 | 16.45 | 17.15 | 16.98 | 16.19 |
| qwen3 | 18.15 | 16.66 | 17.99 | 17.61 | 15.4 |
| small | 16.44 | 16.05 | 19.2 | 16.96 | 15.8 |
| small_ft | 16.39 | 16.21 | 19.17 | 17.2 | 15.64 |
| small_hf | 17.78 | 17.38 | 20.53 | 18.27 | 16.93 |
| tiny | 19.79 | 19.43 | 22.2 | 20.07 | 19.01 |
| tiny_ft | 19.54 | 19.14 | 21.82 | 19.63 | 18.45 |
| tiny_hf | 22.49 | 22.1 | 24.86 | 22.67 | 21.72 |

## Modes

| Mode | Reference | Normalizer |
|---|---|---|
| `transcript_raw` | gold | minimal |
| `transcript_clean` | gold | custom |
| `hf_raw` | alt | minimal |
| `hf_clean` | alt | custom |
| `whisper_norm` | gold | whisper |
