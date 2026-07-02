# WER Summary — TIE_shorts — All Models x Modes

## Corpus WER (%) Matrix

| model | transcript_raw | transcript_clean | hf_raw | hf_clean | whisper_norm |
| --- | --- | --- | --- | --- | --- |
| base | 17.91 | 17.53 | 20.24 | 18.07 | 17.03 |
| large | 16.31 | 15.93 | 19.14 | 16.94 | 15.76 |
| medium | 15.11 | 14.76 | 18.01 | 15.76 | 14.48 |
| medium_ft | 14.71 | 14.61 | 17.7 | 15.7 | 14.31 |
| medium_hf | 14.75 | 14.42 | 17.72 | 15.51 | 14.23 |
| parakeet | 15.97 | 15.6 | 18.54 | 16.4 | 15.17 |
| qwen3 | 18.15 | 16.66 | 17.99 | 17.61 | 15.4 |

## Modes

| Mode | Reference | Normalizer |
|---|---|---|
| `transcript_raw` | gold | minimal |
| `transcript_clean` | gold | custom |
| `hf_raw` | alt | minimal |
| `hf_clean` | alt | custom |
| `whisper_norm` | gold | whisper |
