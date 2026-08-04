# WER Summary: Svarah, all models x modes

## Corpus WER (%) Matrix

| model | transcript_raw | transcript_clean | whisper_norm |
| --- | --- | --- | --- |
| base | 14.88 | 14.53 | 14.37 |
| large | 7.49 | 7.11 | 6.8 |
| large_v3_turbo | 8.32 | 8.1 | 7.76 |
| medium | 8.18 | 7.89 | 7.69 |
| parakeet | 13.03 | 11.73 | 8.35 |
| parakeet_ctc | 17.71 | 15.65 | 11.18 |
| qwen3 | 13.48 | 11.82 | 8.32 |
| small | 10.4 | 10.06 | 9.91 |
| tiny | 20.33 | 19.96 | 19.52 |

## Modes

| Mode | Reference | Normalizer |
|---|---|---|
| `transcript_raw` | gold | minimal |
| `transcript_clean` | gold | custom |
| `whisper_norm` | gold | whisper |
