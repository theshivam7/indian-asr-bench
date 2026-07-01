# WER Summary — All Models × All Modes

## Corpus WER (%) Matrix

| model | transcript_raw | transcript_clean | hf_raw | hf_clean |
| --- | --- | --- | --- | --- |
| base | 17.91 | 17.53 | 20.24 | 18.07 |
| large | 16.31 | 15.93 | 19.14 | 16.94 |
| medium | 15.11 | 14.76 | 18.01 | 15.76 |
| medium_ft | 14.71 | 14.61 | 17.7 | 15.7 |
| medium_hf | 14.75 | 14.42 | 17.72 | 15.51 |
| parakeet | 15.97 | 15.6 | 18.54 | 16.4 |
| qwen3 | 18.15 | 16.66 | 17.99 | 17.61 |

## Mode Descriptions

| Mode | Reference | Cleanup | Symmetric? | Purpose |
|------|-----------|---------|------------|---------|
| `transcript_raw` | Transcript | minimal (lowercase, strip punctuation + wrapping quotes) | Yes | Light-cleanup baseline |
| `transcript_clean` | Transcript | full normalization | Yes | Gold standard (paper primary) |
| `hf_raw` | Normalised_Transcript | minimal (lowercase, strip punctuation + wrapping quotes) | Yes | HF source, light cleanup |
| `hf_clean` | Normalised_Transcript | full normalization | Yes | HF + our normalizer |

## CSV Columns

Each result CSV contains:

| Column | Description |
|--------|-------------|
| `reference_raw` | Reference text **before** normalization |
| `reference` | Reference text **after** normalization (used for WER) |
| `hypothesis_raw` | Raw Whisper output **before** normalization |
| `hypothesis` | Whisper output **after** normalization (used for WER) |
| `wer` | Per-sample WER |

In `*_raw` modes, `reference`/`hypothesis` carry minimal cleanup (lowercase + punctuation/wrapping-quote removal); `*_clean` modes apply full normalization.
