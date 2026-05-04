# WER Summary — All Models × All Modes

## Corpus WER (%) Matrix

| model | transcript_raw | transcript_clean | hf_raw | hf_clean |
| --- | --- | --- | --- | --- |
| base | 27.95 | 17.44 | 31.76 | 18.0 |
| large | 25.62 | 15.88 | 30.95 | 16.91 |
| medium | 24.14 | 14.72 | 29.83 | 15.73 |
| youtube_aligned | 51.72 | 51.88 | 65.94 | 53.35 |

## Mode Descriptions

| Mode | Reference | Before norm | After norm | Symmetric? | Purpose |
|------|-----------|-------------|------------|------------|---------|
| `transcript_raw` | Transcript | as-is | as-is | Yes | Upper bound baseline |
| `transcript_clean` | Transcript | Transcript | normalized | Yes | Gold standard (paper primary) |
| `hf_raw` | Normalised_Transcript | as-is | as-is | Yes | HuggingFace normalization as-is |
| `hf_clean` | Normalised_Transcript | Normalised_Transcript | normalized | Yes | HF + our normalizer |

## CSV Columns

Each result CSV contains:

| Column | Description |
|--------|-------------|
| `reference_raw` | Reference text **before** normalization |
| `reference` | Reference text **after** normalization (used for WER) |
| `hypothesis_raw` | Raw Whisper output **before** normalization |
| `hypothesis` | Whisper output **after** normalization (used for WER) |
| `wer` | Per-sample WER |

In `*_raw` modes: `reference_raw == reference` and `hypothesis_raw == hypothesis`.
