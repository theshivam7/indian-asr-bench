# WER Evaluation Summary -- TIE_shorts (Indian English)

## Corpus-level WER (%) by Model and Evaluation Mode

| model | transcript_raw | transcript_clean | hf_raw | hf_clean |
| --- | --- | --- | --- | --- |
| base | 17.91 | 17.53 | 20.24 | 18.07 |
| medium | 15.11 | 14.76 | 18.01 | 15.76 |
| large | 16.31 | 15.93 | 19.14 | 16.94 |
| parakeet | 15.97 | 15.6 | 18.54 | 16.4 |
| qwen3 | 18.15 | 16.66 | 17.99 | 17.61 |
| medium_hf | 14.75 | 14.42 | 17.72 | 15.51 |
| medium_ft | 14.71 | 14.61 | 17.7 | 15.7 |

## Evaluation Modes

| Mode | Reference | Before norm | After norm | Purpose |
|------|-----------|-------------|------------|---------|
| `transcript_raw` | Transcript | as-is | as-is | Upper bound baseline |
| `transcript_clean` | Transcript | Transcript | normalized | Gold standard — paper primary |
| `hf_raw` | Normalised_Transcript | as-is | as-is | HuggingFace normalization as-is |
| `hf_clean` | Normalised_Transcript | Normalised_Transcript | normalized | HF + our normalizer |

## Normalization Notes

- `transcript_clean` is the gold standard: uses original ground truth with correct forward normalization.
- `hf_raw` and `hf_clean` show the impact of the dataset's broken `Normalised_Transcript` (e.g. '1st' → 'one s t').
- All modes are **symmetric**: same normalization applied to both reference and hypothesis.
- Normalization: lowercase + expand contractions + fix possessives + digits/ordinals → words (num2words).

## Column Schema

Each result CSV contains:
`split, ID, Speaker_ID, Gender, Speech_Class, Native_Region, Speech_Duration_seconds,`
`Discipline_Group, Topic, model, mode, reference_source, reference_raw, reference,`
`hypothesis_raw, hypothesis, wer`

- `reference_raw`: original Transcript before normalization (for manual verification)
- `reference`: text used for WER after normalization
- `hypothesis_raw`: raw ASR output before normalization
- `hypothesis`: ASR output after normalization

## Best Model per Mode

- **transcript_raw**: Whisper Medium (FT) (14.71%)
- **transcript_clean**: Whisper Medium (HF) (14.42%)
- **hf_raw**: Whisper Medium (FT) (17.70%)
- **hf_clean**: Whisper Medium (HF) (15.51%)
