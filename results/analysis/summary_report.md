# WER Evaluation Summary -- TIE_shorts (Indian English)

## Corpus-level WER (%) by Model and Evaluation Mode

| model | transcript_raw | transcript_clean | hf_raw | hf_clean |
| --- | --- | --- | --- | --- |
| base | 27.95 | 17.44 | 31.76 | 18.0 |
| medium | 24.14 | 14.72 | 29.83 | 15.73 |
| large | 25.62 | 15.88 | 30.95 | 16.91 |
| youtube | 100.0 | 100.0 | 100.0 | 100.0 |

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
- `hypothesis_raw`: raw Whisper output before normalization
- `hypothesis`: Whisper output after normalization

## Best Model per Mode

- **transcript_raw**: Whisper medium (24.14%)
- **transcript_clean**: Whisper medium (14.72%)
- **hf_raw**: Whisper medium (29.83%)
- **hf_clean**: Whisper medium (15.73%)
