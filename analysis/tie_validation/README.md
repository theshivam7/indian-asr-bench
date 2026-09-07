# TIE human review sample

Review status: complete (single annotator, non-blind). Results in SUMMARY.md under Classifier validation (human review).

This folder holds the human review of TIE_shorts clips that several strong models get wrong at once. The point is to tell, per clip, whether the high WER comes from the audio, the reference transcript, or the models.

Files:

- `review_sheet.csv`: source of truth, 49 rows. One row per clip: `sample_id`, `reference`, the raw hypothesis and WER for each of Large-v3, Parakeet-TDT, Parakeet-CTC, Qwen3-ASR and Medium (`hyp_<model>`, `wer_<model>`), `avg_wer`, `n_models_flagged`, `native_region`, `duration_seconds`, then the reviewer columns: `reference_check`, `corrected_reference`, `normalised_corrected_reference`, `hyp_<model>_check`, `error_type`, `reviewer_decision`, `reviewer_notes`. The `wer_<model>_true` and `avg_wer_true` columns are WER against the corrected reference.
- `review_sheet.xlsx`: the same data with dropdowns on the reviewer columns.
- `build_sample.py`: selects the clips and writes both files with the reviewer columns empty.
- `fill_checks.py`: after `corrected_reference` is hand-filled, derives the `_check`, `_true` and decision columns from a text diff.
- `review_report.txt`: per-row log from `fill_checks.py`.
- `audio/`: 16 kHz WAVs for the reviewer. Not tracked (WAV files are gitignored).

Selection rule: WER above 40 percent on `transcript_clean` for at least 3 of Large-v3, Parakeet-TDT, Parakeet-CTC and Qwen3-ASR. Medium is shown but not used for selection. That gives 49 clips.

To rebuild the sheet (needs Stage-2 CSVs, so run `python normalize_and_score.py --dataset tie` first):

```bash
uv run --with openpyxl python3 analysis/tie_validation/build_sample.py
```

To extract the audio (run where the HF dataset cache already exists):

```bash
python analysis/extract_review_audio.py --dataset tie --csv analysis/tie_validation/review_sheet.csv --out-dir analysis/tie_validation/audio
```
