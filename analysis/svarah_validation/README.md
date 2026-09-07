# Svarah human review sample

Review status: pending. The sheet is built; no reviewer columns are filled yet. Nothing in the repo depends on this sheet until the review is done.

This folder holds the human review sample of Svarah clips that several strong models get wrong at once. Same protocol as `analysis/tie_validation/`: the reviewer sees every model's hypothesis and judges, per clip, whether the high WER comes from the audio, the reference transcript, or the models.

Files:

- `review_sheet.csv`: the full flagged pool, 499 rows. One row per clip: `sample_id`, `reference`, the raw hypothesis and WER for each of Large-v3, Parakeet-TDT, Parakeet-CTC, Qwen3-ASR and Medium (`hyp_<model>`, `wer_<model>`), `avg_wer`, `n_models_flagged`, `native_language`, `duration_seconds`, `ref_words`, then the empty reviewer columns: `reference_check`, `corrected_reference`, `hyp_<model>_check`, `error_type`, `reviewer_decision`, `reviewer_notes`.
- `review_sample.csv`: the 60 clips to review, same columns plus `audio_path`.
- `review_sheet.xlsx`, `review_sample.xlsx`: the same data with dropdowns on the reviewer columns.
- `build_sample.py`: selects the clips and writes all four files.
- `audio/`: 16 kHz WAVs for the reviewer. Not tracked (WAV files are gitignored).

Selection rule: WER above 40 percent on `transcript_clean` for at least 3 of Large-v3, Parakeet-TDT, Parakeet-CTC and Qwen3-ASR. Medium is shown but not used for selection. That flags 499 clips. Many of those have one-word references, where WER can only be 0 or 100 percent, so the pool is filtered to references of at least 3 words (209 clips) and then sampled by duration band with a fixed seed to get 60 clips. See the comments in `build_sample.py` for the bands.

To rebuild the sheet (needs Stage-2 CSVs, so run `python normalize_and_score.py --dataset svarah` first):

```bash
uv run --with openpyxl python3 analysis/svarah_validation/build_sample.py
```

To extract the audio (run where the HF dataset cache already exists):

```bash
python analysis/extract_review_audio.py --dataset svarah --csv analysis/svarah_validation/review_sample.csv --out-dir analysis/svarah_validation/audio
```
