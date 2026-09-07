# AESRC human review sample

Review status: pending. The sheet is built; no reviewer columns are filled yet. Nothing in the repo depends on this sheet until the review is done.

This folder holds the human review sample of AESRC2020 Indian-accent clips that several strong models get wrong at once. Same protocol as `analysis/tie_validation/`: the reviewer sees every model's hypothesis and judges, per clip, whether the high WER comes from the audio, the reference transcript, or the models. It uses the pretrained model outputs, not the AESRC fine-tuned checkpoints, because the goal is to find hard clips, not to judge fine-tuning.

Files:

- `review_sheet.csv`: source of truth, 28 rows. One row per clip: `sample_id`, `reference`, the raw hypothesis and WER for each of Large-v3, Parakeet-TDT, Parakeet-CTC, Qwen3-ASR and Medium (`hyp_<model>`, `wer_<model>`), `avg_wer`, `n_models_flagged`, `duration_seconds`, then the empty reviewer columns: `reference_check`, `corrected_reference`, `hyp_<model>_check`, `error_type`, `reviewer_decision`, `reviewer_notes`, and `audio_path`. There is no demographic column because the whole subset is Indian-accent.
- `review_sheet.xlsx`: the same data with dropdowns on the reviewer columns.
- `build_sample.py`: selects the clips and writes both files with the reviewer columns empty.
- `audio/`: 16 kHz WAVs for the reviewer. Not tracked (WAV files are gitignored).

Selection rule: WER above 40 percent on `transcript_clean` for at least 3 of Large-v3, Parakeet-TDT, Parakeet-CTC and Qwen3-ASR. Medium is shown but not used for selection. That gives 28 clips.

To rebuild the sheet (needs Stage-2 CSVs, so run `python normalize_and_score.py --dataset aesrc` first):

```bash
uv run --with openpyxl python3 analysis/aesrc_validation/build_sample.py
```

To extract the audio (run where the HF dataset cache already exists):

```bash
python analysis/extract_review_audio.py --dataset aesrc --csv analysis/aesrc_validation/review_sheet.csv --out-dir analysis/aesrc_validation/audio
```
