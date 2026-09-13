# TIE_shorts human review sample

Review status: complete (single annotator, non-blind). Mean WER across the 49 clips is 64.8% against the dataset reference and 17.0% against the corrected one. Full numbers in [`results/tie/analysis/human_review_stats.md`](../../results/tie/analysis/human_review_stats.md).

This folder holds the human review of lecture clips from NPTEL-derived TIE_shorts that several strong models get wrong at once. The reviewer listens to each clip, types what it actually says, and the rest is derived from that. All three corpora use the same protocol and the same error labels.

Files:

- `review_sheet.csv`: source of truth, 49 rows. One row per clip: `sample_id`, `reference`, the raw hypothesis and WER for each of Large-v3, Parakeet-TDT, Parakeet-CTC, Qwen3-ASR and Medium (`hyp_<model>`, `wer_<model>`), `avg_wer`, the WER of each against the corrected reference (`wer_<model>_true`, `avg_wer_true`), `n_models_flagged`, `native_region`, `duration_seconds`, then the reviewer columns: `reference_check`, `corrected_reference`, `normalised_corrected_reference`, `hyp_<model>_check`, `error_type`, `reviewer_decision`, `reviewer_notes`. There is no demographic column beyond `native_region`.
- `review_sheet.xlsx`: the same data with dropdowns on the reviewer columns.
- `error_types.csv`: the final reading of what went wrong on each clip, written after going through every row one by one. Its `workbook_error_type` column keeps the label first typed on a clip where it differs from the final one.
- `review_report.txt`: per-row log from the fill script.
- `audio/`: 16 kHz WAVs for the reviewer. Not tracked (WAV files are gitignored).

Verdicts: 46 reference errors, 2 genuine model errors, 1 left undecided.

| Error label | Clips |
|---|:---:|
| Reference error | 42 |
| Technical vocabulary | 13 |
| Disfluency | 7 |
| Misalignment | 5 |
| Number formatting | 1 |

A clip can carry more than one label. The label set is shared across all three sheets, so the same cause reads the same way everywhere; `analysis/fill_review_checks.py` lists it in full.

Selection rule: WER above 40 percent on `transcript_clean` for at least 3 of Large-v3, Parakeet-TDT, Parakeet-CTC and Qwen3-ASR. Medium is shown but not used for selection. That gives 49 clips.

Scripts, all shared by the three corpora:

```bash
python analysis/build_review_sample.py --dataset tie   # rebuild the empty sheet (needs Stage 2)
python analysis/extract_review_audio.py --dataset tie --csv analysis/tie_validation/review_sheet.csv --out-dir analysis/tie_validation/audio
python analysis/fill_review_checks.py --dataset tie    # derive the check, true-WER and label columns
python analysis/review_stats.py --dataset tie          # write results/tie/analysis/human_review_stats.md
```

The filled sheet is a finished artifact and is not regenerated. `build_review_sample.py` overwrites it, so only run that on an empty corpus.
