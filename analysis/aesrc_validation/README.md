# AESRC2020 (Indian) human review sample

Review status: complete (single annotator, non-blind). Mean WER across the 28 clips is 51.5% against the dataset reference and 48.4% against the corrected one. Full numbers in [`results/aesrc/analysis/human_review_stats.md`](../../results/aesrc/analysis/human_review_stats.md).

This folder holds the human review of AESRC2020 Indian-accent clips that several strong models get wrong at once. The reviewer listens to each clip, types what it actually says, and the rest is derived from that. All three corpora use the same protocol and the same error labels.

Files:

- `review_sheet.csv`: source of truth, 28 rows. One row per clip: `sample_id`, `reference`, the raw hypothesis and WER for each of Large-v3, Parakeet-TDT, Parakeet-CTC, Qwen3-ASR and Medium (`hyp_<model>`, `wer_<model>`), `avg_wer`, the WER of each against the corrected reference (`wer_<model>_true`, `avg_wer_true`), `n_models_flagged`, `duration_seconds`, then the reviewer columns: `reference_check`, `corrected_reference`, `normalised_corrected_reference`, `hyp_<model>_check`, `error_type`, `reviewer_decision`, `reviewer_notes`, and `audio_path`. There is no demographic column because the whole subset is Indian-accent.
- `review_sheet.xlsx`: the same data with dropdowns on the reviewer columns.
- `error_types.csv`: the final reading of what went wrong on each clip, written after going through every row one by one. Its `workbook_error_type` column keeps the label first typed on a clip where it differs from the final one.
- `review_report.txt`: per-row log from the fill script.
- `audio/`: 16 kHz WAVs for the reviewer. Not tracked (WAV files are gitignored).

Verdicts: 22 genuine model errors, 3 reference errors, 2 not real errors, 1 unsure.

| Error label | Clips |
|---|:---:|
| Hindi named entity | 13 |
| Accent / pronunciation | 7 |
| Short utterance | 5 |
| Foreign named entity | 5 |
| Reference error | 4 |
| Indian-language named entity | 3 |
| Spelling variant | 3 |
| English name or rare word | 2 |
| Number formatting | 1 |
| Acronym or code | 1 |
| Brand or product name | 1 |

A clip can carry more than one label. The label set is shared across all three sheets, so the same cause reads the same way everywhere; `analysis/fill_review_checks.py` lists it in full.

Selection rule: WER above 40 percent on `transcript_clean` for at least 3 of Large-v3, Parakeet-TDT, Parakeet-CTC and Qwen3-ASR. Medium is shown but not used for selection. That gives 28 clips.

Scripts, all shared by the three corpora:

```bash
python analysis/build_review_sample.py --dataset aesrc   # rebuild the empty sheet (needs Stage 2)
python analysis/extract_review_audio.py --dataset aesrc --csv analysis/aesrc_validation/review_sheet.csv --out-dir analysis/aesrc_validation/audio
python analysis/fill_review_checks.py --dataset aesrc    # derive the check, true-WER and label columns
python analysis/review_stats.py --dataset aesrc          # write results/aesrc/analysis/human_review_stats.md
```

The filled sheet is a finished artifact and is not regenerated. `build_review_sample.py` overwrites it, so only run that on an empty corpus.
