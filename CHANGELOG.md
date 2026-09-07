# Changelog

Dated list of changes that affect a published number, a tracked result file, or how the
repository is used. Code-only refactors are not listed; see the git log for those.

## 2026-09-08

- TIE `whisper_norm` now scores the same 985 clips as every other mode. One clip whose
  reference is `..` used to survive only under the Whisper normalizer. The TIE
  `whisper_norm` corpus WER moves by 0.01 pp for seven models; significance counts and the
  six flipped verdicts are unchanged.
- The p90 and p95 columns in `wer_summary_all_models.csv` now use the nearest-rank
  percentile. No committed value changed (the old index only differs when 0.9 x n is a
  whole number).
- The batch-1 single-stream efficiency experiment was removed from the repository. The
  quality-gated offline throughput sweep is the only efficiency result.
- Per-clip Stage-2 CSVs and the flat per-model top-20 dumps are no longer tracked. Run
  `python normalize_and_score.py --dataset <ds>` on a fresh clone before any `analysis/`
  script. CI checks the per-dataset summary CSV instead.
- `results/tie/analysis/summary_report.md` and the AESRC one now name the best pretrained
  model per mode; fine-tuned and HF-engine rows were being counted before.
- Added `analysis/tie_validation/review_stats.py`, which recomputes the human-review
  statistics quoted in SUMMARY.md. The bootstrap CI lower bound is 40.3 pp with the
  script's seed (40.4 was quoted before).
- Generated reports use ASCII hyphens instead of en dashes and the Unicode minus sign.
- README.md and SUMMARY.md rewritten to the repository scope: fine-tuning kept as an
  exploratory study, human-review status stated per corpus, tested environment, data
  availability and a table-to-command reproduction map added.

## 2026-09-04

- Bootstrap resamples raised from 2,000 to 10,000 so the two-sided p floor (2/(B+1))
  sits at 0.0002 instead of 0.001, which Holm correction across 36 pairs was pushing to
  0.036.
- `num2words` missing now raises instead of leaving digits unconverted in the
  `*_clean` modes.
- The throughput aggregator reports `gate_cost_x` and `padded_rtfx_audio_s_per_s` for
  every model.
