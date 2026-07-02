# Codified error analysis — TIE_shorts — mode `transcript_clean`

Top-20 highest-WER clips per model, 5 models (100 rows -> 42 distinct clips).

**Artifact share (clip_over_run + content_mismatch): 61.9%** of the worst distinct clips are dataset artifacts, not model errors (row-weighted, counting each model-clip separately: 65.0%).

Classifier thresholds: clip_over_run = recall>=0.80 & ratio>=1.50; content_mismatch = recall<0.40; else genuine_error.

## Taxonomy (distinct worst-clips)

| category | n_clips | share_pct | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- |
| clip_over_run | 21 | 50.0 | 0.95 | 1.96 | 1.033 |
| content_mismatch | 5 | 11.9 | 0.19 | 0.87 | 1.003 |
| genuine_error | 16 | 38.1 | 0.7 | 1.36 | 0.817 |

## Cross-architecture agreement

12 clips appear in the worst-20 of >=3 distinct architectures (of 5 models spanning enc_dec / transducer / ctc / llm). Across those disjoint architectures the mean per-clip spread is recall std=0.018, length-ratio std=0.101 — near-identical failure on models that share no architecture is only possible if the fault is in the audio/reference.
