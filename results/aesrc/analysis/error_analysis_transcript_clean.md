# Codified error analysis — AESRC2020 (Indian) — mode `transcript_clean`

9 models, 1731 common clips. Classifier thresholds: clip_over_run = recall>=0.80 & ratio>=1.50; content_mismatch = recall<0.40; short_ref = reference <4 words (recall/ratio are quantized below usability there, so those clips are unclassifiable by this instrument and excluded from the artifact share); else unflagged. Consensus = per-clip mean of recall/ratio across all models.

## Full-corpus taxonomy (all clips)

**Artifact share over the classifiable corpus: 0.1% (95% Wilson CI 0.0–0.4%; 2/1719 clips with references >=4 words).** A further 12 clips (0.7% of the corpus) have <4-word references and are reported as `short_ref`: on those, single-word mistakes on decontextualized sub-second audio saturate WER, and the artifact signals carry no information.

| category | n_clips | share_pct | share_ci_lo | share_ci_hi | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 0 | 0.0 | 0.0 | 0.2 | N/A | N/A | N/A |
| content_mismatch | 2 | 0.1 | 0.0 | 0.4 | 0.36 | 0.9 | 0.674 |
| short_ref | 12 | 0.7 | 0.4 | 1.2 | 0.74 | 1.08 | 0.364 |
| unflagged | 1717 | 99.2 | 98.6 | 99.5 | 0.93 | 1.0 | 0.082 |

## Artifact-adjusted corpus WER

Corpus WER on all common clips vs. excluding consensus-artifact clips (`wer_adjusted_pct`; `artifact_inflation_pp` is how many WER points the benchmark's own reference faults add to each model's score) and vs. excluding the `short_ref` clips (`wer_excl_shortref_pct`; quantifies the isolated-word subtask, a data-composition property rather than an artifact).

| model | display | wer_all_pct | wer_adjusted_pct | artifact_inflation_pp | wer_excl_shortref_pct | n_clips_all | n_clips_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | Whisper Tiny | 13.66 | 13.61 | 0.06 | 13.61 | 1731 | 1729 |
| base | Whisper Base | 9.96 | 9.88 | 0.08 | 9.92 | 1731 | 1729 |
| small | Whisper Small | 7.23 | 7.19 | 0.04 | 7.15 | 1731 | 1729 |
| medium | Whisper Medium | 5.73 | 5.7 | 0.04 | 5.7 | 1731 | 1729 |
| large | Whisper Large-v3 | 5.2 | 5.15 | 0.05 | 5.18 | 1731 | 1729 |
| large_v3_turbo | Whisper large-v3-turbo | 5.81 | 5.75 | 0.05 | 5.78 | 1731 | 1729 |
| parakeet | Parakeet-TDT-0.6B-v2 | 6.26 | 6.21 | 0.05 | 6.21 | 1731 | 1729 |
| parakeet_ctc | Parakeet-CTC-1.1B | 7.5 | 7.45 | 0.05 | 7.45 | 1731 | 1729 |
| qwen3 | Qwen3-ASR-1.7B | 5.23 | 5.2 | 0.03 | 5.17 | 1731 | 1729 |

## Inter-hypothesis agreement

`inter_hyp_dist` = mean normalized word edit distance BETWEEN model hypotheses (all pairs); `cross_arch_dist` = same, restricted to (enc_dec|llm) x (ctc|transducer) pairs; `hyp_to_ref_wer` = mean WER against the reference. Hypotheses that agree with each other but not with the reference localize the fault in the reference — across architectures that share no decoder or training objective. `ref_wer_grounded` vs `ref_wer_free` on flagged clips is a contamination probe (free-decoding models matching a flawed reference better than acoustically-grounded ones would suggest caption memorization).

| category | n_clips | inter_hyp_dist | cross_arch_dist | hyp_to_ref_wer | ref_wer_grounded | ref_wer_free |
| --- | --- | --- | --- | --- | --- | --- |
| content_mismatch | 2 | 0.822 | 0.791 | 0.674 | 0.706 | 0.665 |
| short_ref | 12 | 0.359 | 0.382 | 0.364 | 0.389 | 0.357 |
| unflagged | 1717 | 0.087 | 0.089 | 0.082 | 0.078 | 0.083 |

### Instrument audit: the same classifier WITHOUT the short-ref guard

A naive (guard-free) run flags 4/1731 clips (0.2%) as artifacts. The agreement table below shows whether those naive flags carry the reference-fault signature (models agree with each other, disagree with the reference). Where they instead show high inter-hypothesis distance, the naive flags are classifier failures on short references, not data faults.

| category | n_clips | inter_hyp_dist | cross_arch_dist | hyp_to_ref_wer | ref_wer_grounded | ref_wer_free |
| --- | --- | --- | --- | --- | --- | --- |
| content_mismatch | 4 | 0.924 | 0.92 | 0.906 | 0.853 | 0.922 |
| unflagged | 1727 | 0.088 | 0.09 | 0.083 | 0.079 | 0.084 |

## Worst-20 tail (continuity with the original hand analysis)

Top-20 highest-WER clips per model (180 rows -> 77 distinct). **Tail artifact share: 20.8%** (95% Wilson CI 13.2–31.1%).

| category | n_clips | share_pct | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- |
| content_mismatch | 16 | 20.8 | 0.28 | 1.11 | 0.942 |
| genuine_error | 52 | 67.5 | 0.54 | 1.13 | 0.652 |
| short_ref | 9 | 11.7 | 0.45 | 1.34 | 0.927 |

11 tail clips appear in the worst-20 of >=3 distinct architectures; across those the mean per-clip spread is recall std=0.129, length-ratio std=0.209.

## Threshold sensitivity

Artifact share (over classifiable clips) under alternative classifier thresholds, including the short-reference guard itself (see threshold_sensitivity CSV for the full grid):

| vary | recall_overrun | ratio_overrun | recall_mismatch | min_ref_words | artifact_share_pct |
| --- | --- | --- | --- | --- | --- |
| mismatch | 0.8 | 1.5 | 0.3 | 4 | 0.0 |
| mismatch | 0.8 | 1.5 | 0.35 | 4 | 0.0 |
| mismatch | 0.8 | 1.5 | 0.4 | 4 | 0.1 |
| mismatch | 0.8 | 1.5 | 0.45 | 4 | 0.1 |
| mismatch | 0.8 | 1.5 | 0.5 | 4 | 0.2 |
| min_ref | 0.8 | 1.5 | 0.4 | 2 | 0.2 |
| min_ref | 0.8 | 1.5 | 0.4 | 3 | 0.2 |
| min_ref | 0.8 | 1.5 | 0.4 | 4 | 0.1 |
| min_ref | 0.8 | 1.5 | 0.4 | 5 | 0.1 |
| min_ref | 0.8 | 1.5 | 0.4 | 6 | 0.1 |
