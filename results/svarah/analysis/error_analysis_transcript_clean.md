# Codified error analysis: Svarah, mode `transcript_clean`

9 models, 6656 common clips. Classifier thresholds: clip_over_run = recall>=0.80 & ratio>=1.50; content_mismatch = recall<0.40; short_ref = reference <4 words (recall/ratio are quantized below usability there, so those clips are unclassifiable by this instrument and excluded from the artifact share); else unflagged. Consensus = per-clip mean of recall/ratio across all models.

## Full-corpus taxonomy (all clips)

**Artifact share over the classifiable corpus: 0.8% (95% Wilson CI 0.6–1.1%; 40/5126 clips with references >=4 words).** A further 1530 clips (23.0% of the corpus) have <4-word references and are reported as `short_ref`: on those, single-word mistakes on decontextualized sub-second audio saturate WER, and the artifact signals carry no information.

| category | n_clips | share_pct | share_ci_lo | share_ci_hi | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 11 | 0.2 | 0.1 | 0.3 | 0.96 | 1.84 | 0.934 |
| content_mismatch | 29 | 0.4 | 0.3 | 0.6 | 0.28 | 0.89 | 0.87 |
| short_ref | 1530 | 23.0 | 22.0 | 24.0 | 0.74 | 1.16 | 0.435 |
| unflagged | 5086 | 76.4 | 75.4 | 77.4 | 0.92 | 1.01 | 0.108 |

## Artifact-adjusted corpus WER

Corpus WER on all common clips vs. excluding consensus-artifact clips (`wer_adjusted_pct`; `artifact_inflation_pp` is how many WER points the benchmark's own reference faults add to each model's score) and vs. excluding the `short_ref` clips (`wer_excl_shortref_pct`; quantifies the isolated-word subtask, a data-composition property rather than an artifact).

| model | display | wer_all_pct | wer_adjusted_pct | artifact_inflation_pp | wer_excl_shortref_pct | n_clips_all | n_clips_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | Whisper Tiny | 19.96 | 19.57 | 0.39 | 18.28 | 6656 | 6616 |
| base | Whisper Base | 14.53 | 14.15 | 0.38 | 13.32 | 6656 | 6616 |
| small | Whisper Small | 10.06 | 9.7 | 0.37 | 9.25 | 6656 | 6616 |
| medium | Whisper Medium | 7.89 | 7.59 | 0.31 | 7.27 | 6656 | 6616 |
| large | Whisper Large-v3 | 7.11 | 6.77 | 0.34 | 6.59 | 6656 | 6616 |
| large_v3_turbo | Whisper large-v3-turbo | 8.1 | 7.75 | 0.35 | 7.5 | 6656 | 6616 |
| parakeet | Parakeet-TDT-0.6B-v2 | 11.73 | 11.38 | 0.34 | 11.11 | 6656 | 6616 |
| parakeet_ctc | Parakeet-CTC-1.1B | 15.65 | 15.29 | 0.36 | 14.84 | 6656 | 6616 |
| qwen3 | Qwen3-ASR-1.7B | 11.82 | 11.47 | 0.35 | 11.54 | 6656 | 6616 |

## Inter-hypothesis agreement

`inter_hyp_dist` = mean normalized word edit distance BETWEEN model hypotheses (all pairs); `cross_arch_dist` = same, restricted to (enc_dec|llm) x (ctc|transducer) pairs; `hyp_to_ref_wer` = mean WER against the reference. Hypotheses that agree with each other but not with the reference localize the fault in the reference, across architectures that share no decoder or training objective. `ref_wer_grounded` vs `ref_wer_free` on flagged clips is a contamination probe (free-decoding models matching a flawed reference better than acoustically-grounded ones would suggest caption memorization).

| category | n_clips | inter_hyp_dist | cross_arch_dist | hyp_to_ref_wer | ref_wer_grounded | ref_wer_free |
| --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 11 | 0.169 | 0.208 | 0.934 | 1.214 | 0.854 |
| content_mismatch | 29 | 0.791 | 0.806 | 0.87 | 0.833 | 0.88 |
| short_ref | 1530 | 0.387 | 0.417 | 0.435 | 0.421 | 0.439 |
| unflagged | 5086 | 0.121 | 0.131 | 0.108 | 0.121 | 0.104 |

### Instrument audit: the same classifier WITHOUT the short-ref guard

A naive (guard-free) run flags 320/6656 clips (4.8%) as artifacts. The agreement table below shows whether those naive flags carry the reference-fault signature (models agree with each other, disagree with the reference). Where they instead show high inter-hypothesis distance, the naive flags are classifier failures on short references, not data faults.

| category | n_clips | inter_hyp_dist | cross_arch_dist | hyp_to_ref_wer | ref_wer_grounded | ref_wer_free |
| --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 28 | 0.299 | 0.309 | 1.287 | 0.849 | 1.412 |
| content_mismatch | 292 | 0.916 | 0.937 | 1.261 | 1.181 | 1.284 |
| unflagged | 6336 | 0.151 | 0.165 | 0.133 | 0.147 | 0.129 |

## Worst-20 tail (continuity with the original hand analysis)

Top-20 highest-WER clips per model (180 rows -> 117 distinct). **Tail artifact share: 3.4%** (95% Wilson CI 1.3–8.5%).

| category | n_clips | share_pct | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- |
| clip_over_run | 1 | 0.9 | 1.0 | 4.06 | 3.062 |
| content_mismatch | 3 | 2.6 | 0.0 | 3.1 | 3.101 |
| genuine_error | 2 | 1.7 | 0.62 | 5.87 | 5.25 |
| short_ref | 111 | 94.9 | 0.05 | 9.58 | 9.524 |

10 tail clips appear in the worst-20 of >=3 distinct architectures; across those the mean per-clip spread is recall std=0.06, length-ratio std=1.837.

## Threshold sensitivity

Artifact share (over classifiable clips) under alternative classifier thresholds, including the short-reference guard itself (see threshold_sensitivity CSV for the full grid):

| vary | recall_overrun | ratio_overrun | recall_mismatch | min_ref_words | artifact_share_pct |
| --- | --- | --- | --- | --- | --- |
| mismatch | 0.8 | 1.5 | 0.3 | 4 | 0.5 |
| mismatch | 0.8 | 1.5 | 0.35 | 4 | 0.6 |
| mismatch | 0.8 | 1.5 | 0.4 | 4 | 0.8 |
| mismatch | 0.8 | 1.5 | 0.45 | 4 | 1.1 |
| mismatch | 0.8 | 1.5 | 0.5 | 4 | 1.2 |
| min_ref | 0.8 | 1.5 | 0.4 | 2 | 1.5 |
| min_ref | 0.8 | 1.5 | 0.4 | 3 | 1.0 |
| min_ref | 0.8 | 1.5 | 0.4 | 4 | 0.8 |
| min_ref | 0.8 | 1.5 | 0.4 | 5 | 0.7 |
| min_ref | 0.8 | 1.5 | 0.4 | 6 | 0.6 |
