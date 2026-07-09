# Codified error analysis — TIE_shorts — mode `transcript_clean`

9 models, 985 common clips. Classifier thresholds: clip_over_run = recall>=0.80 & ratio>=1.50; content_mismatch = recall<0.40; short_ref = reference <4 words (recall/ratio are quantized below usability there, so those clips are unclassifiable by this instrument and excluded from the artifact share); else unflagged. Consensus = per-clip mean of recall/ratio across all models.

## Full-corpus taxonomy (all clips)

**Artifact share over the classifiable corpus: 1.2% (95% Wilson CI 0.7–2.1%; 12/984 clips with references >=4 words).** A further 1 clips (0.1% of the corpus) have <4-word references and are reported as `short_ref`: on those, single-word mistakes on decontextualized sub-second audio saturate WER, and the artifact signals carry no information.

| category | n_clips | share_pct | share_ci_lo | share_ci_hi | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 7 | 0.7 | 0.3 | 1.5 | 0.95 | 1.79 | 0.877 |
| content_mismatch | 5 | 0.5 | 0.2 | 1.2 | 0.18 | 0.86 | 1.006 |
| short_ref | 1 | 0.1 | 0.0 | 0.6 | 1.0 | 1.0 | 0.0 |
| unflagged | 972 | 98.7 | 97.8 | 99.2 | 0.92 | 1.03 | 0.167 |

## Artifact-adjusted corpus WER

Corpus WER on all common clips vs. excluding consensus-artifact clips (`wer_adjusted_pct`; `artifact_inflation_pp` is how many WER points the benchmark's own reference faults add to each model's score) and vs. excluding the `short_ref` clips (`wer_excl_shortref_pct`; quantifies the isolated-word subtask, a data-composition property rather than an artifact).

| model | display | wer_all_pct | wer_adjusted_pct | artifact_inflation_pp | wer_excl_shortref_pct | n_clips_all | n_clips_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | Whisper Tiny | 19.43 | 18.87 | 0.56 | 19.43 | 985 | 973 |
| base | Whisper Base | 17.53 | 16.96 | 0.57 | 17.53 | 985 | 973 |
| small | Whisper Small | 16.05 | 15.5 | 0.56 | 16.05 | 985 | 973 |
| medium | Whisper Medium | 14.76 | 14.21 | 0.55 | 14.76 | 985 | 973 |
| large | Whisper Large-v3 | 15.93 | 15.33 | 0.6 | 15.93 | 985 | 973 |
| large_v3_turbo | Whisper large-v3-turbo | 17.98 | 17.22 | 0.75 | 17.98 | 985 | 973 |
| parakeet | Parakeet-TDT-0.6B-v2 | 15.6 | 14.98 | 0.61 | 15.6 | 985 | 973 |
| parakeet_ctc | Parakeet-CTC-1.1B | 16.45 | 15.86 | 0.59 | 16.45 | 985 | 973 |
| qwen3 | Qwen3-ASR-1.7B | 16.66 | 16.08 | 0.59 | 16.66 | 985 | 973 |

## Inter-hypothesis agreement

`inter_hyp_dist` = mean normalized word edit distance BETWEEN model hypotheses (all pairs); `cross_arch_dist` = same, restricted to (enc_dec|llm) x (ctc|transducer) pairs; `hyp_to_ref_wer` = mean WER against the reference. Hypotheses that agree with each other but not with the reference localize the fault in the reference — across architectures that share no decoder or training objective. `ref_wer_grounded` vs `ref_wer_free` on flagged clips is a contamination probe (free-decoding models matching a flawed reference better than acoustically-grounded ones would suggest caption memorization).

| category | n_clips | inter_hyp_dist | cross_arch_dist | hyp_to_ref_wer | ref_wer_grounded | ref_wer_free |
| --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 7 | 0.226 | 0.196 | 0.877 | 0.887 | 0.874 |
| content_mismatch | 5 | 0.166 | 0.164 | 1.006 | 1.022 | 1.001 |
| short_ref | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| unflagged | 972 | 0.112 | 0.107 | 0.167 | 0.161 | 0.168 |

### Instrument audit: the same classifier WITHOUT the short-ref guard

A naive (guard-free) run flags 12/985 clips (1.2%) as artifacts. The agreement table below shows whether those naive flags carry the reference-fault signature (models agree with each other, disagree with the reference). Where they instead show high inter-hypothesis distance, the naive flags are classifier failures on short references, not data faults.

| category | n_clips | inter_hyp_dist | cross_arch_dist | hyp_to_ref_wer | ref_wer_grounded | ref_wer_free |
| --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 7 | 0.226 | 0.196 | 0.877 | 0.887 | 0.874 |
| content_mismatch | 5 | 0.166 | 0.164 | 1.006 | 1.022 | 1.001 |
| unflagged | 973 | 0.112 | 0.107 | 0.166 | 0.16 | 0.168 |

## Worst-20 tail (continuity with the original hand analysis)

Top-20 highest-WER clips per model (180 rows -> 55 distinct). **Tail artifact share: 65.5%** (95% Wilson CI 52.3–76.6%).

| category | n_clips | share_pct | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- |
| clip_over_run | 31 | 56.4 | 0.94 | 2.03 | 1.118 |
| content_mismatch | 5 | 9.1 | 0.18 | 0.86 | 1.006 |
| genuine_error | 19 | 34.5 | 0.69 | 1.41 | 0.863 |

17 tail clips appear in the worst-20 of >=3 distinct architectures; across those the mean per-clip spread is recall std=0.024, length-ratio std=0.159.

## Threshold sensitivity

Artifact share (over classifiable clips) under alternative classifier thresholds, including the short-reference guard itself (see threshold_sensitivity CSV for the full grid):

| vary | recall_overrun | ratio_overrun | recall_mismatch | min_ref_words | artifact_share_pct |
| --- | --- | --- | --- | --- | --- |
| mismatch | 0.8 | 1.5 | 0.3 | 4 | 1.2 |
| mismatch | 0.8 | 1.5 | 0.35 | 4 | 1.2 |
| mismatch | 0.8 | 1.5 | 0.4 | 4 | 1.2 |
| mismatch | 0.8 | 1.5 | 0.45 | 4 | 1.2 |
| mismatch | 0.8 | 1.5 | 0.5 | 4 | 1.2 |
| min_ref | 0.8 | 1.5 | 0.4 | 2 | 1.2 |
| min_ref | 0.8 | 1.5 | 0.4 | 3 | 1.2 |
| min_ref | 0.8 | 1.5 | 0.4 | 4 | 1.2 |
| min_ref | 0.8 | 1.5 | 0.4 | 5 | 1.2 |
| min_ref | 0.8 | 1.5 | 0.4 | 6 | 1.1 |
