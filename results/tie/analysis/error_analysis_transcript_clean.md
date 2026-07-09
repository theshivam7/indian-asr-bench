# Codified error analysis — TIE_shorts — mode `transcript_clean`

7 models, 985 common clips. Classifier thresholds: clip_over_run = recall>=0.80 & ratio>=1.50; content_mismatch = recall<0.40; short_ref = reference <4 words (recall/ratio are quantized below usability there, so those clips are unclassifiable by this instrument and excluded from the artifact share); else unflagged. Consensus = per-clip mean of recall/ratio across all models.

## Full-corpus taxonomy (all clips)

**Artifact share over the classifiable corpus: 1.1% (95% Wilson CI 0.6–2.0%; 11/984 clips with references >=4 words).** A further 1 clips (0.1% of the corpus) have <4-word references and are reported as `short_ref`: on those, single-word mistakes on decontextualized sub-second audio saturate WER, and the artifact signals carry no information.

| category | n_clips | share_pct | share_ci_lo | share_ci_hi | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 6 | 0.6 | 0.3 | 1.3 | 0.95 | 1.83 | 0.899 |
| content_mismatch | 5 | 0.5 | 0.2 | 1.2 | 0.19 | 0.87 | 1.003 |
| short_ref | 1 | 0.1 | 0.0 | 0.6 | 1.0 | 1.0 | 0.0 |
| unflagged | 973 | 98.8 | 97.9 | 99.3 | 0.92 | 1.03 | 0.166 |

## Artifact-adjusted corpus WER

Corpus WER on all common clips vs. excluding consensus-artifact clips (`wer_adjusted_pct`; `artifact_inflation_pp` is how many WER points the benchmark's own reference faults add to each model's score) and vs. excluding the `short_ref` clips (`wer_excl_shortref_pct`; quantifies the isolated-word subtask, a data-composition property rather than an artifact).

| model | display | wer_all_pct | wer_adjusted_pct | artifact_inflation_pp | wer_excl_shortref_pct | n_clips_all | n_clips_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | Whisper Tiny | 19.43 | 18.89 | 0.54 | 19.43 | 985 | 974 |
| base | Whisper Base | 17.53 | 16.97 | 0.55 | 17.53 | 985 | 974 |
| small | Whisper Small | 16.05 | 15.51 | 0.54 | 16.05 | 985 | 974 |
| medium | Whisper Medium | 14.76 | 14.23 | 0.53 | 14.76 | 985 | 974 |
| large | Whisper Large-v3 | 15.93 | 15.36 | 0.57 | 15.93 | 985 | 974 |
| parakeet | Parakeet-TDT-0.6B-v2 | 15.6 | 15.0 | 0.6 | 15.6 | 985 | 974 |
| qwen3 | Qwen3-ASR-1.7B | 16.66 | 16.1 | 0.57 | 16.66 | 985 | 974 |

## Inter-hypothesis agreement

`inter_hyp_dist` = mean normalized word edit distance BETWEEN model hypotheses (all pairs); `cross_arch_dist` = same, restricted to (enc_dec|llm) x (ctc|transducer) pairs; `hyp_to_ref_wer` = mean WER against the reference. Hypotheses that agree with each other but not with the reference localize the fault in the reference — across architectures that share no decoder or training objective. `ref_wer_grounded` vs `ref_wer_free` on flagged clips is a contamination probe (free-decoding models matching a flawed reference better than acoustically-grounded ones would suggest caption memorization).

| category | n_clips | inter_hyp_dist | cross_arch_dist | hyp_to_ref_wer | ref_wer_grounded | ref_wer_free |
| --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 6 | 0.202 | 0.182 | 0.899 | 0.996 | 0.883 |
| content_mismatch | 5 | 0.172 | 0.17 | 1.003 | 1.025 | 0.999 |
| short_ref | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| unflagged | 973 | 0.113 | 0.105 | 0.166 | 0.158 | 0.167 |

### Instrument audit: the same classifier WITHOUT the short-ref guard

A naive (guard-free) run flags 11/985 clips (1.1%) as artifacts. The agreement table below shows whether those naive flags carry the reference-fault signature (models agree with each other, disagree with the reference). Where they instead show high inter-hypothesis distance, the naive flags are classifier failures on short references, not data faults.

| category | n_clips | inter_hyp_dist | cross_arch_dist | hyp_to_ref_wer | ref_wer_grounded | ref_wer_free |
| --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 6 | 0.202 | 0.182 | 0.899 | 0.996 | 0.883 |
| content_mismatch | 5 | 0.172 | 0.17 | 1.003 | 1.025 | 0.999 |
| unflagged | 974 | 0.113 | 0.105 | 0.166 | 0.158 | 0.167 |

## Worst-20 tail (continuity with the original hand analysis)

Top-20 highest-WER clips per model (140 rows -> 47 distinct). **Tail artifact share: 61.7%** (95% Wilson CI 47.4–74.2%).

| category | n_clips | share_pct | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- |
| clip_over_run | 24 | 51.1 | 0.95 | 1.94 | 1.008 |
| content_mismatch | 5 | 10.6 | 0.19 | 0.87 | 1.003 |
| genuine_error | 18 | 38.3 | 0.68 | 1.37 | 0.845 |

13 tail clips appear in the worst-20 of >=3 distinct architectures; across those the mean per-clip spread is recall std=0.02, length-ratio std=0.116.

## Threshold sensitivity

Artifact share (over classifiable clips) under alternative classifier thresholds, including the short-reference guard itself (see threshold_sensitivity CSV for the full grid):

| vary | recall_overrun | ratio_overrun | recall_mismatch | min_ref_words | artifact_share_pct |
| --- | --- | --- | --- | --- | --- |
| mismatch | 0.8 | 1.5 | 0.3 | 4 | 1.1 |
| mismatch | 0.8 | 1.5 | 0.35 | 4 | 1.1 |
| mismatch | 0.8 | 1.5 | 0.4 | 4 | 1.1 |
| mismatch | 0.8 | 1.5 | 0.45 | 4 | 1.1 |
| mismatch | 0.8 | 1.5 | 0.5 | 4 | 1.1 |
| min_ref | 0.8 | 1.5 | 0.4 | 2 | 1.1 |
| min_ref | 0.8 | 1.5 | 0.4 | 3 | 1.1 |
| min_ref | 0.8 | 1.5 | 0.4 | 4 | 1.1 |
| min_ref | 0.8 | 1.5 | 0.4 | 5 | 1.1 |
| min_ref | 0.8 | 1.5 | 0.4 | 6 | 1.0 |
