# Codified error analysis — TIE_shorts — mode `transcript_clean`

5 models, 985 common clips. Classifier thresholds: clip_over_run = recall>=0.80 & ratio>=1.50; content_mismatch = recall<0.40; else unflagged. Consensus = per-clip mean of recall/ratio across all models.

## Full-corpus taxonomy (all clips)

**Artifact share over the full corpus: 1.0% (95% Wilson CI 0.6–1.9%)** of clips carry an artifact signature.

| category | n_clips | share_pct | share_ci_lo | share_ci_hi | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 5 | 0.5 | 0.2 | 1.2 | 0.95 | 1.92 | 0.988 |
| content_mismatch | 5 | 0.5 | 0.2 | 1.2 | 0.19 | 0.87 | 1.003 |
| unflagged | 975 | 99.0 | 98.1 | 99.4 | 0.92 | 1.03 | 0.161 |

## Artifact-adjusted corpus WER

Corpus WER on all common clips vs. on the unflagged subset. `artifact_inflation_pp` is how many WER points the benchmark's own artifacts add to each model's score.

| model | display | wer_all_pct | wer_adjusted_pct | artifact_inflation_pp | n_clips_all | n_clips_adjusted |
| --- | --- | --- | --- | --- | --- | --- |
| base | Whisper Base | 17.53 | 16.99 | 0.54 | 985 | 975 |
| medium | Whisper Medium | 14.76 | 14.23 | 0.53 | 985 | 975 |
| large | Whisper Large | 15.93 | 15.36 | 0.57 | 985 | 975 |
| parakeet | Parakeet-TDT-0.6B | 15.6 | 15.02 | 0.58 | 985 | 975 |
| qwen3 | Qwen3-ASR-1.7B | 16.66 | 16.12 | 0.55 | 985 | 975 |

## Inter-hypothesis agreement

`inter_hyp_dist` = mean normalized word edit distance BETWEEN model hypotheses (all pairs); `cross_arch_dist` = same, restricted to (enc_dec|llm) x (ctc|transducer) pairs; `hyp_to_ref_wer` = mean WER against the reference. Hypotheses that agree with each other but not with the reference localize the fault in the reference — across architectures that share no decoder or training objective. `ref_wer_grounded` vs `ref_wer_free` on flagged clips is a contamination probe (free-decoding models matching a flawed reference better than acoustically-grounded ones would suggest caption memorization).

| category | n_clips | inter_hyp_dist | cross_arch_dist | hyp_to_ref_wer | ref_wer_grounded | ref_wer_free |
| --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 5 | 0.2 | 0.177 | 0.988 | 1.021 | 0.98 |
| content_mismatch | 5 | 0.165 | 0.163 | 1.003 | 1.025 | 0.997 |
| unflagged | 975 | 0.106 | 0.099 | 0.161 | 0.159 | 0.162 |

## Worst-20 tail (continuity with the original hand analysis)

Top-20 highest-WER clips per model (100 rows -> 42 distinct). **Tail artifact share: 61.9%** (95% Wilson CI 46.8–75.0%).

| category | n_clips | share_pct | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- |
| clip_over_run | 21 | 50.0 | 0.95 | 1.96 | 1.033 |
| content_mismatch | 5 | 11.9 | 0.19 | 0.87 | 1.003 |
| genuine_error | 16 | 38.1 | 0.7 | 1.36 | 0.817 |

12 tail clips appear in the worst-20 of >=3 distinct architectures; across those the mean per-clip spread is recall std=0.018, length-ratio std=0.101.

## Threshold sensitivity

Full-corpus artifact share under alternative classifier thresholds (see threshold_sensitivity CSV for the full grid):

| vary | recall_overrun | ratio_overrun | recall_mismatch | artifact_share_pct |
| --- | --- | --- | --- | --- |
| mismatch | 0.8 | 1.5 | 0.3 | 1.0 |
| mismatch | 0.8 | 1.5 | 0.35 | 1.0 |
| mismatch | 0.8 | 1.5 | 0.4 | 1.0 |
| mismatch | 0.8 | 1.5 | 0.45 | 1.0 |
| mismatch | 0.8 | 1.5 | 0.5 | 1.0 |
