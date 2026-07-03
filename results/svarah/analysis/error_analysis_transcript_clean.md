# Codified error analysis — Svarah — mode `transcript_clean`

7 models, 6656 common clips. Classifier thresholds: clip_over_run = recall>=0.80 & ratio>=1.50; content_mismatch = recall<0.40; else unflagged. Consensus = per-clip mean of recall/ratio across all models.

## Full-corpus taxonomy (all clips)

**Artifact share over the full corpus: 4.4% (95% Wilson CI 4.0–5.0%)** of clips carry an artifact signature.

| category | n_clips | share_pct | share_ci_lo | share_ci_hi | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 33 | 0.5 | 0.4 | 0.7 | 0.95 | 1.9 | 0.982 |
| content_mismatch | 262 | 3.9 | 3.5 | 4.4 | 0.13 | 1.3 | 1.203 |
| unflagged | 6361 | 95.6 | 95.0 | 96.0 | 0.92 | 1.02 | 0.119 |

## Artifact-adjusted corpus WER

Corpus WER on all common clips vs. on the unflagged subset. `artifact_inflation_pp` is how many WER points the benchmark's own artifacts add to each model's score.

| model | display | wer_all_pct | wer_adjusted_pct | artifact_inflation_pp | n_clips_all | n_clips_adjusted |
| --- | --- | --- | --- | --- | --- | --- |
| base | Whisper Base | 14.53 | 13.52 | 1.01 | 6656 | 6361 |
| medium | Whisper Medium | 7.89 | 7.15 | 0.74 | 6656 | 6361 |
| large | Whisper Large | 7.11 | 6.39 | 0.72 | 6656 | 6361 |
| large_v3_turbo | Whisper large-v3-turbo | 8.1 | 7.36 | 0.74 | 6656 | 6361 |
| parakeet | Parakeet-TDT-0.6B | 11.73 | 10.99 | 0.73 | 6656 | 6361 |
| parakeet_ctc | Parakeet-CTC-1.1B | 15.65 | 14.84 | 0.8 | 6656 | 6361 |
| qwen3 | Qwen3-ASR-1.7B | 11.82 | 11.17 | 0.65 | 6656 | 6361 |

## Inter-hypothesis agreement

`inter_hyp_dist` = mean normalized word edit distance BETWEEN model hypotheses (all pairs); `cross_arch_dist` = same, restricted to (enc_dec|llm) x (ctc|transducer) pairs; `hyp_to_ref_wer` = mean WER against the reference. Hypotheses that agree with each other but not with the reference localize the fault in the reference — across architectures that share no decoder or training objective. `ref_wer_grounded` vs `ref_wer_free` on flagged clips is a contamination probe (free-decoding models matching a flawed reference better than acoustically-grounded ones would suggest caption memorization).

| category | n_clips | inter_hyp_dist | cross_arch_dist | hyp_to_ref_wer | ref_wer_grounded | ref_wer_free |
| --- | --- | --- | --- | --- | --- | --- |
| clip_over_run | 33 | 0.32 | 0.324 | 0.982 | 0.882 | 1.021 |
| content_mismatch | 262 | 0.892 | 0.922 | 1.203 | 1.196 | 1.205 |
| unflagged | 6361 | 0.14 | 0.156 | 0.119 | 0.15 | 0.107 |

## Worst-20 tail (continuity with the original hand analysis)

Top-20 highest-WER clips per model (140 rows -> 97 distinct). **Tail artifact share: 97.9%** (95% Wilson CI 92.8–99.4%).

| category | n_clips | share_pct | mean_recall | mean_ratio | mean_wer |
| --- | --- | --- | --- | --- | --- |
| clip_over_run | 6 | 6.2 | 0.97 | 6.99 | 6.027 |
| content_mismatch | 89 | 91.8 | 0.0 | 5.83 | 5.824 |
| genuine_error | 2 | 2.1 | 0.62 | 3.87 | 3.25 |

7 tail clips appear in the worst-20 of >=3 distinct architectures; across those the mean per-clip spread is recall std=0.08, length-ratio std=0.161.

## Threshold sensitivity

Full-corpus artifact share under alternative classifier thresholds (see threshold_sensitivity CSV for the full grid):

| vary | recall_overrun | ratio_overrun | recall_mismatch | artifact_share_pct |
| --- | --- | --- | --- | --- |
| mismatch | 0.8 | 1.5 | 0.3 | 4.2 |
| mismatch | 0.8 | 1.5 | 0.35 | 4.3 |
| mismatch | 0.8 | 1.5 | 0.4 | 4.4 |
| mismatch | 0.8 | 1.5 | 0.45 | 5.7 |
| mismatch | 0.8 | 1.5 | 0.5 | 5.8 |
