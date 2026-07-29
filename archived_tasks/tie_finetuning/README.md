# Archived: TIE_shorts fine-tuning capacity study

This folder documents the TIE_shorts side of the Whisper Tiny/Small/Medium fine-tuning capacity
study. It is **archived, not part of the main benchmark documentation** (README.md, SUMMARY.md).
The AESRC2020 (Indian subset) side of the same study is still the main fine-tuning result and
stays in SUMMARY.md.

## Why archived

TIE's fine-tuned models did not reach a strong enough result to include in the paper. No size
(Tiny, Small, or Medium) showed a statistically significant gain over its pretrained baseline
after Holm correction, and the test set is speaker-matched with training (100% speaker overlap),
so any gain that did appear would be confounded with speaker adaptation rather than clean
evidence of accent or content learning. AESRC's test set is natively speaker-disjoint and shows
significant gains for Small and Medium, so it carries the fine-tuning story instead.

## Nothing was deleted

All code, checkpoints, and experiment outputs are exactly where they were:

- Training scripts: [`finetune/finetune_medium.py`](../../finetune/finetune_medium.py) (Medium,
  full fine-tune), [`finetune/finetune_tiny_small.py`](../../finetune/finetune_tiny_small.py)
  (Tiny/Small, step-based recipe).
- Speaker-overlap check: [`finetune/check_speaker_overlap.py`](../../finetune/check_speaker_overlap.py),
  full report at [`results/tie/analysis/speaker_overlap.md`](../../results/tie/analysis/speaker_overlap.md).
- Full per-size reports: [`findings_tiny_small_ft.md`](../../results/tie/analysis/findings_tiny_small_ft.md),
  [`finetune_comparison.md`](../../results/tie/analysis/finetune_comparison.md) (Medium),
  [`finetune_comparison_small.md`](../../results/tie/analysis/finetune_comparison_small.md),
  [`finetune_comparison_tiny.md`](../../results/tie/analysis/finetune_comparison_tiny.md).
- Figure: [`results/tie/analysis/finetune_comparison.png`](../../results/tie/analysis/finetune_comparison.png).
- Published checkpoints (still live on the Hugging Face Hub, unaffected by this archival):
  [whisper-tiny-indian-english](https://huggingface.co/theshivam7/whisper-tiny-indian-english),
  [whisper-small-indian-english](https://huggingface.co/theshivam7/whisper-small-indian-english),
  [whisper-medium-indian-english](https://huggingface.co/theshivam7/whisper-medium-indian-english).

## Results

**Setup:** Medium was fully fine-tuned via `transformers` `Seq2SeqTrainer` (bf16, epoch-based,
early stopping on validation WER). Tiny and Small used a step-based recipe (`max_steps=2000`,
effective batch 32, fp16, best checkpoint by validation WER), a disclosed recipe difference,
not a bug. Every comparison decodes fine-tuned and pretrained through the identical HF pipeline,
so fine-tuning is isolated from engine effects. Statistics use a paired speaker-clustered
bootstrap over 280 speakers, Holm-corrected across this 3-test family.

| Size | Params | Pretrained (HF) | Fine-tuned | Delta (paired) | 95% CI | p (Holm) |
|------|:------:|:---:|:---:|:---:|:---:|:---:|
| Whisper Tiny | 39M | 22.10% | 19.14% | -2.96 pp | [-6.35, +0.13] | 0.195 |
| Whisper Small | 244M | 17.38% | 16.21% | -1.17 pp | [-3.97, +1.21] | 0.774 |
| Whisper Medium | 769M | 14.42% | 14.61% | +0.20 pp | [-0.46, +1.03] | 0.774 |

A capacity gradient, but not a significant one. Point gains shrink monotonically as capacity
grows, exactly the shape a capacity ceiling predicts, but no delta survives Holm correction at
985 test clips.

Details behind the headline number:

- More clips got worse than better for both smaller sizes (Tiny: 313 improved vs 326 regressed;
  Small: 263 vs 403).
- The net gain mostly comes from fixing a few severe repetition loops. One Tiny clip fell from
  977.8% to 55.6% WER. Fine-tuning also introduced the same pathology elsewhere: one Small clip
  rose from 66.2% to 445.1%.
- Both runs show a healthy learn-then-overfit trajectory (best checkpoints at steps 600 and 800
  of 2000), which rules out a no-learning explanation.
- Absolute WER stays at 14 to 22% after fine-tuning because the domain is hard. Whisper Large-v3
  scores 15.93% on TIE vs 7.11% on Svarah with identical weights.
- 100% of test speakers, and 100% of test clips, come from speakers also seen in training. There
  is no clip-level leakage, but the comparison is speaker-matched, so part of any gain reflects
  speaker adaptation rather than accent or content learning.
- On 60s+ clips the HF chunked pipeline scores much higher WER than `openai-whisper` with
  identical weights. This hits pretrained and fine-tuned equally, so the head-to-head stays
  fair, but it inflates tail metrics for HF-pipeline runs.
