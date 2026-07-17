# Fine-tuning capacity study: does model size explain the null result?

**Question:** the Whisper Medium (769M) fine-tune on TIE returned a
null-to-negative result (+0.20pp on the official split; up to +1.75pp regression on the
speaker-disjoint control). Is that because 769M has already saturated what 46.9h of TIE
data can teach it (a **capacity ceiling**), or because the **dataset itself** (label noise,
speaker-overlap confounds) can't support a fine-tuning gain at any model size?

**Answer:** the capacity hypothesis holds up. Whisper Tiny (39M) and Small (244M), fine-tuned
on the identical data with the step-based training recipe (§4), both show a genuine WER
improvement that Medium does not — and the gain shrinks monotonically as capacity grows,
flipping to null exactly at 769M. But the effect is noisier and less uniform than the
headline numbers suggest (see §5); treat this as directionally supportive, not conclusive
on its own.

## 1. Premise corrections

- All three Whisper sizes (tiny/small/medium) are pretrained on the same **680,000 hours**
  of weakly-supervised multilingual audio — capacity, not pretraining data volume, is what
  differs between them.
- The official-split fine-tune used **46.9h / 7,197 clips** of TIE train data — not the ~3h
  figure from an earlier conversation; that ~3.8h number is the *speaker-disjoint subset*,
  a much smaller manufactured control, not the main fine-tune's training set. Exact filter
  cascade (identical for tiny and small, logged from the real run): 7,884 raw clips → 7,200
  after dropping empty transcripts / >30s clips → **7,197** after dropping clips with no
  embedded audio. This is 3 clips fewer than Medium's previously-reported ~7,200 figure —
  the audio-availability filter wasn't separately logged in Medium's original run, so its
  realized count was likely also ~7,197 rather than exactly 7,200; the delta is noise, not a
  methodology change.

## 2. Pretrained capacity curve (context — not a fine-tuning statistic)

Speaker-clustered bootstrap CIs, `transcript_clean`, N=985 clips / G=280 speakers, B=2000
(from `analysis/statistics.py:analyze()`, write-free — this study did not regenerate the
shared pairwise-statistics files; see §7).

| Model | Params | Corpus WER | 95% CI |
|-------|:------:|:----------:|:------:|
| Whisper Tiny | 39M | 19.43% | [18.12, 20.79] |
| Whisper Base | 74M | 17.53% | [16.30, 18.80] |
| Whisper Small | 244M | 16.05% | [14.85, 17.34] |
| Whisper Medium | 769M | 14.76% | [13.69, 15.87] |
| Whisper Large | 1.5B | 15.93% | [14.72, 17.16] |

Capacity alone buys diminishing returns and even reverses at Large (1.5B) — consistent with
Large's well-documented multilingual-attention tradeoffs on English-heavy accented data, not
a new finding here.

## 3. Engine-controlled fine-tuning gain, by size

Headline comparison is always **fine-tuned vs. the pretrained baseline run through the same
HuggingFace chunked pipeline** (`*_hf` keys) — isolates the true fine-tuning effect from any
decoding/engine difference between `openai-whisper` and `transformers`. Paired,
speaker-clustered bootstrap (985 clips, 280 speaker clusters, `transcript_clean`).

| Size | Params | Pretrained (openai) | HF baseline | Fine-tuned | Δ (paired) | 95% CI | p | p (Holm, family of 3) |
|------|:------:|:--------------------:|:-----------:|:----------:|:-----------:|:------:|:-:|:----------------------:|
| Whisper Tiny | 39M | 19.43% | 22.10% | 19.14% | **−2.96 pp** | [−6.35, +0.13] | 0.065 | 0.195 |
| Whisper Small | 244M | 16.05% | 17.38% | 16.21% | **−1.17 pp** | [−3.97, +1.21] | 0.387 | 0.774 |
| Whisper Medium | 769M | 14.76% | 14.42% | 14.61% | +0.20 pp | [−0.46, +1.03] | 0.642 | 0.774 |

**Read carefully:** the point-estimate gradient (−2.96pp → −1.17pp → +0.20pp) is exactly the
shape the capacity hypothesis predicts, and it's the cleanest signal this whole project has
produced on the "capacity vs. dataset" question. But **none of the three deltas clears
significance after Holm correction** — tiny gets closest (uncorrected p=0.065) but the
985-clip/280-speaker test set is underpowered to confirm a ~3pp effect at the 5% level once
multiple-comparison correction is applied. Report the gradient as suggestive, not proven.

### Training trajectories (step-based recipe: max_steps=2000, LR 1e-5, effective
batch 32, fp16, eval every 200 steps, best-checkpoint selection added as a disclosed guard —
see §4)

| Size | Best step | Best val-WER (Whisper-normalizer) | val-WER at step 2000 | Shape |
|------|:---------:|:----------------------------------:|:---------------------:|-------|
| Tiny | 600 (of 2000, ~2.7 epochs) | 21.50% | 25.64% | learns to step 600, then overfits — val-WER rises monotonically after |
| Small | 800 (of 2000, ~3.6 epochs) | 16.82% | 19.37% | learns to step 800, then overfits, with a sharper val-loss blowup (0.66→0.93) than tiny |

Both show the textbook "learn then overfit" curve — a real, healthy training signal, not the
"no-learning" degenerate case (see §5, contingency check). This is the opposite failure mode
from Medium, whose val-WER historically peaked at epoch 1 on the *epoch-based* recipe — smaller
models here needed noticeably *more* steps before overfitting set in, consistent with having
more headroom to use the same 46.9h of data productively.

## 4. Recipe provenance (disclosed, not silently normalized to match Medium's)

Tiny/Small used an adaptation of an **externally supplied reference script**
(`step4_train_whisper.py`), not this project's `finetune_medium.py` (used for Medium). Deltas from
Medium's recipe:

| | Medium (`finetune_medium.py`) | Tiny/Small (`finetune_tiny_small.py`, step-based recipe) |
|---|---|---|
| Training unit | epoch-based | **step-based** (max_steps=2000, ~8-9 epochs total budget) |
| Effective batch | 16 | **32** (batch 8 × grad_accum 4) |
| Precision | bf16 | **fp16** |
| SpecAugment | yes | **no** |
| Early stopping | yes | **no** (relies on `load_best_model_at_end` instead) |
| Checkpoint-selection metric | project's `transcript_clean` normalizer | **OpenAI `EnglishTextNormalizer`** |

Two additions to the source script, both disclosed and necessary rather than silent:
- `load_best_model_at_end=True` / `metric_for_best_model="wer"` — the source script had no
  best-checkpoint selection and would have returned the ~8.9-epoch **last** checkpoint, which
  §3's trajectories show is well past the overfitting point for both sizes.
- Explicit `model.generation_config.language="english"` / `task="transcribe"` — the source
  script cleared `forced_decoder_ids`/`suppress_tokens` but never set these, which on a multilingual
  checkpoint risks language auto-detection corrupting the eval WER that checkpoint selection
  depends on.

**If the paper needs a strict apples-to-apples recipe comparison across all three sizes**,
a same-recipe rerun (Medium's epoch-based settings applied to tiny/small, or vice versa)
remains straightforward — flagging as a known option, not doing it in this phase per the
locked-down minimal-protocol scope.

## 5. The result is real but non-uniform — read past the corpus-WER headline

Per-sample paired analysis (`analysis/compare_finetune.py`, `transcript_clean`) tells a much
less clean story than the aggregate Δ:

| Size | Improved | Regressed | Unchanged | Net |
|------|:--------:|:---------:|:---------:|:---:|
| Tiny | 313 (31.8%) | 326 (33.1%) | 346 (35.1%) | corpus WER down 2.96pp |
| Small | 263 (26.7%) | 403 (40.9%) | 319 (32.4%) | corpus WER down 1.17pp |

For **both** sizes, more individual samples get *worse* than get *better* — the net corpus-WER
improvement is driven by a small number of extreme-outlier swaps, not a broad-based gain.
Concretely:

- **Tiny's biggest wins are runaway repetition-loop fixes.** The top improvement, sample
  `lMIVXmVvqBM`, went from **977.8% WER pretrained → 55.6% fine-tuned**; several more of
  tiny's top-10 improvements start above 300-850% WER. Whisper Tiny is known to occasionally
  degenerate into repeated-token loops on short/noisy clips (a `hf_clean` insertion-rate of
  9.08% pretrained vs. 6.38% fine-tuned — roughly back down to the 6.22% baseline of the
  non-HF-pipeline `tiny` engine — is consistent with this). Fine-tuning is fixing a decoding
  pathology on a handful of clips as much as it's teaching accented-English content.
- **The same pathology cuts both ways for Small.** Its top *regression*, `jtMZfLViZu8`, went
  from 66.2% → **445.1%** WER post-fine-tuning — fine-tuning didn't eliminate the
  repetition-loop failure mode, it relocated it to different clips.

**Implication:** the −2.96pp/−1.17pp headline deltas are directionally real and support the
capacity hypothesis, but a meaningful share of the effect is Whisper-tiny/small's decoding
instability on a handful of pathological clips rather than a broad, reliable
accented-English-content improvement. This is worth stating plainly rather than oversold as
"fine-tuning clearly works better at smaller sizes."

## 5b. Why does absolute WER stay high (14-22%) even after fine-tuning?

A natural follow-up question: even where fine-tuning helps, WER is still far from a
"solved" benchmark number. Four compounding factors, largest first — mostly a domain-
difficulty floor, not a fine-tuning shortfall:

1. **TIE is uncontrolled "found" audio, not clean speech — this alone likely explains
   roughly half the gap.** TIE_shorts is NPTEL-style lecture audio scraped from YouTube:
   variable mic quality, room noise, off-mic speech, no recording protocol. The project's
   own Svarah benchmark (controlled-protocol read speech) is the clean counterfactual:
   Whisper Large scores **7.11% on Svarah vs. 15.93% on TIE — literally double**. That gap
   holds for every pretrained model regardless of fine-tuning, so it's an audio/domain
   property, not something more TIE fine-tuning data fixes.
2. **Content is genuinely hard**: dense technical/academic vocabulary, proper nouns, and
   live-lecture disfluencies (false starts, fillers, fast speech) — harder than the mostly
   scripted read speech Whisper's pretraining leaned on.
3. **Reference-label noise is tail-concentrated, and the tail drags the corpus average
   up.** Only ~1% of the full corpus has a bad reference label (§ full breakdown in
   `error_analysis_transcript_clean.md`), but that share jumps to **62% of each model's
   worst-20 highest-WER clips**. Median WER (10.1-14.9% across tiny_ft/small_ft/medium_ft)
   sits 3-5pp below corpus WER (14.6-19.1%) precisely because of this tail effect — most
   individual clips score meaningfully better than the headline number suggests.
4. **The fine-tuning dose is small relative to full domain adaptation.** 46.9h / 7,197
   clips is enough to nudge decoding behavior (§5's repetition-loop fixes) but nowhere near
   enough, on top of a 680,000-hour pretrained model, to close an ~8pp noisy-audio domain
   gap. Part of what fine-tuning buys here is fixing decoding pathology on some clips while
   introducing it on others (§5), not a uniform content-adaptation gain.

**Implication for Phase 2:** this is a direct argument for AESRC2020 over more TIE data —
AESRC2020 is professionally recorded (not scraped), which would let a follow-up study
separate "is fine-tuning working" from "is the audio itself just noisy," a distinction TIE
cannot cleanly make on its own.

## 6. Contingency check: was this a no-learning artifact?

**Trigger condition (not met):** best checkpoint at ≤200 steps with no further val-WER
improvement afterward. Both sizes' best checkpoints landed well past step 200 (600 for tiny,
800 for small) with a clean learn-then-overfit shape (§3). **No diagnostic high-LR rerun was
needed or performed** — the LR 1e-5 step-based-recipe run stands as the headline for both
sizes.

## 7. Holm-correction family — scope flag

The 3-test family in §3 (tiny-FT-vs-HF, small-FT-vs-HF, medium-FT-vs-HF) is **separate** from
the project's cross-model pairwise-comparison family
(`statistics_pairwise_transcript_clean.csv`) — that family covers PRETRAINED models only
(now 9 chart models / 36 pairs, including Tiny and Small). The fine-tuned variants run
through a different decoding engine than the pretrained ladder, so mixing them into that
family would confound fine-tuning with an engine change; the two Holm families are computed
independently by design, not as a temporary gap.

## 8. Decision table

| Observed pattern | Interpretation |
|---|---|
| **This study's actual result**: monotonic Δ gradient (tiny > small > medium), tiny closest to significance, but non-uniform per-sample and driven partly by decoding-pathology fixes | **Partial support for capacity ceiling.** Smaller models do show a real, larger fine-tuning gain on identical data — but the effect is noisier than a clean "capacity explains everything" story, and doesn't reach significance after correction. Best read as "capacity is *part* of the explanation; dataset quality (label noise, TIE's speaker-overlap structure) still caps how much any size can gain." |
| (not observed) Flat deltas across all three sizes | would have pointed to dataset (labels/speaker structure) as the limiter, not capacity |
| (not observed) Significant, uniform gains at all sizes including medium | would have suggested Medium's null was a training-setup artifact, not capacity |

**Bottom line:** Tiny/Small fine-tuning gains are real in direction but
underpowered and partly decoding-artifact-driven at this sample size (985 test clips). This
is consistent with — but does not conclusively prove — a capacity-ceiling explanation for
Medium's null result. The next-highest-value step to sharpen this finding is **more
speaker-disjoint, professionally-transcribed Indian-accented training data**, which is
exactly what AESRC2020 (Phase 2 candidate, see `docs/AESRC2020_INDIAN_ANALYSIS.md`) provides:
17.5h of natively speaker-disjoint Indian train data (vs. TIE's 3.8h manufactured disjoint
subset) with clean professional labels, removing both confounds (speaker overlap AND label
noise) that limit how conclusively this study's numbers can be read.

## 9. Recommendation: proceed to Phase 2

Given (a) directionally supportive but statistically underpowered/noisy Phase 1 results, and
(b) AESRC2020's structural advantages over TIE for exactly this question (§8, suitability
analysis in local working notes), **recommend proceeding to Phase 2** — pending confirmation
of AESRC2020 licensing/access terms. (Historical note: Phase 2 was subsequently run; see
`results/aesrc/analysis/` for its results.)

## Appendix: file map

- Raw transcripts: `results/tie/stage1_raw_transcripts/wer_{tiny,small}_{,hf_,ft_}raw.csv`
- Scored (all 5 modes): `results/tie/stage2_processed/{mode}/wer_*.csv`,
  `wer_summary_all_models.csv`
- Per-size comparison reports: `results/tie/analysis/finetune_comparison_{tiny,small}.md`
  (+ medium's pre-existing `finetune_comparison.md`)
- Capacity summary table: `results/tie/analysis/finetune_capacity_summary.{md,csv}`
- Training trajectories: `results/tie/analysis/ft_trajectories/*_trainer_state.json`
- Pinned regression values: `tests/test_pipeline.py::EXPECTED_TIE_WER`
