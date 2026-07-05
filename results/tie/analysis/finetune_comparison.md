# Whisper Medium — Pretrained vs Fine-tuned

Fine-tuned on the `raianand/TIE_shorts` **train** split (7884 clips, clips >30s filtered),
best checkpoint selected on the **validation** split, evaluated on the **test** split (986 clips) —
the same test set used for every pretrained model in this benchmark.

**Headline comparison** is against `medium_hf` — the *pretrained* Whisper Medium run through the
**same** HuggingFace chunked pipeline as the fine-tuned model. This isolates the fine-tuning gain
from any decoding/engine differences. The original `openai-whisper` number is shown as a secondary
reference.

## Corpus WER (%) by evaluation mode

| Mode | Pretrained (HF) | Fine-tuned | Δ abs | Δ rel | _openai-whisper ref_ |
|------|:---------------:|:----------:|:-----:|:-----:|:--------------------:|
| `transcript_raw` | 14.75% | 14.71% | −0.04 pp | −0.3% | 15.11% |
| `transcript_clean` **(gold)** | 14.42% | 14.61% | +0.20 pp | +1.4% | 14.76% |
| `hf_raw` | 17.72% | 17.70% | −0.02 pp | −0.1% | 18.01% |
| `hf_clean` | 15.51% | 15.70% | +0.19 pp | +1.3% | 15.76% |
| `whisper_norm` | 14.23% | 14.31% | +0.08 pp | +0.6% | 14.48% |

> **Headline (transcript_clean)**: fine-tuning does NOT improve WER 14.42% → 14.61%  (+0.20 pp, +1.4% relative).

## Speaker-disjoint re-split fine-tune (multi-seed)

Same recipe as the headline fine-tune, but every train clip whose speaker also appears in `test` is removed first (see `speaker_overlap.md`). Evaluated on the SAME test set as `medium_hf`, so any gain here cannot come from speaker adaptation. Run with multiple training seeds: a null claim from one seed would be indistinguishable from seed variance.

> **Training-set confound (disclosed)**: TIE_shorts' official test speakers are so entangled with train that removing them keeps only **567/7200 train clips (3.8/46.9 h, 51/331 speakers)**. The disjoint runs therefore differ from the headline fine-tune in BOTH speaker overlap and training-set size (~13x smaller) — this dataset cannot support a size-matched speaker-disjoint split at all, which is itself an evaluation-validity finding. Any WER regression below must not be attributed to speaker-disjointness alone; see the size-matched control section below for the separation.

| Seed | WER (`transcript_clean`) | Δ vs pretrained (paired, speaker-resampled) | 95% CI | p | p (Holm) |
|------|:----:|:----:|:----:|:----:|:----:|
| 42 | 16.17% | +1.75 pp | [+0.13, +4.17] * | 0.016 | 0.048 |
| 43 | 14.80% | +0.38 pp | [-0.01, +0.74] | 0.058 | 0.116 |
| 44 | 15.20% | +0.79 pp | [-0.18, +2.25] | 0.163 | 0.163 |

_\* = uncorrected 95% CI excludes 0. Use the Holm-adjusted p (multiplicity-corrected across these 3 seeds) for significance calls._

Across 3 seeds: WER 15.39% (range 14.80–16.17%), mean Δ vs pretrained +0.97 pp; seed-to-seed spread 1.37 pp.

> **Mixed result, not a clean null**: 1/3 seed(s) show a Holm-corrected significant WORSENING relative to pretrained (fine-tuning increases WER), while the remaining seed(s) fall within the ≈1.20 pp minimum detectable effect. The seed-to-seed spread (1.37 pp) is itself larger than the per-seed effect being estimated, so a single-seed run — including the checkpoint published as the 'primary' disjoint model — is not representative of the study as a whole. The safe claim is: fine-tuning on the speaker-disjoint training subset (567 clips) shows no evidence of improving WER over pretrained, and at least one seed shows evidence of making it worse. Whether the worsening is caused by the disjointness or by the 13x-smaller training set is separated by the size-matched control below.

## Size-matched control (speaker-overlapping, multi-seed)

Same recipe and clip count as the disjoint runs (567 train clips), but sampled at random from the FULL train split — speaker overlap with test is preserved. If these runs regress like the disjoint runs, the disjoint regression is a small-training-set effect; if they hold up, the disjointness itself is implicated.

| Seed | WER (`transcript_clean`) | Δ vs pretrained (paired, speaker-resampled) | 95% CI | p | p (Holm) |
|------|:----:|:----:|:----:|:----:|:----:|
| 42 | 14.33% | -0.09 pp | [-0.45, +0.23] | 0.581 | 1.000 |
| 43 | 14.40% | -0.02 pp | [-1.78, +1.53] | 0.997 | 1.000 |
| 44 | 14.40% | -0.02 pp | [-1.85, +1.90] | 0.985 | 1.000 |

> **Confound resolved**: all 3 size-matched seeds are statistically indistinguishable from pretrained (0/3 Holm-significant), while 1/3 disjoint seed(s) regressed significantly. Since both conditions train on the identical 567-clip count, training-set size alone cannot explain the disjoint regression — **speaker-disjointness is the cause**, not the smaller training set.

## By Region (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| EAST | 14.08% | 14.66% | +0.57 pp | 352 |
| NORTH | 13.92% | 13.31% | −0.60 pp | 202 |
| SOUTH | 14.08% | 13.66% | −0.42 pp | 362 |
| WEST | 18.84% | 22.53% | +3.69 pp | 69 |

## By Speech rate (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| AVG | 14.64% | 15.75% | +1.11 pp | 199 |
| FAST | 11.91% | 12.28% | +0.37 pp | 413 |
| SLOW | 17.69% | 17.10% | −0.59 pp | 373 |

## By Gender (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| F | 19.08% | 24.30% | +5.23 pp | 58 |
| M | 14.14% | 14.03% | −0.11 pp | 927 |

## By Discipline (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| Engineering | 14.39% | 14.46% | +0.07 pp | 691 |
| Non-Engineering | 14.48% | 14.97% | +0.49 pp | 294 |

## By Audio Duration (`transcript_clean`)

| Duration | Pretrained (HF) | Fine-tuned | Δ abs |
|----------|:---------------:|:----------:|:-----:|
| 0-5s | 25.00% | 35.00% | +10.00 pp |
| 5-15s | 21.01% | 20.92% | −0.09 pp |
| 15-30s | 12.20% | 12.29% | +0.09 pp |
| 30-60s | 25.41% | 25.69% | +0.28 pp |
| 60s+ | 119.27% | 133.03% | +13.76 pp |

## Per-sample paired analysis (`transcript_clean`)

- Samples compared: **985**
- Improved by fine-tuning: **250** (25.4%)
- Regressed: **307** (31.2%)
- Unchanged: **428** (43.5%)

### Biggest improvements (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| ZvsSe5sJGdc | 71.4% | 14.3% | −57.1 pp |
| z04lmkvw2wU | 68.2% | 12.7% | −55.5 pp |
| sdx6E2w9Td0 | 47.1% | 11.8% | −35.3 pp |
| 6aec4K8H9lE | 37.8% | 5.4% | −32.4 pp |
| vlImg6wCr8M | 82.5% | 50.9% | −31.6 pp |
| rbE6EuQLZbY | 54.8% | 25.8% | −29.0 pp |
| vggipesVGRU | 40.6% | 13.0% | −27.5 pp |
| 7IX-mDdkNKU | 31.8% | 4.5% | −27.3 pp |
| UdwIYN6xHeY | 59.3% | 34.9% | −24.4 pp |
| K5Bd9i0p7uQ | 47.4% | 23.7% | −23.7 pp |

### Biggest regressions (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| RbuSM2lRW_o | 288.0% | 455.4% | +167.4 pp |
| VVTFxiwiWB8 | 462.1% | 619.0% | +156.9 pp |
| vlhnD_C2zWY | 25.0% | 75.0% | +50.0 pp |
| aGvMubj8NXk | 18.8% | 43.8% | +25.0 pp |
| 7OWHcamBI3Q | 9.2% | 32.3% | +23.1 pp |
| wGFFxY1gxSU | 36.6% | 59.2% | +22.5 pp |
| SRH6EOBTy00 | 6.2% | 28.1% | +21.9 pp |
| vf0S_1ZITuA | 105.0% | 125.0% | +20.0 pp |
| G0rbpTX_ytE | 26.1% | 45.6% | +19.6 pp |
| YWACaAGz0Y8 | 34.8% | 52.2% | +17.4 pp |

## Caveats

- **Engine**: the headline compares fine-tuned vs *pretrained-via-HF* (`medium_hf`), both decoded
  with the same chunked `transformers` pipeline, so the engine is held constant. The original
  `openai-whisper` number is shown only as a continuity reference.
- **Speaker overlap**: see `speaker_overlap.md`. If test speakers also appear in train, part of the
  gain reflects speaker adaptation (disclosed, per the dataset's official splits).
- **Disjoint train-set size**: the speaker-disjoint runs train on 567 clips (3.8 h) vs the official
  split's 7200 (46.9 h, after the same duration/text filters) — speaker-disjointness and training-set
  size are confounded on this dataset by construction. The size-matched control isolates the size effect.
