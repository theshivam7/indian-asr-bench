# Whisper Medium — Pretrained vs Fine-tuned

Fine-tuned on the AESRC2020 **Indian** train split (`pengyizhou/accented_english`,
accent == INDIAN: 12,820 raw clips, 17.5h), best checkpoint selected on the
**validation** split, evaluated on the **test** split (1,731 clips) — the same test
set used for every pretrained model on this dataset.

**Headline comparison** is against `medium_hf` — the *pretrained* Whisper Medium run
through the **same** HuggingFace chunked pipeline as the fine-tuned model. This isolates
the fine-tuning gain from any decoding/engine differences. The original `openai-whisper`
number is shown as a secondary reference.

## Corpus WER (%) by evaluation mode

| Mode | Pretrained (HF) | Fine-tuned | Δ abs | Δ rel | _openai-whisper ref_ |
|------|:---------------:|:----------:|:-----:|:-----:|:--------------------:|
| `transcript_raw` | 5.92% | 4.37% | −1.55 pp | −26.2% | 6.05% |
| `transcript_clean` **(gold)** | 5.63% | 4.48% | −1.15 pp | −20.5% | 5.73% |
| `whisper_norm` | 5.26% | 4.18% | −1.08 pp | −20.5% | 5.41% |

> **Headline (transcript_clean)**: fine-tuning improves WER 5.63% → 4.48%  (−1.15 pp, −20.5% relative).

## By Audio Duration (`transcript_clean`)

| Duration | Pretrained (HF) | Fine-tuned | Δ abs |
|----------|:---------------:|:----------:|:-----:|
| 0-5s | 6.20% | 4.52% | −1.68 pp |
| 5-15s | 4.73% | 4.41% | −0.32 pp |

## Per-sample paired analysis (`transcript_clean`)

- Samples compared: **1731**
- Improved by fine-tuning: **276** (15.9%)
- Regressed: **156** (9.0%)
- Unchanged: **1299** (75.0%)

### Biggest improvements (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| AESRC2020-INDIAN-ACCENT-G13915-G13915S5399 | 110.0% | 0.0% | −110.0 pp |
| AESRC2020-INDIAN-ACCENT-G33434-G33434S2355 | 87.5% | 12.5% | −75.0 pp |
| AESRC2020-INDIAN-ACCENT-G02684-G02684S2355 | 66.7% | 0.0% | −66.7 pp |
| AESRC2020-INDIAN-ACCENT-G03884-G03884S3378 | 50.0% | 0.0% | −50.0 pp |
| AESRC2020-INDIAN-ACCENT-G13667-G13667S2366 | 50.0% | 0.0% | −50.0 pp |
| AESRC2020-INDIAN-ACCENT-G13915-G13915S5400 | 50.0% | 0.0% | −50.0 pp |
| AESRC2020-INDIAN-ACCENT-G02725-G02725S2354 | 45.5% | 0.0% | −45.5 pp |
| AESRC2020-INDIAN-ACCENT-G12816-G12816S2355 | 42.9% | 0.0% | −42.9 pp |
| AESRC2020-INDIAN-ACCENT-G12416-G12416S2355 | 50.0% | 10.0% | −40.0 pp |
| AESRC2020-INDIAN-ACCENT-G03161-G03161S2365 | 37.5% | 0.0% | −37.5 pp |

### Biggest regressions (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| AESRC2020-INDIAN-ACCENT-G13083-G13083S2364 | 150.0% | 200.0% | +50.0 pp |
| AESRC2020-INDIAN-ACCENT-G22609-G22609S2354 | 0.0% | 50.0% | +50.0 pp |
| AESRC2020-INDIAN-ACCENT-G64015-G64015S2373 | 0.0% | 50.0% | +50.0 pp |
| AESRC2020-INDIAN-ACCENT-G03838-G03838S2373 | 0.0% | 42.9% | +42.9 pp |
| AESRC2020-INDIAN-ACCENT-G00942-G00942S3370 | 0.0% | 40.0% | +40.0 pp |
| AESRC2020-INDIAN-ACCENT-G13121-G13121S1300 | 0.0% | 40.0% | +40.0 pp |
| AESRC2020-INDIAN-ACCENT-G01960-G01960S2365 | 0.0% | 37.5% | +37.5 pp |
| AESRC2020-INDIAN-ACCENT-G13176-G13176S2365 | 33.3% | 66.7% | +33.3 pp |
| AESRC2020-INDIAN-ACCENT-G23324-G23324S2354 | 33.3% | 66.7% | +33.3 pp |
| AESRC2020-INDIAN-ACCENT-G01964-G01964S2365 | 0.0% | 28.6% | +28.6 pp |

## Caveats

- **Engine**: the headline compares fine-tuned vs *pretrained-via-HF* (`medium_hf`), both decoded
  with the same chunked `transformers` pipeline, so the engine is held constant. The original
  `openai-whisper` number is shown only as a continuity reference.
- **Speaker structure**: the test split's 481 speakers are fully disjoint from the 38
  train/validation speakers, so the delta measures genuine speaker generalization.
  The validation split shares train's speaker set, so checkpoint selection measures
  fit only (see docs/AESRC2020_INDIAN_ANALYSIS.md).
