# Whisper Small: Pretrained vs Fine-tuned

Fine-tuned on the AESRC2020 **Indian** train split (`pengyizhou/accented_english`,
accent == INDIAN: 12,820 raw clips, 17.5h), best checkpoint selected on the
**validation** split, evaluated on the **test** split (1,731 clips), the same test
set used for every pretrained model on this dataset.

**Headline comparison** is against `small_hf`, the *pretrained* Whisper Small run
through the **same** HuggingFace chunked pipeline as the fine-tuned model. This isolates
the fine-tuning gain from any decoding/engine differences. The original `openai-whisper`
number is shown as a secondary reference.

## Corpus WER (%) by evaluation mode

| Mode | Pretrained (HF) | Fine-tuned | Δ abs | Δ rel | _openai-whisper ref_ |
|------|:---------------:|:----------:|:-----:|:-----:|:--------------------:|
| `transcript_raw` | 7.49% | 5.55% | −1.94 pp | −25.9% | 7.52% |
| `transcript_clean` **(gold)** | 7.22% | 5.64% | −1.58 pp | −21.9% | 7.23% |
| `whisper_norm` | 6.91% | 5.36% | −1.55 pp | −22.4% | 6.96% |

> **Headline (transcript_clean)**: fine-tuning improves WER 7.22% → 5.64%  (−1.58 pp, −21.9% relative).

## By Audio Duration (`transcript_clean`)

| Duration | Pretrained (HF) | Fine-tuned | Δ abs |
|----------|:---------------:|:----------:|:-----:|
| 0-5s | 8.13% | 5.80% | −2.34 pp |
| 5-15s | 5.79% | 5.39% | −0.40 pp |

## Per-sample paired analysis (`transcript_clean`)

- Samples compared: **1731**
- Improved by fine-tuning: **351** (20.3%)
- Regressed: **195** (11.3%)
- Unchanged: **1185** (68.5%)

### Biggest improvements (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| AESRC2020-INDIAN-ACCENT-G22849-G22849S2354 | 90.0% | 0.0% | −90.0 pp |
| AESRC2020-INDIAN-ACCENT-G13842-G13842S2373 | 100.0% | 16.7% | −83.3 pp |
| AESRC2020-INDIAN-ACCENT-G02287-G02287S3376 | 100.0% | 33.3% | −66.7 pp |
| AESRC2020-INDIAN-ACCENT-G43652-G43652S2366 | 66.7% | 0.0% | −66.7 pp |
| AESRC2020-INDIAN-ACCENT-G33140-G33140S2365 | 57.1% | 0.0% | −57.1 pp |
| AESRC2020-INDIAN-ACCENT-G02320-G02320S2355 | 50.0% | 0.0% | −50.0 pp |
| AESRC2020-INDIAN-ACCENT-G02502-G02502S2355 | 50.0% | 0.0% | −50.0 pp |
| AESRC2020-INDIAN-ACCENT-G03161-G03161S2365 | 50.0% | 0.0% | −50.0 pp |
| AESRC2020-INDIAN-ACCENT-G03556-G03556S2365 | 50.0% | 0.0% | −50.0 pp |
| AESRC2020-INDIAN-ACCENT-G03884-G03884S3378 | 50.0% | 0.0% | −50.0 pp |

### Biggest regressions (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| AESRC2020-INDIAN-ACCENT-G12238-G12238S2366 | 16.7% | 66.7% | +50.0 pp |
| AESRC2020-INDIAN-ACCENT-G13083-G13083S2364 | 150.0% | 200.0% | +50.0 pp |
| AESRC2020-INDIAN-ACCENT-G03838-G03838S2373 | 0.0% | 42.9% | +42.9 pp |
| AESRC2020-INDIAN-ACCENT-G13963-G13963S2372 | 0.0% | 42.9% | +42.9 pp |
| AESRC2020-INDIAN-ACCENT-G01442-G01442S2354 | 0.0% | 40.0% | +40.0 pp |
| AESRC2020-INDIAN-ACCENT-G03247-G03247S2366 | 40.0% | 80.0% | +40.0 pp |
| AESRC2020-INDIAN-ACCENT-G12093-G12093S2365 | 0.0% | 40.0% | +40.0 pp |
| AESRC2020-INDIAN-ACCENT-G03370-G03370S2354 | 25.0% | 62.5% | +37.5 pp |
| AESRC2020-INDIAN-ACCENT-G11935-G11935S2364 | 0.0% | 37.5% | +37.5 pp |
| AESRC2020-INDIAN-ACCENT-G43810-G43810S2372 | 0.0% | 33.3% | +33.3 pp |

## Caveats

- **Engine**: the headline compares fine-tuned vs *pretrained-via-HF* (`small_hf`), both decoded
  with the same chunked `transformers` pipeline, so the engine is held constant. The original
  `openai-whisper` number is shown only as a continuity reference.
- **Speaker structure**: the test split's 481 speakers are fully disjoint from the 38
  train/validation speakers, so the delta measures genuine speaker generalization.
  The validation split shares train's speaker set, so checkpoint selection measures
  fit only (see `results/aesrc/analysis/speaker_overlap.md`).
