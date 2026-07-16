# Whisper Tiny — Pretrained vs Fine-tuned

Fine-tuned on the AESRC2020 **Indian** train split (`pengyizhou/accented_english`,
accent == INDIAN: 12,820 raw clips, 17.5h), best checkpoint selected on the
**validation** split, evaluated on the **test** split (1,731 clips) — the same test
set used for every pretrained model on this dataset.

**Headline comparison** is against `tiny_hf` — the *pretrained* Whisper Tiny run
through the **same** HuggingFace chunked pipeline as the fine-tuned model. This isolates
the fine-tuning gain from any decoding/engine differences. The original `openai-whisper`
number is shown as a secondary reference.

## Corpus WER (%) by evaluation mode

| Mode | Pretrained (HF) | Fine-tuned | Δ abs | Δ rel | _openai-whisper ref_ |
|------|:---------------:|:----------:|:-----:|:-----:|:--------------------:|
| `transcript_raw` | 17.64% | 12.49% | −5.15 pp | −29.2% | 13.91% |
| `transcript_clean` **(gold)** | 17.45% | 12.64% | −4.81 pp | −27.6% | 13.66% |
| `whisper_norm` | 16.97% | 9.83% | −7.13 pp | −42.0% | 13.21% |

> **Headline (transcript_clean)**: fine-tuning improves WER 17.45% → 12.64%  (−4.81 pp, −27.6% relative).

## By Audio Duration (`transcript_clean`)

| Duration | Pretrained (HF) | Fine-tuned | Δ abs |
|----------|:---------------:|:----------:|:-----:|
| 0-5s | 19.22% | 14.74% | −4.48 pp |
| 5-15s | 14.66% | 9.34% | −5.32 pp |

## Per-sample paired analysis (`transcript_clean`)

- Samples compared: **1731**
- Improved by fine-tuning: **533** (30.8%)
- Regressed: **251** (14.5%)
- Unchanged: **947** (54.7%)

### Biggest improvements (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| AESRC2020-INDIAN-ACCENT-G12929-G12929S2350 | 3990.9% | 18.2% | −3972.7 pp |
| AESRC2020-INDIAN-ACCENT-G13842-G13842S1299 | 1453.3% | 26.7% | −1426.7 pp |
| AESRC2020-INDIAN-ACCENT-G33289-G33289S2353 | 116.7% | 0.0% | −116.7 pp |
| AESRC2020-INDIAN-ACCENT-G01737-G01737S2355 | 100.0% | 0.0% | −100.0 pp |
| AESRC2020-INDIAN-ACCENT-G03840-G03840S2373 | 100.0% | 0.0% | −100.0 pp |
| AESRC2020-INDIAN-ACCENT-G12737-G12737S2355 | 100.0% | 12.5% | −87.5 pp |
| AESRC2020-INDIAN-ACCENT-G73344-G73344S2355 | 85.7% | 0.0% | −85.7 pp |
| AESRC2020-INDIAN-ACCENT-G43065-G43065S3375 | 100.0% | 16.7% | −83.3 pp |
| AESRC2020-INDIAN-ACCENT-G03370-G03370S3370 | 80.0% | 0.0% | −80.0 pp |
| AESRC2020-INDIAN-ACCENT-G13236-G13236S2364 | 80.0% | 0.0% | −80.0 pp |

### Biggest regressions (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| AESRC2020-INDIAN-ACCENT-G32681-G32681S2355 | 77.8% | 4911.1% | +4833.3 pp |
| AESRC2020-INDIAN-ACCENT-G93639-G93639S2366 | 20.0% | 100.0% | +80.0 pp |
| AESRC2020-INDIAN-ACCENT-G02287-G02287S3376 | 66.7% | 133.3% | +66.7 pp |
| AESRC2020-INDIAN-ACCENT-G00847-G00847S3370 | 0.0% | 50.0% | +50.0 pp |
| AESRC2020-INDIAN-ACCENT-G01722-G01722S1300 | 8.3% | 58.3% | +50.0 pp |
| AESRC2020-INDIAN-ACCENT-G03809-G03809S2372 | 16.7% | 66.7% | +50.0 pp |
| AESRC2020-INDIAN-ACCENT-G13176-G13176S2365 | 16.7% | 66.7% | +50.0 pp |
| AESRC2020-INDIAN-ACCENT-G94051-G94051S2354 | 0.0% | 50.0% | +50.0 pp |
| AESRC2020-INDIAN-ACCENT-G13448-G13448S2355 | 0.0% | 42.9% | +42.9 pp |
| AESRC2020-INDIAN-ACCENT-G53472-G53472S2354 | 0.0% | 42.9% | +42.9 pp |

## Caveats

- **Engine**: the headline compares fine-tuned vs *pretrained-via-HF* (`tiny_hf`), both decoded
  with the same chunked `transformers` pipeline, so the engine is held constant. The original
  `openai-whisper` number is shown only as a continuity reference.
- **Speaker structure**: the test split's 481 speakers are fully disjoint from the 38
  train/validation speakers, so the delta measures genuine speaker generalization.
  The validation split shares train's speaker set, so checkpoint selection measures
  fit only (see docs/AESRC2020_INDIAN_ANALYSIS.md).
