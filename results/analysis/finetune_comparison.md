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

> **Headline (transcript_clean)**: fine-tuning does NOT improve WER 14.42% → 14.61%  (+0.20 pp, +1.4% relative).

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
