# Whisper Tiny — Pretrained vs Fine-tuned

Fine-tuned on the `raianand/TIE_shorts` **train** split (7,884 raw clips; ~7,200 remain
after dropping empty transcripts, clips >30s, and clips with no embedded audio — see the
run log for the exact realized count), best checkpoint selected on the **validation**
split, evaluated on the **test** split (986 clips) — the same test set used for every
pretrained model in this benchmark.

**Headline comparison** is against `tiny_hf` — the *pretrained* Whisper Tiny run
through the **same** HuggingFace chunked pipeline as the fine-tuned model. This isolates
the fine-tuning gain from any decoding/engine differences. The original `openai-whisper`
number is shown as a secondary reference.

## Corpus WER (%) by evaluation mode

| Mode | Pretrained (HF) | Fine-tuned | Δ abs | Δ rel | _openai-whisper ref_ |
|------|:---------------:|:----------:|:-----:|:-----:|:--------------------:|
| `transcript_raw` | 22.49% | 19.54% | −2.96 pp | −13.1% | 19.79% |
| `transcript_clean` **(gold)** | 22.10% | 19.14% | −2.96 pp | −13.4% | 19.43% |
| `hf_raw` | 24.86% | 21.82% | −3.04 pp | −12.2% | 22.20% |
| `hf_clean` | 22.67% | 19.63% | −3.05 pp | −13.4% | 20.07% |
| `whisper_norm` | 21.72% | 18.45% | −3.28 pp | −15.1% | 19.01% |

> **Headline (transcript_clean)**: fine-tuning improves WER 22.10% → 19.14%  (−2.96 pp, −13.4% relative).

## By Region (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| EAST | 22.83% | 20.25% | −2.59 pp | 352 |
| NORTH | 24.07% | 18.65% | −5.42 pp | 202 |
| SOUTH | 21.28% | 18.73% | −2.55 pp | 362 |
| WEST | 17.76% | 17.69% | −0.07 pp | 69 |

## By Speech rate (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| AVG | 20.38% | 16.82% | −3.56 pp | 199 |
| FAST | 21.83% | 17.09% | −4.74 pp | 413 |
| SLOW | 23.52% | 23.34% | −0.18 pp | 373 |

## By Gender (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| F | 14.66% | 15.31% | +0.64 pp | 58 |
| M | 22.55% | 19.37% | −3.18 pp | 927 |

## By Discipline (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| Engineering | 21.35% | 18.82% | −2.53 pp | 691 |
| Non-Engineering | 23.87% | 19.88% | −3.98 pp | 294 |

## By Audio Duration (`transcript_clean`)

| Duration | Pretrained (HF) | Fine-tuned | Δ abs |
|----------|:---------------:|:----------:|:-----:|
| 0-5s | 30.00% | 60.00% | +30.00 pp |
| 5-15s | 27.24% | 27.75% | +0.51 pp |
| 15-30s | 19.91% | 17.46% | −2.46 pp |
| 30-60s | 38.52% | 29.88% | −8.64 pp |
| 60s+ | 39.14% | 44.34% | +5.20 pp |

## Per-sample paired analysis (`transcript_clean`)

- Samples compared: **985**
- Improved by fine-tuning: **313** (31.8%)
- Regressed: **326** (33.1%)
- Unchanged: **346** (35.1%)

### Biggest improvements (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| lMIVXmVvqBM | 977.8% | 55.6% | −922.2 pp |
| 7jwzkMvfbbU | 852.9% | 35.3% | −817.7 pp |
| tYaqEzhHolE | 659.1% | 33.3% | −625.8 pp |
| bkzKVsIEjxk | 491.1% | 80.0% | −411.1 pp |
| J7-9nhlJWXA | 313.0% | 29.0% | −284.1 pp |
| kPnr_57oii4 | 60.0% | 0.0% | −60.0 pp |
| QhRrY6GlnEE | 45.1% | 4.2% | −40.8 pp |
| Tr-KsYvetMQ | 50.6% | 15.3% | −35.3 pp |
| 4XulH3TfbT0 | 139.0% | 104.9% | −34.1 pp |
| l8kIWLWfbZ8 | 122.2% | 88.9% | −33.3 pp |

### Biggest regressions (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| aNOkZZjUKoM | 14.3% | 461.9% | +447.6 pp |
| -2aOCNaOiLs | 80.0% | 200.0% | +120.0 pp |
| 8PZx5kgLSqQ | 42.9% | 73.5% | +30.6 pp |
| 3K_BNO3gZwc | 0.0% | 25.0% | +25.0 pp |
| aJyQNyGXJcw | 2.2% | 26.7% | +24.4 pp |
| 1ElNjIBL7Ys | 17.6% | 41.2% | +23.5 pp |
| 6CwSfoOR7-U | 73.3% | 93.3% | +20.0 pp |
| 4yCE67VwYJA | 40.0% | 60.0% | +20.0 pp |
| tGjpFNTay90 | 11.8% | 29.4% | +17.6 pp |
| 9hxEPGhusAE | 8.2% | 24.6% | +16.4 pp |

## Caveats

- **Engine**: the headline compares fine-tuned vs *pretrained-via-HF* (`tiny_hf`), both decoded
  with the same chunked `transformers` pipeline, so the engine is held constant. The original
  `openai-whisper` number is shown only as a continuity reference.
- **Speaker overlap**: see `speaker_overlap.md`. If test speakers also appear in train, part of the
  gain reflects speaker adaptation (disclosed, per the dataset's official splits).
