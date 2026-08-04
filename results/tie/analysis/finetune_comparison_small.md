# Whisper Small: Pretrained vs Fine-tuned

Fine-tuned on the `raianand/TIE_shorts` **train** split (7,884 raw clips; ~7,200 remain
after dropping empty transcripts, clips >30s, and clips with no embedded audio, see the
run log for the exact realized count), best checkpoint selected on the **validation**
split, evaluated on the **test** split (986 clips), the same test set used for every
pretrained model in this benchmark.

**Headline comparison** is against `small_hf`, the *pretrained* Whisper Small run
through the **same** HuggingFace chunked pipeline as the fine-tuned model. This isolates
the fine-tuning gain from any decoding/engine differences. The original `openai-whisper`
number is shown as a secondary reference.

## Corpus WER (%) by evaluation mode

| Mode | Pretrained (HF) | Fine-tuned | Δ abs | Δ rel | _openai-whisper ref_ |
|------|:---------------:|:----------:|:-----:|:-----:|:--------------------:|
| `transcript_raw` | 17.78% | 16.39% | −1.39 pp | −7.8% | 16.44% |
| `transcript_clean` **(gold)** | 17.38% | 16.21% | −1.17 pp | −6.7% | 16.05% |
| `hf_raw` | 20.53% | 19.17% | −1.36 pp | −6.6% | 19.20% |
| `hf_clean` | 18.27% | 17.20% | −1.07 pp | −5.8% | 16.96% |
| `whisper_norm` | 16.93% | 15.64% | −1.29 pp | −7.6% | 15.80% |

> **Headline (transcript_clean)**: fine-tuning improves WER 17.38% → 16.21%  (−1.17 pp, −6.7% relative).

## By Region (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| EAST | 18.23% | 16.79% | −1.44 pp | 352 |
| NORTH | 14.80% | 17.48% | +2.68 pp | 202 |
| SOUTH | 15.95% | 15.13% | −0.82 pp | 362 |
| WEST | 27.53% | 15.57% | −11.96 pp | 69 |

## By Speech rate (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| AVG | 18.27% | 13.80% | −4.47 pp | 199 |
| FAST | 12.97% | 13.66% | +0.69 pp | 413 |
| SLOW | 22.84% | 21.13% | −1.71 pp | 373 |

## By Gender (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| F | 20.30% | 12.02% | −8.28 pp | 58 |
| M | 17.21% | 16.46% | −0.75 pp | 927 |

## By Discipline (`transcript_clean`)

| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |
|-------|:---------------:|:----------:|:-----:|:-------:|
| Engineering | 17.44% | 17.02% | −0.42 pp | 691 |
| Non-Engineering | 17.25% | 14.31% | −2.93 pp | 294 |

## By Audio Duration (`transcript_clean`)

| Duration | Pretrained (HF) | Fine-tuned | Δ abs |
|----------|:---------------:|:----------:|:-----:|
| 0-5s | 25.00% | 35.00% | +10.00 pp |
| 5-15s | 21.86% | 23.06% | +1.20 pp |
| 15-30s | 13.12% | 13.84% | +0.72 pp |
| 30-60s | 52.54% | 34.03% | −18.50 pp |
| 60s+ | 18.96% | 27.83% | +8.87 pp |

## Per-sample paired analysis (`transcript_clean`)

- Samples compared: **985**
- Improved by fine-tuning: **263** (26.7%)
- Regressed: **403** (40.9%)
- Unchanged: **319** (32.4%)

### Biggest improvements (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| V-kLaH4139o | 833.3% | 20.3% | −813.0 pp |
| rbE6EuQLZbY | 519.4% | 37.1% | −482.2 pp |
| JmyxDMTpQ4o | 409.2% | 5.3% | −404.0 pp |
| ifQZgKgZoOQ | 330.2% | 15.1% | −315.1 pp |
| RbuSM2lRW_o | 308.7% | 27.2% | −281.5 pp |
| 6CwSfoOR7-U | 80.0% | 13.3% | −66.7 pp |
| kPnr_57oii4 | 50.0% | 0.0% | −50.0 pp |
| vlImg6wCr8M | 77.2% | 35.1% | −42.1 pp |
| 6aec4K8H9lE | 44.6% | 12.2% | −32.4 pp |
| 3luBQ6LxMEE | 33.3% | 6.7% | −26.7 pp |

### Biggest regressions (top 10)

| ID | Pretrained WER | Fine-tuned WER | Δ |
|----|:--------------:|:--------------:|:-:|
| YOi4ONq4Zsk | 30.1% | 572.6% | +542.5 pp |
| jtMZfLViZu8 | 66.2% | 445.1% | +378.9 pp |
| eF3zaTryw6k | 7.8% | 81.2% | +73.4 pp |
| BETX-s42Df4 | 10.9% | 56.5% | +45.6 pp |
| HQSgTo0Er5c | 4.1% | 49.0% | +44.9 pp |
| QhRrY6GlnEE | 9.9% | 45.1% | +35.2 pp |
| 0FA1bzcwRvU | 26.8% | 53.6% | +26.8 pp |
| rNIYCv7W7EI | 2.0% | 27.6% | +25.5 pp |
| vlhnD_C2zWY | 25.0% | 50.0% | +25.0 pp |
| VPwq6SN4Zos | 48.6% | 71.4% | +22.9 pp |

## Caveats

- **Engine**: the headline compares fine-tuned vs *pretrained-via-HF* (`small_hf`), both decoded
  with the same chunked `transformers` pipeline, so the engine is held constant. The original
  `openai-whisper` number is shown only as a continuity reference.
- **Speaker overlap**: see `speaker_overlap.md`. If test speakers also appear in train, part of the
  gain reflects speaker adaptation (disclosed, per the dataset's official splits).
