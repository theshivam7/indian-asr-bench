# WER Analysis Summary — ASR Benchmark on Indian English (TIE_shorts)

## Dataset

- **Source:** [raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts)
- **Split:** `test` (986 samples, ~52,178 reference words)
- **Domain:** Indian English NPTEL academic lectures
- **Distribution:** 928M / 58F | FAST 413, SLOW 373, AVG 199 | SOUTH 362, EAST 352, NORTH 202, WEST 69

---

## Main Results (transcript_clean — gold standard)

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 | Samples |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|:-------:|
| **Whisper Medium** | **14.72%** | **15.39%** | **10.91%** | 15.92% | 31.58% | 38.46% | 986 |
| Parakeet-TDT-0.6B | 15.54% | 16.70% | 11.63% | 17.50% | 34.38% | 44.12% | 986 |
| Whisper Large | 15.88% | 16.83% | 11.36% | 19.27% | 35.21% | 48.94% | 986 |
| Qwen3-ASR-1.7B | 15.93% | 16.64% | **12.28%** | **15.88%** | **33.85%** | 44.90% | 986 |
| Whisper Base | 17.44% | 18.29% | 13.33% | 16.99% | 38.16% | 50.00% | 986 |

---

## Normalization Impact

| Mode | Base | Medium | Large | Parakeet | Qwen3 |
|------|:----:|:------:|:-----:|:--------:|:-----:|
| `transcript_raw` | 27.95% | 24.14% | 25.62% | 28.09% | 33.16% |
| `transcript_clean` | **17.44%** | **14.72%** | **15.88%** | **15.54%** | **15.93%** |
| `hf_raw` | 31.76% | 29.83% | 30.95% | 33.85% | 36.36% |
| `hf_clean` | 18.00% | 15.73% | 16.91% | 16.34% | 16.87% |

**Critical finding:** `hf_raw` is 3.81–5.69 pp **worse** than even the raw mode — the dataset's `Normalised_Transcript` column contains harmful systematic errors. Using it as a reference without correction gives invalid WER.

**Normalization reduces WER by ~10–17 pp** — the Qwen3 gap (33.16% → 15.93% = −17.23 pp) reflects the model's heavy use of punctuation and casing that normalization removes.

---

## Model Analysis

### Whisper Medium vs Whisper Large

| Metric | Medium | Large |
|--------|:------:|:-----:|
| Corpus WER | **14.72%** | 15.88% |
| Std Dev | **15.92%** | 19.27% |
| P95 WER | **38.46%** | 48.94% |

Whisper Large is more prone to hallucination on Indian-accented speech — generates confident but incorrect text during pauses, occasionally producing non-English characters (Korean, Cyrillic). Medium is more conservative and consistent.

### Parakeet-TDT-0.6B

- **Beats Whisper Large (15.54% vs 15.88%)** despite being 0.6B vs ~1.5B parameters
- Best model for **female speakers** (11.61%) and **Non-Engineering** (13.85%)
- **Excellent on 60s+ clips** (18.35%) — far better than Whisper Large (38.23%)
- Weakest on very short clips (0–5s: 40.00%) — TDT architecture struggles with minimal context

### Qwen3-ASR-1.7B

- Tied with Whisper Large (15.93% vs 15.88%) at ~3× fewer parameters
- **Lowest standard deviation** (15.88%) — most consistent model overall
- **Second best on 60s+ clips** (20.49%) — robust to long audio
- High `transcript_raw` (33.16%) due to rich punctuation output, but fully corrected by normalization

---

## Breakdown by Speech Rate

| Speech Rate | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:-----------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| FAST | 16.35% | **13.46%** | 13.77% | 14.30% | 14.76% | 413 |
| AVG | 15.89% | **13.41%** | 16.00% | 13.89% | 14.91% | 199 |
| SLOW | 19.85% | 17.21% | 18.69% | 18.23% | **18.14%** | 373 |

SLOW speech is consistently hardest. Qwen3 edges out all models on SLOW (18.14%); Medium dominates FAST and AVG.

---

## Breakdown by Region

| Region | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| EAST | 16.78% | **13.92%** | 16.94% | 15.42% | 15.42% | 352 |
| NORTH | 17.01% | **14.72%** | 15.08% | 15.98% | 15.55% | 202 |
| SOUTH | 18.27% | **15.27%** | 15.58% | 15.57% | 16.57% | 362 |
| WEST | 17.29% | 15.40% | 14.98% | **14.76%** | 16.01% | 69 |

Moderate regional variation (~1.5 pp range for Medium). Parakeet leads for WEST.

---

## Breakdown by Gender

| Gender | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| Female | 13.88% | 12.02% | 12.49% | **11.61%** | 13.17% | 58 |
| Male | 17.65% | **14.88%** | 16.09% | 15.78% | 16.09% | 927 |

Parakeet achieves best female-speaker WER (11.61%). ~3 pp gender gap consistent across all models. Dataset is 94% male.

---

## Breakdown by Discipline

| Discipline | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:----------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| Engineering | 17.92% | **15.06%** | 16.02% | 16.27% | 16.48% | 691 |
| Non-Engineering | 16.30% | 13.90% | 15.55% | **13.85%** | 14.63% | 294 |

Parakeet best for Non-Engineering (13.85%). Engineering is harder due to domain-specific vocabulary.

---

## Breakdown by Audio Duration

| Duration | Base | Medium | Large | Parakeet | Qwen3 |
|:--------:|:----:|:------:|:-----:|:--------:|:-----:|
| 0–5s | 25.00% | 25.00% | 25.00% | 40.00% | 30.00% |
| 5–15s | 24.72% | 21.23% | 24.89% | 23.79% | 23.19% |
| **15–30s** | 16.87% | **13.78%** | 14.73% | 14.90% | 15.20% |
| 30–60s | 19.63% | 19.80% | 22.31% | **18.93%** | 20.15% |
| **60s+** | 33.33% | 37.31% | 38.23% | **18.35%** | 20.49% |

**Most striking finding:** Parakeet and Qwen3 outperform ALL Whisper models on 60s+ clips (18–20% vs 33–38%). Whisper hallucinates heavily on long audio; Parakeet-TDT and Qwen3 are architecturally more robust.

15–30s is the sweet spot for all models. Very short clips (0–5s) are uniformly harder — single-word errors cause high WER and Parakeet struggles most (40%).

---

## Common Error Patterns

1. **Mathematical notation** — equations have no standard spoken form; variable names are misrecognized
2. **SLOW speech hallucinations** — Whisper (especially Large) generates filler text during long pauses
3. **Technical vocabulary** — domain terms (`"gel permeation chromatography"`, `"sludge drying beds"`) frequently misrecognized
4. **Code-switching** — Hindi/regional language words in English lectures cause confusion
5. **Very short references** — 1–3 word references inflate WER (single error → WER = 0.33–1.0)

---

## Conclusions

1. **Whisper Medium (14.72%)** is the best overall model for Indian English academic speech.
2. **Parakeet-TDT-0.6B (15.54%) beats Whisper Large (15.88%)** — specialized smaller models can outperform larger general-purpose ones.
3. **Qwen3-ASR-1.7B (15.93%)** is competitive with Whisper Large and has the lowest standard deviation of any model — a reliable choice when consistency matters.
4. **Parakeet and Qwen3 dominate long audio (60s+)** — 18–20% vs 37–38% for Whisper Large. Critical for deployment on lecture recordings.
5. **Normalization choice matters more than model size** — ~10–17 pp swing from normalization vs 1–2 pp between adjacent models.
6. **The dataset's `Normalised_Transcript` is unreliable** — contains systematic errors that inflate WER by 3.8–5.7 pp. Always use `transcript_clean` for research.
7. **SLOW speech and 60s+ audio** are the hardest conditions universally.
