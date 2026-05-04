# WER Analysis Summary — Whisper + YouTube on Indian English (TIE_shorts)

## Dataset

- **Source:** [raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts)
- **Split:** `test` (986 samples, 52,178 reference words)
- **Domain:** Indian English NPTEL academic lectures
- **Distribution:** 928M / 58F | FAST 413, SLOW 373, AVG 199 | SOUTH 362, EAST 352, NORTH 202, WEST 69

---

## Main Results (transcript_clean — gold standard)

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 | Samples |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|:-------:|
| **Whisper Medium** | **14.72%** | **15.39%** | **10.91%** | **15.92%** | **31.58%** | **38.46%** | 986 |
| Whisper Large | 15.88% | 16.83% | 11.36% | 19.27% | 35.21% | 48.94% | 986 |
| Whisper Base | 17.44% | 18.29% | 13.33% | 16.99% | 38.16% | 50.00% | 986 |
| YouTube (clip-aligned) | 51.88% | 51.59% | 50.00% | 8.35% | — | — | 190† |

†YouTube WER computed only on the 190 samples with manual English captions (19.3% of 986).

**Whisper Medium is the best model for Indian English speech.**

---

## Normalization Impact

| Mode | Description | Base | Medium | Large | YouTube† |
|------|-------------|:----:|:------:|:-----:|:--------:|
| `transcript_raw` | No normalization | 27.95% | 24.14% | 25.62% | 51.72% |
| `transcript_clean` | Forward normalization, clean ref | **17.44%** | **14.72%** | **15.88%** | **51.88%** |
| `hf_raw` | Dataset's broken normalization | 31.76% | 29.83% | 30.95% | 65.94% |
| `hf_clean` | Dataset norm + our fix | 18.00% | 15.73% | 16.91% | 53.35% |

**Critical finding:** `hf_raw` is 3.81–5.69 pp **worse** than even the raw mode, proving the dataset's `Normalised_Transcript` column contains harmful errors. Using it as a reference without correction gives invalid WER.

**Normalization reduces WER by ~10 pp for Whisper** — larger than the gap between any two models. For YouTube, normalization has near-zero impact (<0.2 pp), indicating errors are vocabulary/content mismatches not formatting differences.

---

## Normalization Pipeline

Applied symmetrically to both reference and hypothesis in `*_clean` modes:

1. Unicode NFC normalization
2. Contraction expansion: `"don't"` → `"do not"`
3. Possessive fix: `"Bernoulli's"` → `"bernoulli s"` (not `"bernoulli is"`)
4. Ordinals to words: `"1st"` → `"first"`, `"2nd"` → `"second"`
5. Cardinals to words: `"100"` → `"one hundred"`, `"60,000"` → `"sixty thousand"`
6. Lowercase
7. Punctuation removal
8. Whitespace normalization

---

## Why Medium Beats Large

| Metric | Medium | Large |
|--------|:------:|:-----:|
| Corpus WER | **14.72%** | 15.88% |
| Std Dev (variance) | **15.92%** | 19.27% |
| Hard cases WER > 1.0 (top 20) | 30% (6/20) | **45% (9/20)** |
| Hallucination rate (top 20) | 40% | **75%** |
| Non-English output observed | No | Yes (Korean, Cyrillic) |

Whisper Large is more prone to hallucination on out-of-distribution Indian English accents. It generates confident but incorrect text during pauses, and occasionally produces non-English characters. Medium is more conservative and consistent.

---

## Breakdown by Speech Rate

| Speech Rate | Base | Medium | Large | Samples |
|:-----------:|:----:|:------:|:-----:|:-------:|
| FAST | 16.35% | **13.46%** | 13.77% | 413 |
| AVG | 15.89% | **13.41%** | 16.00% | 199 |
| SLOW | 19.85% | **17.21%** | 18.69% | 373 |

SLOW speech is consistently hardest (+3.7 pp vs FAST for Medium) — Whisper hallucinates during long pauses.

---

## Breakdown by Region

| Region | Base | Medium | Large | Samples |
|:------:|:----:|:------:|:-----:|:-------:|
| EAST | 16.78% | **13.92%** | 16.94% | 352 |
| NORTH | 17.01% | **14.72%** | 15.08% | 202 |
| SOUTH | 18.27% | **15.27%** | 15.58% | 362 |
| WEST | 17.29% | 15.40% | **14.98%** | 69 |

Moderate regional variation (~1.5 pp range for Medium). SOUTH has the most speakers and highest WER for Base.

---

## Breakdown by Gender

| Gender | Base | Medium | Large | Samples |
|:------:|:----:|:------:|:-----:|:-------:|
| Female | 13.88% | **12.02%** | 12.49% | 58 |
| Male | 17.65% | **14.88%** | 16.09% | 927 |

Female speakers have ~3 pp lower WER consistently across all models. Dataset is 94% male — this finding should be interpreted carefully.

---

## Breakdown by Discipline

| Discipline | Base | Medium | Large | Samples |
|:----------:|:----:|:------:|:-----:|:-------:|
| Engineering | 17.92% | **15.06%** | 16.02% | 691 |
| Non-Engineering | 16.30% | **13.90%** | 15.55% | 294 |

Engineering lectures have slightly higher WER (~1.2 pp) due to domain-specific mathematical/technical vocabulary.

---

## Breakdown by Audio Duration

| Duration | Base | Medium | Large |
|:--------:|:----:|:------:|:-----:|
| 0–5s | 25.0% | 25.0% | 25.0% |
| 5–15s | 24.72% | 21.23% | 24.89% |
| **15–30s** | **16.87%** | **13.78%** | **14.73%** |
| 30–60s | 19.63% | 19.80% | 22.31% |
| 60s+ | 33.33% | 37.31% | 38.23% |

15–30s is the sweet spot. 60s+ shows severe degradation (+23 pp vs 15–30s for Medium) — all models struggle on very long clips.

---

## Common Error Patterns

1. **Mathematical notation** — Equations like `"ds/dt = π r² H"` have no standard spoken form; variable names are misrecognized
2. **SLOW speech hallucinations** — Whisper (especially Large) generates filler text during long pauses
3. **Technical vocabulary** — Domain terms (`"gel permeation chromatography"`, `"sludge drying beds"`) frequently misrecognized
4. **Code-switching** — Hindi/regional language words in English lectures cause confusion
5. **Very short references** — 1–3 word references inflate WER (a single error → WER = 0.33–1.0)

---

## YouTube Captions

YouTube captions fetched via Google Colab (`task4_youtube_captions/fetch_youtube_captions_colab.ipynb`) to bypass IP blocking on the NSCC server. Clip alignment implemented via sliding-window Jaccard (`task4_youtube_captions/align_youtube_captions.py`).

---

## YouTube Results

### Coverage

| Metric | Value |
|--------|-------|
| Total samples | 986 |
| With English captions | 190 (19.3%) |
| Unavailable | 796 (80.7%) |
| Caption type | manual only |

### WER — Raw vs Clip-Aligned (available samples only, n=190)

Full-video hypothesis (~6,300 words mean) vs short clip reference (~53 words mean). Direct WER is ~11,000%. After sliding-window Jaccard alignment (window = ref_len × 1.5):

| Mode | Full Video WER | Clip-Aligned WER | Reduction |
|------|:--------------:|:----------------:|:---------:|
| `transcript_raw` | 11,156% | **51.72%** | −11,104 pp |
| `transcript_clean` | 11,877% | **51.88%** | −11,825 pp |
| `hf_raw` | 11,034% | 65.94% | −10,968 pp |
| `hf_clean` | 11,841% | 53.35% | −11,788 pp |

Normalization has minimal impact for YouTube (51.72% → 51.88%), unlike Whisper where it reduces WER by ~10 pp. The `hf_raw` mode is 14 pp worse — consistent with the broken `Normalised_Transcript` effect seen on all models.

### Fair Model Comparison — Same 190 Samples (transcript_clean)

| Model | Corpus WER | Mean WER | Median WER | Samples |
|-------|:----------:|:--------:|:----------:|:-------:|
| **Whisper Medium** | **13.67%** | **14.65%** | **11.04%** | 190 |
| Whisper Large | 14.35% | 14.98% | 10.57% | 190 |
| Whisper Base | 15.66% | 16.23% | 13.11% | 190 |
| YouTube (clip-aligned) | 51.88% | 51.59% | 50.00% | 190 |

YouTube WER is **3.8× worse** than Whisper Medium on identical samples. Low std dev (8.35%) indicates consistently poor performance — alignment locates the correct window, but YouTube ASR struggles with Indian-accented academic speech and domain vocabulary.

### Breakdown by Region (transcript_clean, same 190 samples)

| Region | YouTube | Medium | n |
|:------:|:-------:|:------:|:-:|
| EAST | 50.99% | 14.39% | 63 |
| WEST | 51.63% | 14.99% | 16 |
| NORTH | 52.09% | 12.38% | 39 |
| SOUTH | 52.54% | 13.48% | 72 |

### Breakdown by Speech Rate (transcript_clean, same 190 samples)

| Speech Rate | YouTube | Medium | n |
|:-----------:|:-------:|:------:|:-:|
| AVG | 51.35% | 11.54% | 45 |
| FAST | 52.06% | 12.22% | 77 |
| SLOW | 52.00% | 17.16% | 68 |

Unlike Whisper (5.6 pp spread across speech rates on this subset), YouTube shows near-zero variation (~0.7 pp) — dominant error source is vocabulary/content mismatch, not speaking rate.

### Coverage by Region / Speech Rate / Discipline

| Region | With Captions | Total | Coverage |
|:------:|:-------------:|:-----:|:--------:|
| WEST | 16 | 69 | 23.2% |
| SOUTH | 72 | 363 | 19.8% |
| NORTH | 39 | 202 | 19.3% |
| EAST | 63 | 352 | 17.9% |

| Speech Rate | With Captions | Total | Coverage |
|:-----------:|:-------------:|:-----:|:--------:|
| AVG | 45 | 199 | 22.6% |
| FAST | 77 | 413 | 18.6% |
| SLOW | 68 | 374 | 18.2% |

| Discipline | With Captions | Total | Coverage |
|:----------:|:-------------:|:-----:|:--------:|
| Non-Engineering | 67 | 295 | 22.7% |
| Engineering | 123 | 691 | 17.8% |

---

## Conclusions

1. **Whisper Medium (14.72% WER)** is the best model for Indian English academic speech — not Large
2. **Normalization choice matters more than model size** — 10 pp swing from normalization vs 3 pp between models
3. **The dataset's `Normalised_Transcript` is unreliable** — contains systematic errors that inflate WER by 3.8–5.7 pp
4. **SLOW speech and 60s+ audio** are the hardest conditions (+3.7 pp and +23 pp respectively)
5. **Forward normalization** (digits → words, contraction expansion, symmetric) is the correct approach for research-grade WER
6. **YouTube (clip-aligned) WER is 51.88%** — 3.8× worse than Whisper Medium on the same 190 clips. Only 19.3% of videos have English captions. Low variance (std 8.35%) indicates consistent failure, not random errors. Normalization has near-zero impact for YouTube (<0.2 pp), confirming errors stem from vocabulary/content mismatch rather than formatting.
