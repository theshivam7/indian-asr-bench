# Full results and analysis

Detailed companion to [README.md](README.md): every breakdown table, statistical test, normalization
detail, and error-analysis finding behind the headline numbers. Start with the README for the
overview; come here for the evidence.

## Contents

- [Pipeline in detail](#pipeline-in-detail)
- [Models](#models)
- [Datasets](#datasets)
- [Results: TIE_shorts](#results-tie_shorts)
- [Results: Svarah](#results-svarah)
- [Results: AESRC2020 (Indian)](#results-aesrc2020-indian)
- [Fine-tuning and split design (exploratory)](#fine-tuning-and-split-design-exploratory)
- [Inference efficiency](#inference-efficiency)
- [Normalization](#normalization)
- [Error analysis](#error-analysis)
- [Reproducing each table](#reproducing-each-table)
- [Tested environment](#tested-environment)
- [Data availability](#data-availability)
- [Limitations](#limitations)
- [Future work](#future-work)

---

## Pipeline in detail

| Step | What it does | Output |
|---|---|---|
| Registry | [`utils/registry.py`](utils/registry.py) defines every model, dataset, mode, and display name. Single source of truth. | - |
| Stage 1 | An engine driver transcribes a dataset's eval split. Committed and immutable: the reproducibility anchor. | `results/<dataset>/stage1_raw_transcripts/` |
| Stage 2 | [`normalize_and_score.py`](normalize_and_score.py) computes per-clip WER/CER under every normalization mode. | `results/<dataset>/stage2_processed/` |
| Stage 3 | Comparisons, cluster-bootstrap statistics, error taxonomy, fine-tuning reports, charts. | `results/<dataset>/analysis/` |

Any normalization or metric change re-runs Stages 2 and 3 from the committed transcripts. No re-inference needed.

The per-clip Stage-2 CSVs are not tracked in git. `python normalize_and_score.py --dataset <ds>` rebuilds them in minutes and must be run before any `analysis/` script on a fresh clone. Only `wer_summary_all_models.csv` per dataset is committed, and CI checks that a rebuild is byte-identical to it.

Extending the benchmark:

- **New dataset**: add one `DatasetSpec` to the registry. No other file changes.
- **New model**: add one `ModelSpec`, then run its engine driver with `--model`.
- **New metric**: add it to [`utils/wer_compute.py`](utils/wer_compute.py) and surface it in Stage 2/3.

### Decode settings

Settings are recorded per run in `results/<dataset>/stage1_raw_transcripts/wer_<model>_manifest.json`: model + dataset revisions, package versions, git commit, decode kwargs, host, timestamp.

| Engine | Explicit settings | Everything else |
|---|---|---|
| openai-whisper (base/medium/large/large-v3-turbo) | `language="en"` (+ `fp16=False` on CPU) | library defaults: greedy decoding with temperature fallback (0.0 to 1.0 in 0.2 steps on quality-gate failure), `condition_on_previous_text=True`, default no-speech/compression thresholds |
| hf_whisper (HF baselines and fine-tuned models, e.g. medium_hf / medium_aesrc_ft) | chunked `transformers` pipeline ([`utils/transcribe_hf.py`](utils/transcribe_hf.py)) | library defaults |
| NeMo (parakeet / parakeet_ctc) | batch transcription, `batch_size=16` | library defaults |
| qwen3 | `language="English"`, `max_new_tokens=512` | library defaults |

openai-whisper's temperature fallback is stochastic: clips that fail its quality gates at temperature 0 are re-decoded at sampled temperatures, so a fresh Stage-1 run can differ slightly on those clips. Decode settings were left at community defaults on purpose, because they are what practitioners run. This is why the **committed Stage-1 raw CSVs are the reproducibility anchor**, not the decode process.

HF dataset revisions are pinned in [`utils/registry.py`](utils/registry.py) (`hf_revision`), so an upstream dataset update cannot silently change the benchmark. The `whisper_norm` mode uses [`whisper_normalizer==0.1.0`](https://pypi.org/project/whisper-normalizer/), verified byte-identical to [`openai/whisper`](https://github.com/openai/whisper)'s `EnglishTextNormalizer` on all 7,391 distinct reference and hypothesis strings in TIE.

---

## Models

| Model | Parameters | Architecture | Reference |
|-------|:----------:|:------------:|-----------|
| Whisper Tiny | 39M | Encoder-Decoder | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| Whisper Base | 74M | Encoder-Decoder | [openai/whisper-base](https://huggingface.co/openai/whisper-base) |
| Whisper Small | 244M | Encoder-Decoder | [openai/whisper-small](https://huggingface.co/openai/whisper-small) |
| Whisper Medium | 769M | Encoder-Decoder | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) |
| Whisper Large-v3 | ~1.5B | Encoder-Decoder | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) |
| Whisper large-v3-turbo | 809M | Encoder-Decoder | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| Parakeet-TDT-0.6B-v2 | 600M | CTC + TDT | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) |
| Parakeet-CTC-1.1B | 1.1B | CTC | [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) |
| Qwen3-ASR-1.7B | 1.7B | LLM-based | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| Whisper Tiny (AESRC fine-tuned) | 39M | Encoder-Decoder | [theshivam7/whisper-tiny-aesrc-indian-english](https://huggingface.co/theshivam7/whisper-tiny-aesrc-indian-english) |
| Whisper Small (AESRC fine-tuned) | 244M | Encoder-Decoder | [theshivam7/whisper-small-aesrc-indian-english](https://huggingface.co/theshivam7/whisper-small-aesrc-indian-english) |
| Whisper Medium (AESRC fine-tuned) | 769M | Encoder-Decoder | [theshivam7/whisper-medium-aesrc-indian-english](https://huggingface.co/theshivam7/whisper-medium-aesrc-indian-english) |

A TIE fine-tuned set (Tiny/Small/Medium) also exists on the HF Hub but is archived from the main benchmark, see [Archived: TIE_shorts fine-tuning](archived_tasks/tie_finetuning/README.md).

All nine pretrained models run as-is on all three datasets; that is the headline benchmark. The fine-tuning study is exploratory and analyzed separately. Fine-tuned models are excluded from the pretrained ranking tables because they decode through a different engine (HF `transformers` rather than `openai-whisper`); their engine-controlled comparison is in [Fine-tuning](#fine-tuning-and-split-design-exploratory).

---

## Datasets

**[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts)**: academic lecture audio scraped from YouTube (NPTEL style). "Found" data, no controlled recording protocol.

**[ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)**: read-speech prompts recorded under a controlled protocol. "Curated" data, the counterpoint to TIE.

**[pengyizhou/accented_english](https://huggingface.co/datasets/pengyizhou/accented_english)** (AESRC2020, Indian subset): short read commands and queries from the Accented English Speech Recognition Challenge 2020 ([Shi et al., ICASSP 2021](https://arxiv.org/abs/2102.10233)). The mirror carries 8 national accents; the pipeline filters to `accent == INDIAN` on load. Its test split is natively speaker-disjoint from train (481 vs 38 speakers, zero overlap), which makes it the clean instrument for the fine-tuning study. Licence position in [Data availability](#data-availability).

| Dataset | Split | Clips | Duration | Mean / clip | Median / clip |
|---------|:-----:|------:|:--------:|:-----------:|:--------------:|
| TIE_shorts | train | 7,200 (filtered from 7,884 raw) | 46.9h | - | - |
| TIE_shorts | validation | 986 | 6.84h | 24.98s | 24.62s |
| TIE_shorts | test (eval, scored) | 986 (985 scored, 1 empty reference) | 6.72h | 24.53s | 24.20s |
| Svarah | test (eval-only, scored) | 6,656 | 9.61h | 5.20s | 4.21s |
| AESRC (Indian) | train | 12,820 | 17.48h | 4.91s | - |
| AESRC (Indian) | validation | 532 | 0.76h | 5.12s | - |
| AESRC (Indian) | test (eval, scored) | 1,731 | 2.15h | 4.47s | - |

TIE's and AESRC's `test` splits are the eval sets; `train` and `validation` are used only for fine-tuning. Svarah has no train or validation split and is eval-only.

**TIE_shorts test-split demographics:**

| Attribute | Distribution |
|-----------|-------------|
| Gender | Male 94.1% (927), Female 5.9% (58) |
| Speech rate | FAST 41.9% (413), SLOW 37.9% (373), AVG 20.2% (199) |
| Region | SOUTH 36.8% (362), EAST 35.7% (352), NORTH 20.5% (202), WEST 7.0% (69) |
| Discipline | Engineering 70.2% (691), Non-Engineering 29.8% (294) |

**Svarah test-split demographics:**

| Attribute | Distribution |
|-----------|-------------|
| Gender | Female 53.8% (3,579), Male 46.2% (3,077) |
| Age | 30-45 40.1% (2,670), 18-30 33.3% (2,219), 45-60 19.6% (1,305), 60+ 6.9% (462) |
| Native language | 19 languages (Assamese, Bengali, Bodo, Gujarati, Hindi, Kannada, and more); 65 districts per the [dataset paper](https://arxiv.org/abs/2305.15760) |

---

## Results: TIE_shorts

All numbers are WER on the `test` split under **`transcript_clean`**, the gold mode: forward normalization applied symmetrically to reference and hypothesis (see [Normalization](#normalization)).

### Primary metric: transcript_clean

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|
| Whisper Medium | 14.76% | 15.45% | 11.11% | 15.90% | 31.58% | 39.62% |
| Parakeet-TDT-0.6B-v2 | 15.60% | 16.75% | 11.86% | 17.47% | 34.38% | 44.12% |
| Whisper Large-v3 | 15.93% | 16.88% | 11.43% | 19.20% | 35.21% | 48.94% |
| Whisper Small | 16.05% | 16.90% | 12.20% | 17.78% | 34.38% | 46.00% |
| Parakeet-CTC-1.1B | 16.45% | 17.23% | 12.90% | 15.95% | 34.69% | 44.19% |
| Qwen3-ASR-1.7B | 16.66% | 17.34% | 12.90% | 15.93% | 35.00% | 45.07% |
| Whisper Base | 17.53% | 18.38% | 13.51% | 16.95% | 38.16% | 50.00% |
| Whisper large-v3-turbo | 17.98% | 18.80% | 12.00% | 23.62% | 38.89% | 56.52% |
| Whisper Tiny | 19.43% | 20.49% | 16.28% | 17.41% | 40.28% | 51.76% |

<p align="center">
  <img src="results/tie/analysis/wer_by_model.png" width="720" alt="TIE_shorts model ranking by corpus WER with 95% confidence intervals">
</p>

Statistical check: speaker-clustered paired bootstrap over 280 speakers, Holm-corrected across all 36 pairs ([full tables](results/tie/analysis/statistics_transcript_clean.md)).

- 24 of the 36 pairs are significant.
- Whisper Medium beats Small (-1.29 pp) and Large-v3 (-1.17 pp). A smaller model wins against bigger ones here.
- Medium also beats Parakeet-TDT (-0.84 pp, p<sub>Holm</sub>=0.031), the narrowest significant margin on this corpus.
- Small, Large-v3, both Parakeets, and Qwen3 are mutually indistinguishable in most pairings.
- Whisper Base (74M) is statistically tied with large-v3-turbo (809M), diff -0.45 pp. Model size alone does not predict rank here.

### By normalization mode

Corpus WER under OpenAI's `EnglishTextNormalizer` (`whisper_norm`) instead of this project's `transcript_clean` normalizer, same gold reference.

| Model | `transcript_clean` (gold) | `whisper_norm` | Δ |
|-------|:--------------------------:|:---------------:|:-:|
| Whisper Medium | 14.76% | 14.47% | -0.29 pp |
| Parakeet-TDT-0.6B-v2 | 15.60% | 15.16% | -0.44 pp |
| Whisper Large-v3 | 15.93% | 15.75% | -0.18 pp |
| Whisper Small | 16.05% | 15.79% | -0.26 pp |
| Parakeet-CTC-1.1B | 16.45% | 16.18% | -0.27 pp |
| Qwen3-ASR-1.7B | 16.66% | 15.40% | -1.26 pp |
| Whisper Base | 17.53% | 17.03% | -0.50 pp |
| Whisper large-v3-turbo | 17.98% | 17.74% | -0.24 pp |
| Whisper Tiny | 19.43% | 19.00% | -0.43 pp |

Both modes score the same 985 clips; the one TIE test clip whose reference is `..` is dropped in every mode.

`whisper_norm` lowers every model's WER, but unevenly. Qwen3 moves the most (-1.26 pp), rising from 6th to 3rd place. The Whisper family barely shifts (~0.2 to 0.5 pp). `transcript_clean` remains the primary metric throughout.

### Key findings

1. Whisper Medium wins at 14.76% corpus WER, and it is also the steadiest model here, with the lowest Std Dev and median of the nine.
2. Parakeet-TDT (600M, 15.60%) edges out Whisper Large-v3 (~1.5B, 15.93%), though not by a significant margin. It does lose to Whisper Medium by a margin that is significant.
3. WER falls as Whisper capacity grows, up to Medium (Tiny 19.43%, Base 17.53%, Small 16.05%, Medium 14.76%), then climbs back up at Large-v3 and large-v3-turbo. Bigger is not better on this data.
4. large-v3-turbo is the least stable model on this corpus, with the highest Std Dev (23.62%). It hallucinates on hard clips more than anything else tested here.
5. Reference and normalizer choice alone move WER by 2.3 to 3.5 pp on this dataset (see [Normalization](#normalization)).

The five breakdowns below use the top 5 models by corpus WER (Medium, Parakeet-TDT, Large-v3, Small, Parakeet-CTC).

### Breakdown by speech rate

| Speech Rate | Medium | Parakeet-TDT | Large-v3 | Small | Parakeet-CTC | Samples |
|:-----------:|:------:|:-------------:|:--------:|:-----:|:------------:|:-------:|
| FAST | 13.54% | 14.38% | 13.85% | 14.53% | 15.44% | 413 |
| AVG | 13.41% | 13.95% | 16.01% | 14.80% | 15.38% | 199 |
| SLOW | 17.24% | 18.25% | 18.72% | 18.88% | 18.47% | 373 |

### Breakdown by region

| Region | Medium | Parakeet-TDT | Large-v3 | Small | Parakeet-CTC | Samples |
|:------:|:------:|:-------------:|:--------:|:-----:|:------------:|:-------:|
| EAST | 13.95% | 15.44% | 16.95% | 15.71% | 15.99% | 352 |
| NORTH | 14.74% | 16.06% | 15.10% | 16.22% | 16.61% | 202 |
| SOUTH | 15.34% | 15.64% | 15.67% | 16.03% | 16.86% | 362 |
| WEST | 15.47% | 14.86% | 15.06% | 17.22% | 15.97% | 69 |

### Breakdown by audio duration

| Duration | Medium | Parakeet-TDT | Large-v3 | Small | Parakeet-CTC |
|:--------:|:------:|:-------------:|:--------:|:-----:|:------------:|
| 0-5s | 25.00% | 40.00% | 25.00% | 25.00% | 30.00% |
| 5-15s | 21.61% | 23.91% | 25.28% | 22.46% | 25.36% |
| 15-30s | 13.82% | 14.96% | 14.77% | 14.90% | 15.79% |
| 30-60s | 19.83% | 18.93% | 22.35% | 22.60% | 19.83% |
| 60s+ | 37.31% | 18.35% | 38.23% | 45.87% | 20.18% |

Both Parakeet variants hold steady on 60s+ clips while every Whisper size degrades: Whisper hallucinates during long pauses, the TDT/CTC decoders do not. The extreme buckets are tiny (n=4 for 0-5s, n=5 for 60s+), so read them qualitatively; 87% of clips sit in 15-30s.

### Breakdown by gender

| Gender | Medium | Parakeet-TDT | Large-v3 | Small | Parakeet-CTC | Samples |
|:------:|:------:|:-------------:|:--------:|:-----:|:------------:|:-------:|
| Female | 12.05% | 11.78% | 12.46% | 13.99% | 12.02% | 58 |
| Male | 14.92% | 15.83% | 16.14% | 16.18% | 16.71% | 927 |

### Breakdown by discipline

| Discipline | Medium | Parakeet-TDT | Large-v3 | Small | Parakeet-CTC | Samples |
|:----------:|:------:|:-------------:|:--------:|:-----:|:------------:|:-------:|
| Engineering | 15.09% | 16.30% | 16.06% | 16.36% | 16.89% | 691 |
| Non-Engineering | 13.99% | 13.95% | 15.64% | 15.35% | 15.41% | 294 |

### YouTube captions (archived reference)

YouTube auto-captions score 51.88% WER on the 190 clips with available English captions, 3.8x worse than Whisper Medium on the same clips (13.67%). Not directly comparable to the main benchmark; kept in [`archived_tasks/youtube_captions/`](archived_tasks/youtube_captions/).

---

## Results: Svarah

Svarah has no alternate dataset-provided reference, so three modes apply: `transcript_raw`, `transcript_clean` (gold), and `whisper_norm`. All nine models were run.

### Primary metric: transcript_clean

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|
| Whisper Large-v3 | 7.11% | 11.68% | 0.00% | 32.27% | 28.57% | 71.43% |
| Whisper Medium | 7.89% | 13.59% | 0.00% | 45.49% | 33.33% | 100.00% |
| Whisper large-v3-turbo | 8.10% | 13.45% | 0.00% | 63.13% | 33.33% | 100.00% |
| Whisper Small | 10.06% | 17.33% | 0.00% | 93.53% | 37.50% | 100.00% |
| Parakeet-TDT-0.6B-v2 | 11.73% | 17.26% | 2.63% | 35.42% | 50.00% | 100.00% |
| Qwen3-ASR-1.7B | 11.82% | 13.35% | 0.00% | 30.98% | 40.00% | 66.67% |
| Whisper Base | 14.53% | 25.36% | 6.67% | 84.42% | 64.29% | 100.00% |
| Parakeet-CTC-1.1B | 15.65% | 21.80% | 6.67% | 40.95% | 66.67% | 100.00% |
| Whisper Tiny | 19.96% | 34.95% | 11.11% | 212.89% | 83.33% | 100.00% |

<p align="center">
  <img src="results/svarah/analysis/wer_by_model.png" width="720" alt="Svarah model ranking by corpus WER with 95% confidence intervals">
</p>

Reading the distribution columns:

- Median WER is 0.00% for five of nine models. Svarah has many short read prompts that good models get exactly right, so corpus WER is the more informative headline.
- Std Dev is far higher than on TIE (Tiny: 212.89% vs 17.41%). On isolated-word items a single wrong word can score far above 100% WER (see [Error Analysis](#error-analysis)).
- Some models return nothing on the shortest clips. Out of 6,656, empty hypotheses under `transcript_clean` are: Tiny 56, Medium 40, Base 38, Large-v3 27, Small 13, Parakeet-CTC 12, Parakeet-TDT 4, large-v3-turbo 0, Qwen3 0. Nearly all are one- or two-word references shorter than a second. An empty hypothesis scores 100% WER and is counted in the numbers above. TIE has no empties and AESRC has one.

Statistical check, this time recording-clustered because the public release exposes no speaker IDs: paired bootstrap over 3,232 recording clusters, Holm-corrected across all 36 pairs ([full tables](results/svarah/analysis/statistics_transcript_clean.md)).

- 34 of the 36 pairs are significant.
- The two that are not: Medium vs large-v3-turbo (-0.21 pp) and Parakeet-TDT vs Qwen3 (-0.09 pp). Everywhere else the ranking holds up.

### By normalization mode

| Model | `transcript_raw` | `transcript_clean` (gold) | `whisper_norm` |
|-------|:---:|:---:|:---:|
| Whisper Large-v3 | 7.49% | 7.11% | 6.80% |
| Whisper Medium | 8.18% | 7.89% | 7.69% |
| Whisper large-v3-turbo | 8.32% | 8.10% | 7.76% |
| Whisper Small | 10.40% | 10.06% | 9.91% |
| Parakeet-TDT-0.6B-v2 | 13.03% | 11.73% | 8.35% |
| Qwen3-ASR-1.7B | 13.48% | 11.82% | 8.32% |
| Whisper Base | 14.88% | 14.53% | 14.37% |
| Parakeet-CTC-1.1B | 17.71% | 15.65% | 11.18% |
| Whisper Tiny | 20.33% | 19.96% | 19.52% |

### Key findings

1. Whisper Large-v3 wins at 7.11%, roughly half its own TIE score (15.93%). Controlled read speech is an easier problem than scraped lecture audio.
2. Normalization matters even more here. Parakeet-TDT drops from 13.03% (raw) to 8.35% (whisper_norm), a 4.7 pp swing, and Parakeet-CTC recovers 6.5 pp. Both transcribe fillers like "and uh" verbatim, which `transcript_clean` counts as insertions and `whisper_norm` strips out. Whisper models drop fillers by training, so they barely move.
3. Svarah is cleaner than TIE once the classifier is audited. Its artifact share among classifiable clips is 0.8%, against TIE's 1.2%. Run the classifier naively and it reports 4.8%, but that is an instrument artifact: isolated-word items auto-flag on any single-word miss ("tree" heard as "three"), and on those clips the models disagree with each other (inter-hypothesis distance 0.92), the opposite signature of a genuine reference fault (see [Error Analysis](#error-analysis)).

---

## Results: AESRC2020 (Indian)

AESRC's Indian subset is short, prompted read speech (mean 4.47s/clip, filtered to `accent == INDIAN`). Like Svarah it has no alternate dataset-provided reference, so three modes apply. All nine models were run.

### Primary metric: transcript_clean

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|
| Whisper Large-v3 | 5.20% | 5.95% | 0.00% | 12.10% | 20.00% | 28.57% |
| Qwen3-ASR-1.7B | 5.23% | 6.12% | 0.00% | 13.35% | 20.00% | 28.57% |
| Whisper Medium | 5.73% | 6.53% | 0.00% | 12.25% | 22.22% | 33.33% |
| Whisper large-v3-turbo | 5.81% | 6.58% | 0.00% | 11.97% | 21.43% | 30.00% |
| Parakeet-TDT-0.6B-v2 | 6.26% | 7.34% | 0.00% | 12.56% | 25.00% | 33.33% |
| Whisper Small | 7.23% | 8.30% | 0.00% | 14.81% | 25.00% | 36.36% |
| Parakeet-CTC-1.1B | 7.50% | 8.75% | 0.00% | 13.74% | 27.27% | 37.50% |
| Whisper Base | 9.96% | 11.33% | 6.67% | 16.53% | 33.33% | 42.86% |
| Whisper Tiny | 13.66% | 15.35% | 9.09% | 19.76% | 40.00% | 50.00% |

<p align="center">
  <img src="results/aesrc/analysis/wer_by_model.png" width="720" alt="AESRC Indian model ranking by corpus WER with 95% confidence intervals">
</p>

Statistical check: speaker-clustered paired bootstrap over 481 speakers, Holm-corrected across all 36 pairs ([full tables](results/aesrc/analysis/statistics_transcript_clean.md)).

- 30 of the 36 pairs come out significant. The smallest difference this corpus can separate (0.53 pp) is finer than Svarah's (0.79 pp) or TIE's (0.84 pp), and its 481 clusters are real speakers.
- Large-v3 and Qwen3 are joint leaders: inseparable from each other (5.20% vs 5.23%, Holm p=1.0). Large-v3 separates from every model below them; Qwen3 from all except Medium (p=0.103).
- Medium, large-v3-turbo, and Parakeet-TDT (5.73-6.26%) have no significant internal pair, and Whisper Small vs Parakeet-CTC (7.23% vs 7.50%) is the remaining tie.

### By normalization mode

| Model | `transcript_raw` | `transcript_clean` (gold) | `whisper_norm` |
|-------|:---:|:---:|:---:|
| Whisper Large-v3 | 5.39% | 5.20% | 4.78% |
| Qwen3-ASR-1.7B | 5.14% | 5.23% | 4.89% |
| Whisper Medium | 6.05% | 5.73% | 5.41% |
| Whisper large-v3-turbo | 6.13% | 5.81% | 5.56% |
| Parakeet-TDT-0.6B-v2 | 6.19% | 6.26% | 5.93% |
| Whisper Small | 7.52% | 7.23% | 6.96% |
| Parakeet-CTC-1.1B | 7.38% | 7.50% | 7.13% |
| Whisper Base | 10.27% | 9.96% | 9.64% |
| Whisper Tiny | 13.91% | 13.66% | 13.21% |

### Key findings

1. Whisper Large-v3 wins at 5.20%, the lowest corpus WER of any model on any dataset in this benchmark. Short, prompted read speech is the easiest condition tested here.
2. Reference quality is excellent. The consensus classifier flags only 0.1% of classifiable clips as artifacts (95% CI 0.0-0.4%), the lowest of the three datasets (TIE 1.2%, Svarah 0.8%).
3. Median WER is 0.00% for seven of nine models. Most clips are short enough that a competent model just gets them right, so corpus WER is again the more honest headline.
4. Qwen3, Parakeet-TDT and Parakeet-CTC score slightly higher under `transcript_clean` than `transcript_raw` (5.14% to 5.23%, 6.19% to 6.26%, 7.38% to 7.50%). The three models whose output is already clean and literal get no help from normalization.

---

## Fine-tuning and split design (exploratory)

This study is included for completeness. It is not part of the paper's core claims, and readers should treat it as exploratory.

Whether in-domain fine-tuning helps is only answerable if the test split isolates the effect being claimed. The two corpora with training splits differ sharply on this.

### TIE_shorts cannot answer the question

All 280 test speakers, and all 986 test clips, come from speakers that also appear in train ([`speaker_overlap.md`](results/tie/analysis/speaker_overlap.md)). There is no clip-level leakage, and this is the corpus's own released partition. But every comparison it supports is speaker-matched, so a gain measured on it mixes accent and content adaptation with adaptation to those particular voices.

Repairing the split in place does not work either. Removing every train speaker who appears in test leaves 567 of 7,200 train clips, a 13x reduction. Three speaker-disjoint runs at that budget all move away from the baseline, one reaching +1.75 pp (p_Holm = 0.048), while three size-matched controls land flat ([`finetune_disjoint_control.md`](results/tie/analysis/finetune_disjoint_control.md)). So the official split's +0.20 pp null was hiding a regression. The TIE fine-tuning study is archived rather than reported, see [Archived: TIE_shorts fine-tuning](archived_tasks/tie_finetuning/README.md).

### AESRC2020 (Indian) can

Its 481 test speakers share zero overlap with the 38 train and validation speakers ([`speaker_overlap.md`](results/aesrc/analysis/speaker_overlap.md)), so a measured gain is generalization to unseen speakers by construction. One recipe ([`finetune_tiny_small.py`](finetune/finetune_tiny_small.py): `max_steps=2000`, effective batch 32, lr 1e-5, fp16, best checkpoint by validation WER) trains all three sizes, so a difference between sizes is a difference in pretrained capacity. Engine-controlled HF-pipeline baseline, 1,731 test clips.

| Size | Params | HF baseline | Fine-tuned | Δ (paired, speaker-clustered) | 95% CI | p (Holm) |
|------|:------:|:-----------:|:----------:|:------------------------------:|:------:|:--------:|
| Whisper Tiny | 39M | 17.45% | 12.64% | -4.81 pp | [-12.30, +1.71] | 0.163 |
| Whisper Small | 244M | 7.22% | 5.64% | -1.58 pp | [-2.01, -1.15] | 0.003 |
| Whisper Medium | 769M | 5.63% | 4.48% | -1.15 pp | [-1.55, -0.77] | 0.003 |

Small and Medium both come out significant. Because train and test share zero speakers, this cannot be memorization of test speakers; the most plausible reading is accent or domain adaptation from the 17.5h of Indian-accent read speech in training. Fine-tuning also cuts the corpus insertion rate at every size: 5.81% to 3.94% (Tiny), 0.95% to 0.79% (Small), 0.70% to 0.50% (Medium) of reference words.

Tiny has the biggest point estimate but a CI wide enough to cross zero. Its outputs are much noisier than the other sizes (Std Dev 103% on the HF baseline, versus 12% for Medium), and that variance keeps the gain from reaching significance.

### Seed study

**A single training run cannot separate a real effect from an unlucky seed**, so all three sizes were retrained from 6 independent seeds (42-47) on the identical recipe and split:

| Size | Seeds | Δ mean (pp) | Δ SD (pp) | Δ min | Δ max |
|------|:---:|:---:|:---:|:---:|:---:|
| Whisper Tiny | 6 | -6.85 | 1.03 | -7.34 | -4.75 |
| Whisper Small | 6 | -1.65 | 0.15 | -1.84 | -1.42 |
| Whisper Medium | 6 | -1.22 | 0.12 | -1.32 | -1.00 |

Every one of the 18 runs improves on its own baseline, and none of the three ranges approaches zero. Tiny's single official-split run (-4.81 pp) was simply the least favorable of its six. This is strong informal evidence of a real effect, not a formal significance claim: no seed-level test has been built. The gain shrinks monotonically with pretrained size, in absolute terms (Tiny -6.85, Small -1.65, Medium -1.22 pp) and relative terms (-39.3%, -22.8%, -21.7%).

**The Tiny gain depends on which baseline you count from.** The deltas above are against the HF-pipeline baseline (17.45%), which is much worse than the same Tiny weights through openai-whisper (13.66%, the leaderboard number) because the chunked HF pipeline hurts Tiny on these short clips. Counted from the leaderboard number, Tiny's 6-seed mean of 10.60% is a gain of 3.06 pp (22%), not 6.85 pp. Small and Medium are unaffected: their two baselines agree within 0.1 pp.

Full seed data: [`finetune_seeds_transcript_clean.md`](results/aesrc/analysis/finetune_seeds_transcript_clean.md) and [`finetune_seeds_transcript_clean_per_seed.csv`](results/aesrc/analysis/finetune_seeds_transcript_clean_per_seed.csv). The six checkpoints per size are on the Hub: [Tiny](https://huggingface.co/theshivam7/whisper-tiny-aesrc-indian-english-seeds), [Small](https://huggingface.co/theshivam7/whisper-small-aesrc-indian-english-seeds), [Medium](https://huggingface.co/theshivam7/whisper-medium-aesrc-indian-english-seeds).

### Normalizer check

**These results were checked against the normalizer choice**, since this repository's own finding is that a single-normalizer result can be an artifact of the normalizer:

| Size | Δ under `transcript_clean` | Δ under `whisper_norm` | Swing |
|------|:---:|:---:|:---:|
| Whisper Tiny, 1 seed (official split) | -4.81 pp | -7.14 pp | 2.33 pp |
| Whisper Small, 1 seed | -1.58 pp | -1.55 pp | 0.03 pp |
| Whisper Medium, 1 seed | -1.15 pp | -1.08 pp | 0.07 pp |
| Whisper Tiny, 6-seed mean | -6.85 pp (SD 1.03) | -7.11 pp (SD 0.04) | 0.26 pp |
| Whisper Small, 6-seed mean | -1.65 pp (SD 0.15) | -1.66 pp (SD 0.13) | 0.01 pp |
| Whisper Medium, 6-seed mean | -1.22 pp (SD 0.12) | -1.15 pp (SD 0.09) | 0.07 pp |

All three sizes are normalizer-invariant at the 6-seed mean (swing 0.26, 0.01, 0.07 pp). Tiny's 2.33 pp swing on its single official-split seed was mostly seed noise: under `transcript_clean` five of its six seeds fall inside a 0.12 pp band (-7.34 to -7.22) and seed 42 alone sits at -4.75, while under `whisper_norm` that same seed is unremarkable. So the instability is one anomalous run whose excess errors `whisper_norm` normalizes away, not a general property of training at 39M. Seed data under `whisper_norm`: [`finetune_seeds_whisper_norm.md`](results/aesrc/analysis/finetune_seeds_whisper_norm.md).

Full per-size reports: [`finetune_comparison_tiny.md`](results/aesrc/analysis/finetune_comparison_tiny.md), [`finetune_comparison_small.md`](results/aesrc/analysis/finetune_comparison_small.md), [`finetune_comparison_medium.md`](results/aesrc/analysis/finetune_comparison_medium.md), [full capacity summary](results/aesrc/analysis/finetune_capacity_summary.md).

<p align="center">
  <img src="results/aesrc/analysis/finetune_comparison_medium.png" width="680" alt="Whisper Medium pretrained vs fine-tuned on AESRC Indian">
</p>

---

## Inference efficiency

WER alone does not say what a system costs to run at scale. The quality-gated offline sweep measures that, and it is complete: 9 models x 3 corpora x 8 batch sizes (1 to 128) = 216 measurements, no OOM and no failed entries. Every run used 512 clips, 3 untimed warmup batches, 3 timed repeats, one A100-SXM4-40GB, and a single CUDA 12.4 runtime for all three engines. All 27 result files carry one provenance digest.

RTFx is audio seconds processed per wall second; higher is faster. `b1` is the batch-1 row of this same sweep.

| Model | RTFx b1 TIE | RTFx best TIE | Batch | RTFx b1 Sva | RTFx best Sva | Batch | RTFx b1 AES | RTFx best AES | Batch |
|---|---:|---:|:-:|---:|---:|:-:|---:|---:|:-:|
| Whisper Tiny | 77.1 | 291 | 128 | 41.9 | 79 | 128 | 37.2 | 67 | 128 |
| Whisper Base | 64.4 | 289 | 128 | 35.2 | 77 | 128 | 31.0 | 66 | 128 |
| Whisper Small | 41.5 | 279 | 128 | 27.3 | 73 | 128 | 23.7 | 62 | 128 |
| Whisper Medium | 24.3 | 236 | 128 | 18.2 | 64 | 64 | 16.4 | 56 | 128 |
| Whisper Large-v3 | 18.9 | 193 | 128 | 14.5 | 56 | 64 | 13.0 | 48 | 64 |
| Whisper large-v3-turbo | 76.6 | 250 | 128 | 37.0 | 62 | 64 | 32.6 | 53 | 64 |
| Parakeet-TDT-0.6B-v2 | 271.4 | 2251 | 128 | 84.6 | 537 | 8 | 69.7 | 1958 | 128 |
| Parakeet-CTC-1.1B | 228.1 | 228 | 1 | 53.6 | 202 | 4 | 44.7 | 1492 | 128 |
| Qwen3-ASR-1.7B | 15.8 | 289 | 128 | 14.9 | 202 | 64 | 14.2 | 263 | 128 |

### Batching changes the ranking that batch 1 reports

Qwen3-ASR is the slowest system at batch 1 on TIE (15.8 RTFx, against 18.9 for Large-v3) and sits near the bottom on the other two corpora, yet it reaches parity with the best Whisper by batch 128. Its batching speedup is the largest of any system on TIE (18.3x) and Svarah (13.6x), and it holds the highest sustained GPU utilization in the panel (83.8% mean SM on TIE). An LLM-based recognizer looks uncompetitive under a single-stream measurement and competitive under an offline one, so the protocol decides the conclusion.

### Whisper barely uses the GPU on short clips

Whisper's short-form path pads every clip to a fixed 30-second window, so on 4-second audio most of the batch is padding. Whisper Tiny sits at 1.8% mean SM utilization on Svarah and 2.0% on AESRC, against 37 to 61% for Parakeet and 52 to 63% for Qwen3, and its batching speedup there is only 1.8 to 1.9x against Parakeet-CTC's 33.4x. Dividing the padded window by the real audio in each 512-clip workload gives 1.29x on TIE, 5.66x on Svarah and 6.62x on AESRC, reported as `padded_rtfx_audio_s_per_s`. So on AESRC a Whisper system credited with 67 RTFx is sustaining 442 RTFx of padded audio. The column is blank for NeMo and Qwen3, which pad dynamically. Parakeet still leads on the padded basis, but the gap narrows from roughly 20x to 3.5x.

### The 0.10 pp quality gate binds asymmetrically

25 of the 216 sweep entries are rejected by the gate, all of them Parakeet or Qwen3 entries. No Whisper entry is ever rejected, because 30-second padding makes Whisper's numerics independent of batch size while NeMo pads to batch maximum.

For Parakeet-TDT and Qwen3 the gate is filtering batch-order noise, not decode drift: it is non-monotonic (TIE Parakeet-TDT fails at 8 to 64 and passes at 128), two-sided (Svarah Parakeet-TDT at batch 4 is rejected for scoring 0.195 pp *better* than batch 1), and corpus-inconsistent (the same models pass at batch 128 on AESRC). Parakeet-CTC on TIE is different: it emits 6 empty hypotheses at batch 1 and 8 or 9 at every larger batch under the fp16 autocast the throughput runtime uses, so those rejections are dropped output. The fp32 leaderboard run of the same model emitted no empties.

The published effect is that Parakeet-CTC on TIE is reported at 228 RTFx when batch 64 measured 1,719, a 7.5x understatement, with 5.8x on Svarah for the same model and 2.9x for Parakeet-TDT. `gate_cost_x` reports what the pre-registered gate gives up: exactly 1.00 for all 18 Whisper rows, and above 1.00 only for Parakeet rows (TIE CTC 7.53, Svarah TDT 2.94, Svarah CTC 5.82). The gate runs in the direction that flatters Whisper. Per-batch reject reasons are in each `throughput_<dataset>.md` and `throughput_<dataset>_sweep.csv`.

### Peak memory is a padding artifact too

Whisper Large-v3 at batch 128 on TIE reaches 38,155 MiB of the 40,442 MiB usable, which is why Svarah and AESRC select batch 64 for it. Those figures describe the padded window, not the model's weights.

Full per-batch data: [`throughput_tie.md`](results/tie/analysis/throughput_tie.md), [`throughput_svarah.md`](results/svarah/analysis/throughput_svarah.md), [`throughput_aesrc.md`](results/aesrc/analysis/throughput_aesrc.md), with the raw sweep in the matching `throughput_<dataset>_sweep.csv`.

---

## Normalization

Every WER number above depends on the reference field and the normalizer chosen before comparison. TIE's reference swap shifts every model 2.3 to 3.5 pp, and normalizer choice alone moves the verbatim models up to 6.5 pp on Svarah. That is as much as the gap between mid-tier models. It also reaches the conclusions: re-running the full inference stack under both normalizers changes **6 of 36 Holm-corrected pairwise verdicts on TIE**, against 0 of 36 on Svarah and AESRC.

### Normalizers and modes

Three normalizers do all the work ([`utils/normalize.py`](utils/normalize.py)):

| Normalizer | What it does | Used by |
|---|---|---|
| `minimal_clean_text` | Strip wrapping quotes, lowercase, remove punctuation. No number or possessive handling. | `*_raw` modes |
| `normalize_text` | Unicode NFC, possessive fix (`"Bernoulli's"` to `"bernoulli s"`), ordinals and cardinals to words (`"1st"` to `"first"`), lowercase, strip punctuation, collapse whitespace. Contractions stay unexpanded on both sides. | `*_clean` modes |
| `whisper_normalize_text` | OpenAI's `EnglishTextNormalizer`, the widely used reference implementation. It does expand contractions. | `whisper_norm` mode |

All normalization is applied symmetrically to reference and hypothesis. TIE has both a gold reference and a dataset-provided alternate, so five modes apply; Svarah and AESRC have only a gold reference, so three:

| Mode | Reference | Normalizer | Purpose |
|------|-----------|:-------------:|---------|
| `transcript_raw` | gold (`Transcript` / `text`) | `minimal_clean_text` | Near-upper-bound baseline |
| `transcript_clean` | gold (`Transcript` / `text`) | `normalize_text` | Verbatim-faithful, pre-registered primary metric |
| `whisper_norm` | gold (`Transcript` / `text`) | `whisper_normalize_text` | Disfluency-insensitive cross-check against a widely used normalizer |
| `hf_raw` | `Normalised_Transcript` (TIE only) | `minimal_clean_text` | Quantifies dataset normalization errors |
| `hf_clean` | `Normalised_Transcript` (TIE only) | `normalize_text` | Dataset normalization plus our fix |

### Why the dataset's `Normalised_Transcript` is unreliable

TIE, corpus WER:

| Mode | Base | Medium | Large-v3 | Parakeet | Qwen3 |
|------|:----:|:------:|:--------:|:--------:|:-----:|
| `transcript_raw` (minimal cleanup) | 17.91% | 15.11% | 16.31% | 15.97% | 18.15% |
| `transcript_clean` (verbatim-faithful, primary) | 17.53% | 14.76% | 15.93% | 15.60% | 16.66% |
| `hf_raw` (dataset's normalization, broken) | 20.24% | 18.01% | 19.14% | 18.54% | 17.99% |
| `hf_clean` (dataset norm + our fix) | 18.07% | 15.76% | 16.94% | 16.40% | 17.61% |

- `Normalised_Transcript` maps `"the 1st component"` to `"the one s t component"` (ordinals split into characters), affecting 50+ clips.
- That inflates `hf_raw` WER by 2.7 to 3.3 pp over the gold mode for the seven Whisper and Parakeet-TDT systems.
- The two most verbatim systems are exceptions: Qwen3 (+1.3 pp) and Parakeet-CTC (+0.7 pp; raw-vs-raw its sign even flips). Their punctuation-rich literal output happens to agree better with the mangled reference.
- Reference faults are style-dependent, so they cannot be differenced out across models. Prefer `transcript_clean` over either `hf_*` mode.

### Does the normalizer change what the benchmark concludes?

`transcript_clean` and `whisper_norm` answer different questions. `transcript_clean` scores against what was actually said, so faithfully transcribed disfluencies count as content. `whisper_norm` deletes fillers and hesitations first, so it measures agreement on lexical content only. It returns a lower WER for every system on every corpus, which reflects leniency rather than accuracy.

Whether that choice matters was tested by re-running the whole inference stack (cluster bootstrap, all 36 pairs, Holm correction) under both:

| Corpus | Significant, `transcript_clean` | Significant, `whisper_norm` | Verdicts that change | WER span, `whisper_norm` |
|---|:---:|:---:|:---:|:---:|
| TIE_shorts | 24/36 | 24/36 | **6** | 4.5 pp |
| Svarah | 34/36 | 34/36 | 0 | 12.7 pp |
| AESRC2020 (Indian) | 30/36 | 30/36 | 0 | 8.4 pp |

The six TIE pairs whose verdict depends on the normalizer:

| Pair | `transcript_clean` | `whisper_norm` |
|---|---|---|
| Base vs Large-v3 | +1.59 pp, p=0.011, significant | +1.28 pp, p=0.055, not significant |
| Base vs Qwen3 | +0.86 pp, p=0.058, not significant | +1.63 pp, p=0.007, significant |
| Medium vs Parakeet-TDT | -0.84 pp, p=0.031, significant | -0.69 pp, p=0.055, not significant |
| large-v3-turbo vs Qwen3 | +1.31 pp, p=0.136, not significant | +2.35 pp, p=0.007, significant |
| Parakeet-CTC vs Qwen3 | -0.22 pp, p=1.000, not significant | +0.79 pp, p=0.007, significant |
| Parakeet-TDT vs Qwen3 | -1.07 pp, p=0.007, significant | -0.23 pp, p=1.000, not significant |

The Parakeet-CTC versus Qwen3 pair reverses the sign of the difference as well as the verdict.

What drives this is movement relative to the margins between systems, not movement alone. Svarah's models move most under the normalizer (mean 1.44 pp, up to 4.47 pp) and reorder nothing, because its nine systems are spread across 12.7 pp. TIE barely moves (mean 0.43 pp) and flips six verdicts, because its nine systems are packed into 4.7 pp and the movement is uneven: Qwen3 gains 1.26 pp where its neighbours gain about 0.25 pp. A densely packed leaderboard is exactly where the normalizer quietly decides the published result, so both modes are reported throughout. Rankings under `whisper_norm` live in `results/<dataset>/analysis/statistics_whisper_norm.csv`.

### Metrics

Defined in [`utils/wer_compute.py`](utils/wer_compute.py). WER and CER are standard substitutions + deletions + insertions over the reference word or character count. An empty hypothesis counts as all-deletions in both metrics.

Confidence intervals use a speaker-clustered (TIE, AESRC) or recording-clustered (Svarah) paired bootstrap with 10,000 resamples and Holm correction across every pairwise family. The two-sided p floor is 2/(B+1), so 10,000 resamples put it at 0.0002; at the earlier 2,000 the floor was 0.001, which Holm correction across 36 pairs pushed to 0.036, close enough to 0.05 that family size rather than evidence was setting the verdict.

---

## Error analysis

Clip/reference misalignment is detected by a full-corpus, multi-model consensus classifier, not a hand-reviewed sample. It uses two per-clip signals averaged across all nine models: reference-word recall and hypothesis/reference length ratio. Clips with references under 4 words are excluded as unclassifiable (`short_ref`): recall is quantized there and one wrong word crosses any threshold. Full evidence: [TIE report](results/tie/analysis/error_analysis_transcript_clean.md), [Svarah report](results/svarah/analysis/error_analysis_transcript_clean.md), [AESRC report](results/aesrc/analysis/error_analysis_transcript_clean.md).

| | TIE_shorts | Svarah | AESRC (Indian) |
|---|:---:|:---:|:---:|
| Detected artifact share, a lower bound (classifiable clips, refs >=4 words) | 1.2% (95% CI 0.7-2.1%) | 0.8% (95% CI 0.6-1.1%) | 0.1% (95% CI 0.0-0.4%) |
| Short-reference (<4 words) share of corpus | 0.1% (1 clip) | 23.0% (1,530 clips) | 0.7% (12 clips) |
| Worst-20-per-model tail: artifacts | 66.7% (54 tail clips) | 3.4% (117 tail clips) | 20.8% (77 tail clips) |
| Per-model WER inflation from artifacts | 0.55-0.75 pp | 0.31-0.39 pp | 0.03-0.08 pp |

How to read this table:

- Detected reference artifacts are rare in all three corpora but dominate TIE's worst-20 tail. The earlier hand-analysis figure of ~70% holds up as a tail statistic; it was never a corpus-level number.
- The share is a lower bound. On the 49 human-reviewed TIE clips the classifier flagged 12, all of them reference errors, but 34 of the 37 unflagged clips were reference errors too (dropped clauses, mangled technical terms). Its precision there is 12 of 12 and its recall about a quarter (12 of 46).
- AESRC has the cleanest references: 2 flagged clips in the whole corpus and at most 0.08 pp of WER inflation. Its worst-20 tail is mostly genuine recognition errors on Indian named entities.
- Svarah's tail is 95% isolated-word items. Run the classifier naively there and it reports 4.8%, an instrument artifact: sub-second single-word clips auto-flag on any miss, yet the models disagree with each other on them (inter-hypothesis distance 0.92 vs 0.17-0.23 on TIE's true artifacts).

Two independent lines of evidence that TIE's flagged clips are reference errors, not model errors:

1. **Clip over-run.** Models transcribe the reference correctly plus real speech the clip cut off. A CTC model that cannot hallucinate (Parakeet), an LLM (Qwen3), and Whisper all emit the same extra words. Example (`-2aOCNaOiLs`): REF "considered in problem forty five"; every model adds "let us do that" and scores 80% WER while being correct.
2. **Inter-hypothesis agreement.** On flagged clips the models agree with each other (mean pairwise distance 0.17 to 0.23) while all disagreeing with the reference (0.88 to 1.0 WER against it). These systems share no decoder or training objective, so the fault sits in the reference.

On Svarah the same check runs in reverse: its `clip_over_run` flags show the agreement signature (0.17) but its `content_mismatch` flags do not (0.79), so Svarah's true reference-fault rate is, if anything, below the 0.8% headline.

Other TIE patterns (evidence in the report):

- SLOW speech is 38% of the data but the majority of the high-WER tail. The cause is truncated reference windows on slow, self-correcting delivery, not worse acoustics.
- Errors are U-shaped by duration: over-represented at 0-5s and 60s+, under-represented in the 15-30s middle.
- Hallucination is the biggest genuine failure mode, and large-v3-turbo (Std Dev 23.62%) is its worst offender.
- No female speaker appears in any model's top-20 worst clips. Small sample, but consistent across models.

Implications:

- Median WER (11.1% for Medium on TIE) is a more honest estimate of typical quality than corpus WER (14.8%). The gap is the rare-but-severe tail.
- Rankings are unaffected because every model hits the same artifacts. Absolute numbers are inflated by roughly 0.6 pp on TIE, 0.35 pp on Svarah, and under 0.1 pp on AESRC.

### Classifier validation (human review)

| Corpus | Status | Sheet |
|---|---|---|
| TIE_shorts | Complete, 49 clips annotated | [`analysis/tie_validation/`](analysis/tie_validation/) |
| Svarah | Pending: 60-clip sheet built, not annotated | [`analysis/svarah_validation/`](analysis/svarah_validation/) |
| AESRC (Indian) | Pending: 28-clip sheet built, not annotated | [`analysis/aesrc_validation/`](analysis/aesrc_validation/) |

The classifier is a heuristic, not ground truth. To check it, a human transcribed the true content of the 49 TIE clips with WER above 40% on at least 3 of 4 strong models (Large-v3, Parakeet-TDT, Parakeet-CTC, Qwen3), listening to the audio directly. Every model hypothesis and the dataset reference were then scored against that corrected transcript under the same `transcript_clean` normalization used everywhere else.

**Headline result.** Mean WER on these 49 clips is 64.8% against the original reference and 17.0% against the corrected one, a 47.8 pp drop (95% bootstrap CI 40.3 to 55.9 pp; Wilcoxon signed-rank p < 1e-8). Every model shows the same pattern, all significant after Holm correction, and 48 of 49 clips improve. The one exception (a list of Gujarat place names) is also the one clip judged a genuine model failure. The numbers are recomputed by `analysis/tie_validation/review_stats.py`, which writes [`human_review_stats.md`](results/tie/analysis/human_review_stats.md).

**Cause per clip:** 46 of 49 are reference errors (a dropped clause, a wrong number, a mangled technical term, or in 5 cases a reference from a different segment of the lecture), 2 are genuine model failures, 1 stays unresolved. The `-2aOCNaOiLs` example above is one of the 49, and the review independently reaches the same verdict for it.

Other findings from the review:

- **Technical vocabulary drives a lot of this.** References regularly mangle domain terms ("singlet state" becomes "simplest state", "resolution" becomes "solution"). 14 of 49 clips show this pattern.
- **Some reference errors flip the meaning.** One reference drops the word "no", turning "there is no functional dependency" into its opposite.
- **Model-level pattern.** Medium comes out cleanest against the corrected reference (mean WER 15.1%, 2 of 49 clips still wrong) and Large-v3 the worst (20.9%, 8 of 49), mostly through degenerate repetition loops.

**Scope:** this sample is not random. It was built by requiring several strong models to agree a clip is hard, so it explains why these 49 clips are hard and does not estimate what fraction of corpus errors are reference-caused. It is also a single annotator working non-blind, chosen for diagnostic depth over a formal blind protocol. Svarah and AESRC review sheets use the same flagging rule and are built but not annotated, so nothing in this document depends on them.

---

## Reproducing each table

Stage 1 (inference) is not re-run; the committed transcripts are the anchor. Everything below runs on CPU from the top-level `requirements.txt`. Run `python normalize_and_score.py --dataset <ds>` first on a fresh clone, because the per-clip Stage-2 CSVs are not tracked and every `analysis/` script reads them.

| Table or figure in this document | Command | Read from |
|---|---|---|
| Primary metric tables (all three datasets) | `python normalize_and_score.py --dataset <ds>` | `results/<ds>/stage2_processed/wer_summary_all_models.csv` |
| By normalization mode tables (all three datasets) | `python normalize_and_score.py --dataset <ds>` | `results/<ds>/stage2_processed/wer_summary_all_models.csv` |
| TIE `Normalised_Transcript` mode table (Normalization section) | `python normalize_and_score.py --dataset tie` | `results/tie/stage2_processed/wer_summary_all_models.csv` |
| TIE breakdowns (speech rate, region, duration, gender, discipline) and `wer_by_model.png` charts | `python analysis/compare_all.py --dataset <ds>` | `results/<ds>/analysis/` |
| Significance counts, pairwise bullets, verdict-change tables (Normalization section) | `python analysis/statistics.py --dataset <ds> [--mode whisper_norm]` | `results/<ds>/analysis/statistics_<mode>.{csv,md}` |
| Error analysis table | `python analysis/error_analysis.py --dataset <ds>` | `results/<ds>/analysis/error_analysis_transcript_clean.{csv,md}` |
| Fine-tuning tables (single-seed, 6-seed, normalizer check) | `python analysis/compare_finetune.py --dataset aesrc` and `python analysis/compare_seeds.py --dataset aesrc --mode all` | `results/aesrc/analysis/finetune_*.{csv,md}` |
| Speaker overlap counts (TIE and AESRC) | `python finetune/check_speaker_overlap.py --dataset <ds>` | `results/<ds>/analysis/speaker_overlap.md` |
| Throughput table and gate-cost figures | `python analysis/compare_throughput.py --dataset <ds>` | `results/<ds>/analysis/throughput_<ds>.{csv,md}` and `throughput_<ds>_sweep.csv` |
| Benchmark overview figure (README hero figure) | `python analysis/make_overview_figure.py` | `results/benchmark_overview.png` |
| Human review numbers (49-clip TIE study) | `python analysis/tie_validation/review_stats.py` | `analysis/tie_validation/review_sheet.csv` in, `results/tie/analysis/human_review_stats.md` out |
| Empty-hypothesis counts | `python normalize_and_score.py --dataset <ds>`, then count empty `hypothesis` cells | `results/<ds>/stage2_processed/transcript_clean/wer_<model>_transcript_clean.csv` |

Two items have no generating script in `analysis/`. The dataset split and demographic tables are taken from the dataset metadata at the pinned HF revisions. The TIE disjoint-control numbers are read from the committed [`finetune_disjoint_control.md`](results/tie/analysis/finetune_disjoint_control.md).

---

## Tested environment

This is what produced the committed results, taken from the run manifests (`results/<ds>/stage1_raw_transcripts/*_manifest.json` has the per-run detail).

- Python 3.10.20 on linux-64 (the cluster), one NVIDIA A100-SXM4-40GB per job, torch 2.5.1.
- Engine environments used CUDA 11.8. The throughput environments used CUDA 12.4, one runtime for all three engines there.
- datasets 4.8.5 everywhere.

Recorded per family:

| Family | Packages recorded in the manifests |
|---|---|
| openai-whisper | openai-whisper 20250625, numpy 2.2.6, jiwer 4.0.0 |
| NeMo | nemo_toolkit 2.7.3, transformers 4.57.6, numpy 2.2.6, jiwer 3.1.0 |
| Qwen3 | qwen-asr 0.0.6, transformers 4.57.6, numpy 2.2.6 |
| Fine-tuning | transformers 4.46.3, numpy 1.26.4, jiwer 3.1.0, soundfile 0.12.1 |

The requirements files were captured after the runs. They pin numpy 1.26.4 and, for Whisper, jiwer 3.1.0, which differ from the manifests. This does not change any published number: scoring runs in the top-level `requirements.txt` environment, and CI rebuilds Stage 2 byte-identically from the committed transcripts.

The CPU analysis path is tested on Python 3.10 and 3.12 in CI (Ubuntu). The two engine conda YAMLs (`environments/parakeet.yaml` and `environments/qwen3.yaml`) no longer solve; [`environments/resolved/`](environments/resolved/) is the working route.

---

## Data availability

In this repository:

- Stage-1 transcripts and run manifests for every run (`results/<ds>/stage1_raw_transcripts/`).
- Aggregate Stage-2 summaries (`results/<ds>/stage2_processed/wer_summary_all_models.csv`).
- All analysis tables and figures (`results/<ds>/analysis/`).
- Throughput JSONs (`results/<ds>/throughput/`).
- Human-review sheets (`analysis/tie_validation/`, `analysis/svarah_validation/`, `analysis/aesrc_validation/`).

On the Hugging Face Hub: the fine-tuned checkpoints, linked in the [Models](#models) section.

Needs your own download: the three datasets, from their HF pages linked in [Datasets](#datasets). Svarah is gated and needs `hf auth login`. Audio is never redistributed by this repository.

AESRC licence position: the mirror (`pengyizhou/accented_english`) states no licence, and the corpus is Datatang's. Access and permission to use it for this research were confirmed through our advisor. Redistribution or commercial use would need separately clarified terms.

---

## Limitations

- The human review behind the classifier check is one annotator, working non-blind, on a targeted 49-clip sample of the hardest TIE clips. It explains why those clips are hard; it does not give a corpus-wide reference-fault rate. On that sample the classifier's precision is 12 of 12 but its recall is about a quarter, so every artifact share in this document is a lower bound.
- Svarah can only be clustered by recording (3,232 clusters), not by its 117 true speakers, since the public release exposes no speaker IDs. True speaker clustering would widen the confidence intervals.
- The fine-tuning seed study has no seed-level significance test. Every seed improves on its baseline, but this is reported as strong informal evidence, not a confirmed result.
- The throughput gate rejects only NeMo and Qwen3 entries, never a Whisper one, so the published operating points are conservative for the batched engines and unaffected for Whisper. RTFx divides by real audio seconds while Whisper pads every clip to 30 seconds, so Whisper's throughput and memory figures on the short-clip corpora reflect the padded window. The gate compares each runtime only to its own batch 1, never to the leaderboard run.
- Training-data contamination is possible: NPTEL lectures are public and may appear in Whisper's training data. A small probe (n=10) found no memorization signal, but it is low-powered.
- Stage-1 transcripts are single runs with temperature-fallback decoding. The committed raw CSVs are the reproducibility anchor.
- Some cells are small: duration extremes have n=4 to 5 clips, and TIE has only 58 female-speaker clips. Read those qualitatively.

---

## Future work

- Annotate the staged Svarah (60-clip) and AESRC (28-clip) review sheets with the same protocol as TIE.
- Run a blind review on a random sample of TIE to get a corpus-wide reference-fault rate, which the targeted 49-clip sample cannot give.
- Evaluate the AESRC fine-tuned checkpoints on TIE and Svarah, to see whether the gains carry across registers or stay domain-locked.
