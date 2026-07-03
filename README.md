<h1 align="center">Indian-ASR-Bench</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Datasets-TIE__shorts%20+%20Svarah-orange" />
  <img src="https://img.shields.io/badge/Test%20clips-986%20+%206656-purple" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <a href="https://huggingface.co/datasets/raianand/TIE_shorts">
    <img src="https://img.shields.io/badge/Dataset-TIE__shorts-yellow?logo=huggingface" />
  </a>
  <a href="https://huggingface.co/datasets/ai4bharat/Svarah">
    <img src="https://img.shields.io/badge/Dataset-Svarah-yellow?logo=huggingface" />
  </a>
  <a href="https://huggingface.co/theshivam7/whisper-medium-indian-english">
    <img src="https://img.shields.io/badge/Model-whisper--medium--indian--english-yellow?logo=huggingface" />
  </a>
  <a href="https://huggingface.co/theshivam7/whisper-medium-indian-english-disjoint">
    <img src="https://img.shields.io/badge/Model-speaker--disjoint%20variant-yellow?logo=huggingface" />
  </a>
</p>

<p align="center">
  <b>A reproducible Word Error Rate benchmark for ASR on Indian English speech —<br>
  two datasets, seven models, five normalization modes, and an honest speaker-disjoint fine-tuning study.</b>
</p>

<p align="center">
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#fine-tuning-whisper-medium">Fine-tuning</a> &nbsp;·&nbsp;
  <a href="#evaluation-methodology">Methodology</a> &nbsp;·&nbsp;
  <a href="#reproducing-results">Reproduce</a>
</p>

---

## Motivation

ASR benchmarks are dominated by American and British English. Indian English — spoken by over a billion people, with distinct phonology, regional accents, and code-switching — is under-evaluated, and academic lectures (rapid speech, technical vocabulary, heavy male-speaker skew) are an especially hard and practically important slice.

This project does four things:

1. **Benchmarks up to seven pretrained ASR systems** on two datasets — [TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) (986 NPTEL-style lecture clips, "found" YouTube data) and [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (6,656 curated read-speech clips) — across five normalization modes and every demographic/acoustic breakdown.
2. **Fine-tunes Whisper Medium** on the in-domain `train` split and evaluates it two ways: under the dataset's official (speaker-overlapping) splits, and under a **speaker-disjoint re-split run with 3 training seeds** — reporting the honest result, including a statistically significant *regression* in one seed, and disclosing that the official splits leave only **567 of 7,200 train clips** speaker-disjoint, so disjointness and training-set size are confounded by construction (see [Fine-tuning](#fine-tuning-whisper-medium)).
3. **Analyzes the failure modes** in depth with a full-corpus, multi-model consensus artifact classifier (not just a hand-reviewed tail): reference artifacts are rare in both corpora (TIE 1.0%, Svarah 0.8% of classifiable clips) but dominate TIE's worst-WER tail (62%), while Svarah's tail is instead dominated by an **isolated-word subtask** (23% of its clips have <4-word references) where WER is quantized and the classifier is undefined — see [Error Analysis](#error-analysis).
4. **Validates that classifier against human judgment** with a blind, stratified annotation protocol, rather than trusting an unvalidated heuristic (see [`analysis/validation/PROTOCOL.md`](analysis/validation/PROTOCOL.md)).

Two recurring themes: **how you normalize text moves WER as much as which model you pick**, and **the median clip is 3–4 pp better than the corpus WER** because a rare-but-severe tail (reference artifacts on TIE; sub-second isolated-word items on Svarah) inflates the average.

---

## Models

| Model | Parameters | Architecture | Reference |
|-------|:----------:|:------------:|-----------|
| **Whisper Base** | 74M | Encoder-Decoder | [openai/whisper-base](https://huggingface.co/openai/whisper-base) |
| **Whisper Medium** | 769M | Encoder-Decoder | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) |
| **Whisper Large** | ~1.5B | Encoder-Decoder | [openai/whisper-large](https://huggingface.co/openai/whisper-large) |
| **Whisper large-v3-turbo** | 809M | Encoder-Decoder | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| **Parakeet-TDT-0.6B-v2** | 600M | CTC + TDT | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) |
| **Parakeet-CTC-1.1B** | 1.1B | CTC | [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) |
| **Qwen3-ASR-1.7B** | 1.7B | LLM-based | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| **Whisper Medium (fine-tuned, official split)** | 769M | Encoder-Decoder | [theshivam7/whisper-medium-indian-english](https://huggingface.co/theshivam7/whisper-medium-indian-english) — *this project* |
| **Whisper Medium (fine-tuned, speaker-disjoint)** | 769M | Encoder-Decoder | [theshivam7/whisper-medium-indian-english-disjoint](https://huggingface.co/theshivam7/whisper-medium-indian-english-disjoint) — *this project* |

The pretrained models are evaluated as-is (the headline benchmark, run on both datasets where the checkpoint applies — `large_v3_turbo` and `parakeet_ctc` are Svarah-only additions). The two fine-tunes are TIE-only and analyzed separately in [Fine-tuning Whisper Medium](#fine-tuning-whisper-medium).

---

## Datasets

**[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts)** — the `test` split (986 clips) of the TIE (Talks in Indian English) dataset: NPTEL-style academic lecture audio scraped from YouTube ("found" data, no controlled recording protocol). 985 clips are scored (one is excluded for an empty reference).

| Attribute | Distribution |
|-----------|-------------|
| Gender | Male 94.1% (927), Female 5.9% (58) |
| Speech rate | FAST 41.9% (413), SLOW 37.9% (373), AVG 20.2% (199) |
| Region | SOUTH 36.8% (362), EAST 35.7% (352), NORTH 20.5% (202), WEST 7.0% (69) |
| Discipline | Engineering 70.2% (691), Non-Engineering 29.8% (294) |

**[ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)** — the `test` split (6,656 clips), read-speech prompts recorded under a controlled protocol across Indian speakers ("curated" data — the counterpoint to TIE's "found" data). Eval-only (no train/validation split; not used for fine-tuning).

| Attribute | Distribution |
|-----------|-------------|
| Gender | Female 53.8% (3,579), Male 46.2% (3,077) |
| Age | 30–45 40.1% (2,670), 18–30 33.3% (2,219), 45–60 19.6% (1,305), 60+ 6.9% (462) |
| Native language | 65 languages represented (Assamese, Bengali, Bodo, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, and more) |

---

## Results

### TIE_shorts

All numbers are corpus/per-sample WER on the `test` split under **`transcript_clean`** (the gold-standard mode — forward normalization applied symmetrically to reference and hypothesis; see [Methodology](#evaluation-methodology)). Regenerated directly from `results/`.

#### Primary metric: `transcript_clean`

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|
| **Whisper Medium** | **14.76%** | **15.45%** | **11.11%** | **15.90%** | **31.58%** | **39.62%** |
| Parakeet-TDT-0.6B | 15.60% | 16.75% | 11.86% | 17.47% | 34.38% | 44.12% |
| Whisper Large | 15.93% | 16.88% | 11.43% | 19.20% | 35.21% | 48.94% |
| Qwen3-ASR-1.7B | 16.66% | 17.34% | 12.90% | 15.93% | 35.00% | 45.07% |
| Whisper Base | 17.53% | 18.38% | 13.51% | 16.95% | 38.16% | 50.00% |

> The **fine-tuned** Whisper Medium is not in this ranking because it runs through a different decoder (HuggingFace `transformers`, not `openai-whisper`) — mixing it here would confound fine-tuning with a decoding-engine change. Its fair, engine-controlled comparison is in [Fine-tuning](#fine-tuning-whisper-medium).

<p align="center">
  <img src="results/tie/analysis/wer_by_model.png" width="680" alt="Model ranking by corpus WER (transcript_clean)">
</p>

#### Key findings

1. **Whisper Medium is best overall (14.76%)** — and also the most consistent (lowest Std Dev and lowest median WER).
2. **Parakeet-TDT-0.6B (15.60%) beats Whisper Large (15.93%)** — a 600M specialized model edges out a ~1.5B general-purpose one.
3. **Whisper Large is the least stable** (Std Dev 19.20%) — it hallucinates on the hardest clips.
4. **Parakeet and Qwen3 dominate long audio (60s+):** 18–21% vs 37–38% for the Whisper models.
5. **Normalization/reference choice moves WER by ~2–3 pp** — comparable to the spread between the best and worst models (see below).

#### Impact of normalization

| Mode | Base | Medium | Large | Parakeet | Qwen3 |
|------|:----:|:------:|:-----:|:--------:|:-----:|
| `transcript_raw` (minimal cleanup) | 17.91% | 15.11% | 16.31% | 15.97% | 18.15% |
| `transcript_clean` (**gold standard**) | 17.53% | **14.76%** | 15.93% | 15.60% | 16.66% |
| `hf_raw` (dataset's normalization, broken) | 20.24% | 18.01% | 19.14% | 18.54% | 17.99% |
| `hf_clean` (dataset norm + our fix) | 18.07% | 15.76% | 16.94% | 16.40% | 17.61% |

The dataset's own `Normalised_Transcript` (`hf_raw`) is **2–3 pp worse** than using the gold `Transcript` with correct normalization — it splits ordinals into characters (`"1st"` → `"one s t"`). **Always use `transcript_clean`.**

#### Breakdown by speech rate

| Speech Rate | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:-----------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| FAST | 16.51% | **13.54%** | 13.85% | 14.38% | 15.63% | 413 |
| AVG | 15.96% | **13.41%** | 16.01% | 13.95% | 15.69% | 199 |
| SLOW | 19.87% | 17.24% | 18.72% | 18.25% | **18.65%** | 373 |

#### Breakdown by region

| Region | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| EAST | 16.81% | **13.95%** | 16.95% | 15.44% | 15.81% | 352 |
| NORTH | 17.07% | **14.74%** | 15.10% | 16.06% | 16.26% | 202 |
| SOUTH | 18.44% | **15.34%** | 15.67% | 15.64% | 17.66% | 362 |
| WEST | 17.34% | 15.47% | 15.06% | **14.86%** | 16.51% | 69 |

#### Breakdown by audio duration

| Duration | Base | Medium | Large | Parakeet | Qwen3 |
|:--------:|:----:|:------:|:-----:|:--------:|:-----:|
| 0–5s | 25.00% | 25.00% | 25.00% | 40.00% | 30.00% |
| 5–15s | 25.11% | 21.61% | 25.28% | 23.91% | 23.65% |
| **15–30s** | 16.97% | **13.82%** | 14.77% | 14.96% | 15.98% |
| 30–60s | 19.63% | 19.83% | 22.35% | **18.93%** | 20.62% |
| **60s+** | 33.33% | 37.31% | 38.23% | **18.35%** | 20.80% |

Parakeet and Qwen3 are far more robust on 60s+ clips — Whisper hallucinates during long pauses; the TDT/LLM decoders do not.

#### Breakdown by gender

| Gender | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| Female | 13.92% | 12.05% | 12.46% | **11.78%** | 14.05% | 58 |
| Male | 17.74% | **14.92%** | 16.14% | 15.83% | 16.82% | 927 |

#### Breakdown by discipline

| Discipline | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:----------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| Engineering | 18.00% | 15.09% | 16.06% | 16.30% | 17.14% | 691 |
| Non-Engineering | 16.42% | **13.99%** | 15.64% | **13.95%** | 15.55% | 294 |

#### YouTube captions (archived reference)

YouTube auto-captions, evaluated on the 190 clips (19.3%) with available English captions via clip-aligned Jaccard matching, score **51.88% WER** — 3.8× worse than Whisper Medium on the same clips (13.67%). Not directly comparable to the main benchmark; kept for reference in [`archived_tasks/youtube_captions/`](archived_tasks/youtube_captions/).

### Svarah

Svarah has no `Normalised_Transcript` field (unlike TIE), so only three modes apply: `transcript_raw`, `transcript_clean` (gold), `whisper_norm`. All 7 pretrained models were run, including two Svarah-only additions (`large_v3_turbo`, `parakeet_ctc`).

| Model | `transcript_raw` | `transcript_clean` (gold) | `whisper_norm` |
|-------|:---:|:---:|:---:|
| **Whisper Large** | 7.49% | **7.11%** | 6.80% |
| Whisper large-v3-turbo | 8.32% | 8.10% | 7.76% |
| Whisper Medium | 8.18% | 7.89% | 7.69% |
| Qwen3-ASR-1.7B | 13.48% | 11.82% | 8.32% |
| Parakeet-TDT-0.6B | 13.03% | 11.73% | 8.35% |
| Parakeet-CTC-1.1B | 17.71% | 15.65% | 11.18% |
| Whisper Base | 14.88% | 14.53% | 14.37% |

<p align="center">
  <img src="results/svarah/analysis/wer_by_model.png" width="680" alt="Model ranking by corpus WER on Svarah (transcript_clean)">
</p>

**Key findings:**

1. **Whisper Large is best on Svarah (7.11%)** — roughly half TIE's error rate, consistent with Svarah's controlled read-speech recording vs. TIE's noisier scraped lecture audio.
2. **Normalization choice matters far more here than on TIE for the CTC/TDT/LLM models** — Parakeet-TDT drops from 13.03% (`raw`) to 8.35% (`whisper_norm`), a 4.7pp swing. The mechanism: Parakeet and Qwen3 transcribe fillers verbatim ("and uh", "mm hmm") on Svarah's spontaneous portions and spell digits differently on its read prompts; `transcript_clean` **penalizes the more faithful transcription**, while `whisper_norm` strips fillers and unifies numerals. Whisper models, which omit fillers by training, barely move (7.89% → 7.69%).
3. **The curated dataset is cleaner, as expected — but only after fixing the classifier**: among classifiable clips (references ≥4 words), Svarah's artifact share is **0.8%** vs. TIE's **1.0%**. A naive run of the same classifier reports 4.4% on Svarah — an artifact *of the instrument*: 23% of Svarah's clips are 1–2-word isolated-word items ("cat", "jump", sub-second audio) where any single-word miss saturates recall and auto-flags the clip. On those flagged short clips the models *disagree with each other* (inter-hypothesis distance 0.89) — the signature of genuinely hard decontextualized words ("tree"→"three", "left"→"lift"), the exact opposite of the models-agree/reference-disagrees signature that defines a true reference fault. Heuristic artifact detectors do not transfer across dataset designs unaudited — itself an evaluation-validity lesson.

---

## Fine-tuning Whisper Medium

We fully fine-tune Whisper Medium (the strongest pretrained model here) on the `train` split and evaluate on the **same** `test` split, then compare it against pretrained Whisper Medium **decoded through the identical HF pipeline** (`medium_hf`) — so the comparison isolates fine-tuning from any decoding-engine effect.

**Setup:** full fine-tune (769M params) via `transformers` `Seq2SeqTrainer`; targets = gold `Transcript`; bf16 + gradient checkpointing; LR 1e-5, weight decay 0.01, warmup 10%, SpecAugment; early stopping (patience 2) on validation WER with `load_best_model_at_end`; clips >30s filtered for training. Full hyperparameters in [`task6_whisper_medium_ft/finetune.py`](task6_whisper_medium_ft/finetune.py).

### Result 1: official split — fine-tuning does not beat the pretrained model

Engine-controlled, `transcript_clean`, both decoded through the same HF chunked pipeline:

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|
| Whisper Medium — pretrained (`medium_hf`) | **14.42%** | **14.64%** | **9.84%** | 22.19% | 30.30% | 40.00% |
| Whisper Medium — fine-tuned (`medium_ft`) | 14.61% | 14.80% | 10.14% | 27.70% | **27.87%** | **36.62%** |

Corpus WER across all five modes:

| Mode | Pretrained (`medium_hf`) | Fine-tuned (`medium_ft`) | Δ |
|------|:------------------------:|:------------------------:|:---:|
| `transcript_clean` (gold) | 14.42% | 14.61% | +0.20 pp |
| `transcript_raw` | 14.75% | 14.71% | −0.04 pp |
| `whisper_norm` | 14.23% | 14.31% | +0.08 pp |
| `hf_clean` | 15.51% | 15.70% | +0.19 pp |
| `hf_raw` | 17.72% | 17.70% | −0.02 pp |

The +0.20 pp gap is within single-run noise, so the honest claim on this official-split comparison is **"no significant gain," not "fine-tuning hurts."** The speaker-disjoint re-split below tells a different, more concerning story.

<p align="center">
  <img src="results/tie/analysis/finetune_comparison.png" width="640" alt="Whisper Medium pretrained vs fine-tuned across all five modes">
</p>

### Fine-tuned breakdowns (same analysis as the pretrained models)

**By speech rate**

| Speech Rate | Pretrained (`medium_hf`) | Fine-tuned (`medium_ft`) | Δ | Samples |
|:-----------:|:---:|:---:|:---:|:---:|
| FAST | 11.91% | 12.28% | +0.37 pp | 413 |
| AVG | 14.64% | 15.75% | +1.11 pp | 199 |
| SLOW | 17.69% | 17.10% | −0.59 pp | 373 |

**By region**

| Region | Pretrained (`medium_hf`) | Fine-tuned (`medium_ft`) | Δ | Samples |
|:------:|:---:|:---:|:---:|:---:|
| EAST | 14.08% | 14.66% | +0.58 pp | 352 |
| NORTH | 13.92% | 13.31% | −0.61 pp | 202 |
| SOUTH | 14.08% | 13.66% | −0.42 pp | 362 |
| WEST | 18.84% | 22.53% | +3.69 pp | 69 |

**By gender**

| Gender | Pretrained (`medium_hf`) | Fine-tuned (`medium_ft`) | Δ | Samples |
|:------:|:---:|:---:|:---:|:---:|
| Female | 19.08% | 24.30% | +5.22 pp | 58 |
| Male | 14.14% | 14.03% | −0.11 pp | 927 |

**By discipline**

| Discipline | Pretrained (`medium_hf`) | Fine-tuned (`medium_ft`) | Δ | Samples |
|:----------:|:---:|:---:|:---:|:---:|
| Engineering | 14.39% | 14.46% | +0.07 pp | 691 |
| Non-Engineering | 14.48% | 14.97% | +0.49 pp | 294 |

**By audio duration**

| Duration | Pretrained (`medium_hf`) | Fine-tuned (`medium_ft`) | Δ |
|:--------:|:---:|:---:|:---:|
| 0–5s | 25.00% | 35.00% | +10.00 pp |
| 5–15s | 21.01% | 20.92% | −0.09 pp |
| 15–30s | 12.20% | 12.29% | +0.09 pp |
| 30–60s | 25.41% | 25.69% | +0.28 pp |
| 60s+ | 119.27% | 133.03% | +13.76 pp |

### Why the pretrained model is already hard to beat

- **Little headroom** — pretrained Whisper Medium (680k hours) already sits at 14.42%; the residual errors (technical vocabulary, math notation, code-switching) aren't fixable from a small in-domain set.
- **Scale mismatch → overfitting** — 769M params on ~7.9k clips is data-starved; the best checkpoint arrived at **epoch 1** and early-stopping fired by epoch 3.
- **Tail regressions** — the train set is ~94% male / ~70% engineering; fine-tuning fit that majority and **regressed on under-represented groups** (Female +5.22 pp, WEST +3.69 pp, 0–5s +10 pp), for a net wash (250 clips improved vs 307 regressed).

### Result 2: speaker-disjoint re-split (multi-seed) — not a clean null

Result 1 above is measured on TIE_shorts' **official splits, which have speaker overlap**: every test speaker also appears in training (see the disclosure below). To check whether that overlap was masking a real effect, we removed every training clip whose speaker appears in `test` and re-ran the fine-tune from scratch with **3 independent training seeds** — a single-seed run isn't credible here, since Whisper fine-tune seed variance is the same order of magnitude as the effect being measured.

> **Training-set confound (disclosed).** The official test speakers are so entangled with train that removing them keeps only **567 of 7,200 train clips (3.8 of 46.9 hours; 51 of 331 speakers)**. The disjoint runs therefore differ from Result 1 in *both* speaker overlap and training-set size (~13× smaller) — on this dataset a size-matched speaker-disjoint split is impossible by construction, which is itself an evaluation-validity finding: **TIE_shorts' official splits cannot measure generalization to unseen speakers.** Any regression below must not be attributed to disjointness alone; a size-matched speaker-overlapping control (`hpc/job_finetune_sizematch.pbs`, 567 random clips, 3 seeds) separates the two effects.

| Seed | WER (`transcript_clean`) | Δ vs. pretrained (paired, speaker-clustered bootstrap) | 95% CI | p (Holm-corrected) |
|:----:|:---:|:---:|:---:|:---:|
| **42** | **16.17%** | **+1.75 pp** | **[+0.13, +4.17]** | **0.048 (significant)** |
| 43 | 14.80% | +0.38 pp | [−0.01, +0.74] | 0.116 |
| 44 | 15.20% | +0.79 pp | [−0.18, +2.25] | 0.163 |

Across 3 seeds: WER 15.39% (range 14.80–16.17%), mean Δ +0.97 pp vs. pretrained, seed-to-seed spread 1.37 pp — itself larger than the per-seed effect being estimated. One seed (42) shows a **statistically significant WER regression** that survives Holm-Bonferroni correction across the 3 seeds; the other two fall within the study's minimum detectable effect (≈1.2 pp) and are not distinguishable from no effect.

**The honest claim:** fine-tuning on the speaker-disjoint training subset (567 clips) shows no evidence of improving WER over pretrained, and at least one seed shows evidence of making it measurably worse. Whether the worsening comes from the disjointness or from the 13×-smaller training set is an open question the size-matched control is designed to answer — but either way, the official-split "no gain" reading in Result 1 was, if anything, generous. Full breakdown and methodology: [`results/tie/analysis/finetune_comparison.md`](results/tie/analysis/finetune_comparison.md).

The checkpoint published as [whisper-medium-indian-english-disjoint](https://huggingface.co/theshivam7/whisper-medium-indian-english-disjoint) is seed 42 — the significant-regression seed — published for exact reproducibility of that result, not as a recommended model; see its model card for the full multi-seed context.

> **Long-clip decoding note.** On 60s+ clips the HF chunked pipeline scores ~119% WER for *both* the pretrained and fine-tuned models, vs 37% for the same weights under `openai-whisper`. That is a **decoding-pipeline artifact** (long-form chunk stitching), not a fine-tuning effect — it hits both equally, so the head-to-head stays fair. It also inflates the Std Dev in the tables above.

> ⚠️ **Speaker overlap (disclosed).** The dataset's official splits share speakers: **100% of test speakers — and 100% of test clips — come from speakers also seen in training** ([`speaker_overlap.md`](results/tie/analysis/speaker_overlap.md), via [`check_speaker_overlap.py`](task6_whisper_medium_ft/check_speaker_overlap.py)). There is **no clip-level leakage**, but the comparison is *speaker-matched*, so any effect partly reflects speaker adaptation. The official splits were not modified; this is disclosed so the numbers read correctly.

Model cards: **[whisper-medium-indian-english](https://huggingface.co/theshivam7/whisper-medium-indian-english)** (official split) and **[whisper-medium-indian-english-disjoint](https://huggingface.co/theshivam7/whisper-medium-indian-english-disjoint)** (speaker-disjoint, seed 42).

---

## Evaluation Methodology

Five modes are available; which ones apply depends on the dataset's schema (TIE has both a gold and a dataset-provided reference, so all five apply; Svarah has only a gold reference, so only `transcript_raw`/`transcript_clean`/`whisper_norm` apply). All normalization is applied **symmetrically** to reference and hypothesis:

| Mode | Reference | Normalization | Purpose |
|------|-----------|:-------------:|---------|
| `transcript_raw` | gold (`Transcript` / `text`) | Minimal (case/punct/quotes) | Near-upper-bound baseline |
| `transcript_clean` | gold (`Transcript` / `text`) | Forward (full) | **Gold standard — primary metric** |
| `whisper_norm` | gold (`Transcript` / `text`) | OpenAI's `EnglishTextNormalizer` | Cross-checks the primary metric against a widely-used reference normalizer |
| `hf_raw` | `Normalised_Transcript` (TIE only) | Minimal | Quantifies dataset normalization errors |
| `hf_clean` | `Normalised_Transcript` (TIE only) | Forward (full) | Dataset norm + our fix |

**Forward normalization** (the `*_clean` modes): Unicode NFC → fix possessives (`"Bernoulli's"` → `"bernoulli s"`) → ordinals/cardinals to words (`"1st"` → `"first"`, `"100"` → `"one hundred"`) → lowercase → strip punctuation → collapse whitespace. Contractions are intentionally **left unexpanded** (`"don't"` → `"dont"`, applied to both sides) so the metric doesn't reward a rewrite neither transcript uses.

The `*_raw` modes apply **minimal cleanup** only — strip wrapping quotes, lowercase, remove punctuation — with no number/possessive handling.

**Why the dataset's `Normalised_Transcript` is unreliable:** it maps `"the 1st component"` → `"the one s t component"` (ordinal split into characters), affecting 50+ clips and inflating `hf_raw` WER by 2–3 pp. Use `transcript_clean`.

---

## Framework architecture

The benchmark is a **generalized multi-dataset framework**: one pipeline, driven by a
central registry, that runs identically on any dataset. Only dataset *loading* is
dataset-specific — everything after Stage 1 is dataset-agnostic.

```
DatasetSpec + ModelSpec (utils/registry.py — single source of truth)
        │
Stage 1  inference driver ──► results/<dataset>/stage1_raw_transcripts/wer_<model>_raw.csv   (GPU; immutable, committed)
        │
Stage 2  normalize_and_score.py --dataset X ──► results/<dataset>/stage2_processed/<mode>/    (CPU; WER + CER + hallucination)
        │
Stage 3  compare_all / statistics / error_analysis / entity_analysis ──► results/<dataset>/analysis/   (CPU)
        │
         paper/figures/make_paper_figures.py ──► paper/figures/   (publication PDFs/SVGs/PNGs)
```

- **`utils/registry.py`** — every model, dataset, evaluation mode, display name and colour. Nothing is defined elsewhere.
- **`utils/datasets.py`** — dataset adapter; validates that a dataset's declared columns exist (catches provisional schemas).
- **Raw transcripts are the immutable source of truth** — always committed, one CSV per (dataset, model). Any normalization/metric change recomputes Stage 2/3 from them with **no re-inference**.

**Add a dataset** → append one `DatasetSpec` to `utils/registry.py` (HF id, column map, subgroup dims, applicable modes). No other file changes.
**Add a model** → append one `ModelSpec` (engine, checkpoint id, arch class, colour) and run its engine driver with `--model`.
**Add a metric** → add it in `utils/wer_compute.py` and surface it in `normalize_and_score.py` / the analysis scripts.

## Reproducing Results

Every Stage-1 run writes a `wer_<model>_manifest.json` beside its raw CSV recording the
model + pinned dataset revision, decode parameters, package versions, git commit, and host.
Decode settings and known nondeterminism are documented in
[`docs/DECODE_CONFIG.md`](docs/DECODE_CONFIG.md) — the committed raw transcripts are the
reproducibility anchor.

**Analysis only (no GPU)** — recompute every table + figure from the committed transcripts:

```bash
git clone https://github.com/theshivam7/indian-asr-bench && cd indian-asr-bench
pip install -r requirements.txt
python normalize_and_score.py --dataset tie      # Stage 2 → results/tie/stage2_processed/
python analysis/compare_all.py --dataset tie     # Stage 3 tables + charts
python analysis/statistics.py --dataset tie      # bootstrap CIs + paired significance
python analysis/error_analysis.py --dataset tie  # codified artifact taxonomy
python paper/figures/make_paper_figures.py --dataset tie
```

**Transcription (GPU)** — registry-driven drivers, `--model` / `--dataset`:

```bash
bash task_whisper/setup.sh                                              # one env for all Whisper models
python task_whisper/run_whisper.py --model large_v3_turbo --dataset tie # → results/tie/stage1_raw_transcripts/
python task4_parakeet/wer_parakeet.py --model parakeet_ctc --dataset tie
python task5_qwen3_asr/wer_qwen3.py --dataset svarah
```

**On a cluster (NSCC / PBS Pro)** — one command submits every remaining experiment with the right parallelism and dependency chaining, printing the job IDs:

```bash
hf auth login                                           # once — Svarah is gated
PROJECT=<nscc_project_id> bash hpc/submit_all.sh        # add --setup to also create the conda envs
```

Or drive pieces individually: `qsub -P <id> -v DATASET=svarah hpc/run_pipeline.pbs` (full run), `qsub -P <id> -v DATASET=tie hpc/job_score.pbs` (CPU-only re-scoring), `qsub -P <id> -v DATASETS=tie,svarah hpc/job_figures.pbs` (combined figures). See [`hpc/README.md`](hpc/README.md).

**Fine-tuning (GPU)** — official split + speaker-disjoint (3 seeds):

```bash
bash task6_whisper_medium_ft/setup.sh
python task6_whisper_medium_ft/finetune.py                                        # → models/whisper_medium_ft/
FT_SPEAKER_DISJOINT=1 FT_OUTPUT_DIR=models/whisper_medium_ft_disjoint \
    python task6_whisper_medium_ft/finetune.py                                    # speaker-disjoint, seed 42
FT_SPEAKER_DISJOINT=1 FT_SEED=43 FT_OUTPUT_DIR=models/whisper_medium_ft_disjoint_s43 \
    python task6_whisper_medium_ft/finetune.py                                    # + seeds 43, 44 (multi-seed study)
FT_SIZE_MATCHED=567 FT_SEED=42 FT_OUTPUT_DIR=models/whisper_medium_ft_sizematch_s42 \
    python task6_whisper_medium_ft/finetune.py                                    # size-matched control (x3 seeds)
MODEL_NAME=medium_hf python task6_whisper_medium_ft/wer_whisper_medium_ft.py
MODEL_NAME=medium_ft python task6_whisper_medium_ft/wer_whisper_medium_ft.py
```

Or on the cluster: `bash hpc/submit_all.sh --phase seeds` submits both extra-seed jobs and chains a single rescore job after both finish (avoids two jobs racing on the shared results files); `qsub -v SEED=42 hpc/job_finetune_sizematch.pbs` (×3 seeds) runs the size-matched control.

Conda specs in [`environments/`](environments/); PBS jobs + the `submit_all.sh` one-shot submitter in [`hpc/`](hpc/)
(`job_whisper`, `job_parakeet`, `job_qwen3`, `job_svarah`, `job_new_models_tie`, `job_finetune_disjoint`,
`job_finetune_disjoint_seed`, `job_finetune_sizematch`, `job_score`, `job_figures`). All runs on a single **NVIDIA A100-40GB** (NSCC ASPIRE2A).

---

## Error Analysis

Dataset artifacts (clip/reference misalignment) are classified with a **full-corpus, multi-model consensus classifier** — every clip, not just a hand-reviewed tail — using per-clip reference-word recall and hypothesis/reference length ratio, averaged across all models. Clips with **<4-word references are excluded as unclassifiable** (`short_ref`): with an *n*-word reference, recall is quantized to multiples of 1/*n* and one wrong word crosses either threshold, so the signals carry no information there. Full analysis with evidence: [`results/tie/analysis/error_analysis_transcript_clean.md`](results/tie/analysis/error_analysis_transcript_clean.md) (TIE) and [`results/svarah/analysis/error_analysis_transcript_clean.md`](results/svarah/analysis/error_analysis_transcript_clean.md) (Svarah).

**Reference artifacts are rare in both corpora; what dominates each dataset's tail differs:**

| | TIE_shorts | Svarah |
|---|:---:|:---:|
| Artifact share (classifiable clips, refs ≥4 words) | **1.0%** (95% CI 0.6–1.9%) | **0.8%** (95% CI 0.6–1.0%) |
| Short-reference (<4 words) share of corpus | 0.1% (1 clip) | **23.0%** (1,530 clips) |
| Worst-20-per-model tail: artifacts | **62%** (95% CI 47–75%) | 4% |
| Worst-20-per-model tail: short-ref clips | 0% | **95%** |
| Per-model WER inflation from artifacts | ≈0.53–0.58 pp | ≈0.29–0.36 pp |

The original hand-analysis figure (**~70% of the worst-20 samples are dataset artifacts**) survives as TIE's **tail** statistic — it was never wrong, but reporting it as if it applied to the whole corpus would be. On Svarah the curated dataset is indeed cleaner (0.8% vs 1.0%), **but a naive run of the same classifier reports 4.4%** — an instrument artifact, not a data artifact: Svarah's isolated-word items (sub-second clips like "cat", "jump") auto-flag on any single-word miss, yet on those clips the models *disagree with each other* (inter-hypothesis distance 0.89 vs 0.17 on TIE's true artifacts) — the signature of genuinely hard decontextualized words, not reference faults. **An artifact classifier tuned on one dataset design does not transfer to another unaudited** — the paper's evaluation-validity thesis applied to its own instrument.

**Two independent lines of evidence that TIE's flagged clips are reference errors, not model errors:**

1. **Clip over-run** — the model transcribes the reference correctly **plus** real speech the clip cut off. Proof: a CTC model (Parakeet, which structurally cannot hallucinate), an LLM (Qwen3), and Whisper all emit the *same* extra words on these clips — real audio the reference omitted, not a hallucination. Example (TIE, `-2aOCNaOiLs`): REF "considered in problem forty five" → every model outputs "…forty five **let us do that**" (80% WER, model perfect).
2. **Inter-hypothesis agreement** — on flagged clips, models agree with *each other* (mean pairwise hypothesis distance ≈0.17–0.20) far more than they agree with the reference (≈0.98–1.0 WER against it) — architecture-independent evidence the fault is in the reference, since these models share no decoder or training objective.

On Svarah the same check is applied honestly in reverse: its `clip_over_run` flags (14 long-ref clips — reference truncation and disfluency clean-up in spontaneous chunks) show the agree-with-each-other signature (0.19), but its residual `content_mismatch` flags (25 clips) do **not** (0.74) — so Svarah's true reference-fault rate is, if anything, *below* the 0.8% headline. The agreement check acts as a built-in audit on the classifier itself.

**Other patterns (TIE, evidence in the doc):**

- **SLOW speech dominates the tail** (38% of data → 69% of high-WER, 1.8×) — via truncated reference windows on slow, self-correcting delivery, *not* acoustics.
- **Errors are U-shaped by duration** — over-represented at 0–5s (12×) and 60s+ (4×), under-represented in the safe 15–30s bulk.
- **Hallucination is the top genuine failure**; Whisper Large has the most WER>100% clips (9 of its 20) — matching its highest Std Dev.
- **No female speaker** appears in any model's top-20 (weak N, but consistent).

**Implication:** median WER (11.1% for Medium on TIE) is a more honest estimate of typical quality than corpus WER (14.8%) — the ~3.5 pp gap is this rare-but-severe tail. Model *rankings* are essentially unaffected (all models hit the same artifacts equally); only the absolute numbers are inflated by a consistent ≈0.55 pp on TIE (≈0.3 pp on Svarah).

**Classifier validation:** the consensus classifier above is a heuristic, not ground truth. A blind, stratified human-annotation protocol — annotators see only the audio and reference, never the model output or predicted label — is implemented in [`analysis/validation/`](analysis/validation/) to measure its precision/recall against human judgment. See [`PROTOCOL.md`](analysis/validation/PROTOCOL.md) for the methodology; results pending the annotation pass.

---

## About

Built by **Shivam Sharma** (student at **IIT Madras**) during a research internship at **Nanyang Technological University (NTU), Singapore**. The project provides a reproducible, transparently-documented WER benchmark for Indian English speech across both found (lecture) and curated (read-speech) audio — including a deliberately-disclosed negative fine-tuning result — so that both the strong pretrained baselines and the limits of in-domain fine-tuning are visible to other researchers.

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE). The dataset ([raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts)) is under its own license; review the dataset card before use.
