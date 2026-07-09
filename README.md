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
</p>

<p align="center">
  <b>A reproducible Word Error Rate benchmark for ASR on Indian English speech:<br>
  two datasets, up to seven models per dataset, five normalization modes, and a fine-tuning capacity study across model sizes.</b>
</p>

<p align="center">
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#fine-tuning-pretrained-vs-fine-tuned-across-sizes">Fine-tuning</a> &nbsp;·&nbsp;
  <a href="#evaluation-methodology">Methodology</a> &nbsp;·&nbsp;
  <a href="#error-analysis">Error&nbsp;Analysis</a> &nbsp;·&nbsp;
  <a href="#reproducing-results">Reproduce</a> &nbsp;·&nbsp;
  <a href="#limitations">Limitations</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

---

## Motivation

ASR benchmarks are dominated by American and British English. Indian English, spoken by over a billion people with distinct phonology, regional accents, and code-switching, is under-evaluated, and academic lectures (rapid speech, technical vocabulary, heavy male-speaker skew) are an especially hard and practically important slice.

This project does four things:

1. It benchmarks up to seven pretrained ASR systems on two datasets: [TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) (986 NPTEL-style lecture clips, "found" YouTube data) and [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (6,656 curated read-speech clips), across five normalization modes and every demographic/acoustic breakdown.
2. It fine-tunes Whisper across three sizes (Tiny 39M, Small 244M, Medium 769M) on the same in-domain TIE data and compares each against its own pretrained baseline through an identical decoding pipeline, to test whether a null fine-tuning result at one size reflects a capacity ceiling or a dataset limitation (see [Fine-tuning](#fine-tuning-pretrained-vs-fine-tuned-across-sizes)).
3. It digs into the failure modes with a full-corpus, multi-model consensus artifact classifier, not just a hand-reviewed tail. Reference artifacts turn out to be rare in both corpora (TIE 1.1%, Svarah 0.8% of classifiable clips) but dominate TIE's worst-WER tail (62%), while Svarah's tail is dominated instead by an isolated-word subtask (23% of its clips have <4-word references) where WER is quantized and the classifier is undefined. See [Error Analysis](#error-analysis).
4. And it checks that classifier against actual human judgment with a blind, stratified annotation protocol, rather than just trusting an unvalidated heuristic (see [`analysis/validation/PROTOCOL.md`](analysis/validation/PROTOCOL.md)).

Two recurring themes: **how you normalize text moves WER as much as which model you pick**, and **the median clip is 3–4 pp better than the corpus WER** because a rare-but-severe tail (reference artifacts on TIE; sub-second isolated-word items on Svarah) inflates the average.

---

## Models

| Model | Parameters | Architecture | Reference |
|-------|:----------:|:------------:|-----------|
| **Whisper Tiny** | 39M | Encoder-Decoder | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| **Whisper Base** | 74M | Encoder-Decoder | [openai/whisper-base](https://huggingface.co/openai/whisper-base) |
| **Whisper Small** | 244M | Encoder-Decoder | [openai/whisper-small](https://huggingface.co/openai/whisper-small) |
| **Whisper Medium** | 769M | Encoder-Decoder | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) |
| **Whisper Large-v3** | ~1.5B | Encoder-Decoder | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) |
| **Whisper large-v3-turbo** | 809M | Encoder-Decoder | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| **Parakeet-TDT-0.6B-v2** | 600M | CTC + TDT | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) |
| **Parakeet-CTC-1.1B** | 1.1B | CTC | [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) |
| **Qwen3-ASR-1.7B** | 1.7B | LLM-based | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| **Whisper Medium (fine-tuned)** | 769M | Encoder-Decoder | [theshivam7/whisper-medium-indian-english](https://huggingface.co/theshivam7/whisper-medium-indian-english) (*this project*) |

The pretrained models are evaluated as-is (the headline benchmark, run on both datasets where the checkpoint applies; Tiny/Small are TIE-only additions for the capacity study, `large_v3_turbo` and `parakeet_ctc` are Svarah-only additions). Fine-tuning is TIE-only and analyzed separately in [Fine-tuning](#fine-tuning-pretrained-vs-fine-tuned-across-sizes).

---

## Datasets

**[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts)**: the `test` split (986 clips) of the TIE (Talks in Indian English) dataset, NPTEL-style academic lecture audio scraped from YouTube ("found" data, no controlled recording protocol). 985 clips are scored (one is excluded for an empty reference).

| Attribute | Distribution |
|-----------|-------------|
| Gender | Male 94.1% (927), Female 5.9% (58) |
| Speech rate | FAST 41.9% (413), SLOW 37.9% (373), AVG 20.2% (199) |
| Region | SOUTH 36.8% (362), EAST 35.7% (352), NORTH 20.5% (202), WEST 7.0% (69) |
| Discipline | Engineering 70.2% (691), Non-Engineering 29.8% (294) |

**[ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)**: the `test` split (6,656 clips), read-speech prompts recorded under a controlled protocol across Indian speakers ("curated" data, the counterpoint to TIE's "found" data). Eval-only (no train/validation split; not used for fine-tuning).

| Attribute | Distribution |
|-----------|-------------|
| Gender | Female 53.8% (3,579), Male 46.2% (3,077) |
| Age | 30–45 40.1% (2,670), 18–30 33.3% (2,219), 45–60 19.6% (1,305), 60+ 6.9% (462) |
| Native language | 19 native languages represented (Assamese, Bengali, Bodo, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, and more); speakers from 65 districts per the [dataset paper](https://arxiv.org/abs/2305.15760) |

---

## Results

### TIE_shorts

All numbers are corpus/per-sample WER on the `test` split under **`transcript_clean`** (the gold-standard mode: forward normalization applied symmetrically to reference and hypothesis; see [Methodology](#evaluation-methodology)). Regenerated directly from `results/`.

#### Primary metric: `transcript_clean`

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|
| **Whisper Medium** | **14.76%** | **15.45%** | **11.11%** | **15.90%** | **31.58%** | **39.62%** |
| Parakeet-TDT-0.6B | 15.60% | 16.75% | 11.86% | 17.47% | 34.38% | 44.12% |
| Whisper Large | 15.93% | 16.88% | 11.43% | 19.20% | 35.21% | 48.94% |
| Whisper Small | 16.05% | 16.90% | 12.20% | 17.78% | 34.38% | 46.00% |
| Qwen3-ASR-1.7B | 16.66% | 17.34% | 12.90% | 15.93% | 35.00% | 45.07% |
| Whisper Base | 17.53% | 18.38% | 13.51% | 16.95% | 38.16% | 50.00% |
| Whisper Tiny | 19.43% | 20.49% | 16.28% | 17.41% | 40.28% | 51.76% |

Is any of this significant? We ran a speaker-clustered paired bootstrap over 280 speakers, Holm-corrected across all 21 pairs (see [`statistics_transcript_clean.md`](results/tie/analysis/statistics_transcript_clean.md)). Whisper Medium's lead over every other model holds up, including over Whisper Large (−1.17 pp, p<sub>Holm</sub>=0.02) and over Whisper Small (−1.29 pp, p<sub>Holm</sub>=0.02) — cases where a smaller model still beats a bigger one. But the middle of the pack is statistically crowded: Small, Large, Parakeet-TDT, and Qwen3 are mutually indistinguishable in most pairings (5 of the 21 pairs aren't significant) — a 244M Whisper, a 1.5B Whisper, a 600M transducer, and a 1.7B LLM landing in the same statistical tier.

> The **fine-tuned** models are not in this ranking because they run through a different decoder (HuggingFace `transformers`, not `openai-whisper`); mixing them in here would confound fine-tuning with a decoding-engine change. Their fair, engine-controlled comparison is in [Fine-tuning](#fine-tuning-pretrained-vs-fine-tuned-across-sizes).

<p align="center">
  <img src="results/tie/analysis/wer_by_model.png" width="680" alt="Model ranking by corpus WER (transcript_clean)">
</p>

#### Key findings

1. **Whisper Medium wins overall** at 14.76% corpus WER, and it's also the most consistent model here (lowest Std Dev, lowest median WER).
2. Parakeet-TDT-0.6B (15.60%) actually beats Whisper Large (15.93%): a 600M specialized model edging out a ~1.5B general-purpose one.
3. **WER falls monotonically with Whisper capacity** (Tiny 19.43% → Base 17.53% → Small 16.05% → Medium 14.76%) but reverses at Large (15.93%) — the biggest model isn't the best one. See [Fine-tuning](#fine-tuning-pretrained-vs-fine-tuned-across-sizes) for what this capacity curve implies about fine-tuning gains.
4. Whisper Large is the *least* stable of the group (Std Dev 19.20%). It hallucinates on the hardest clips more than the others do.
5. Normalization and reference choice alone move WER by roughly 2–3 pp, about as much as the gap between the best and worst mid-tier models (details below).

#### Impact of normalization

| Mode | Base | Medium | Large | Parakeet | Qwen3 |
|------|:----:|:------:|:-----:|:--------:|:-----:|
| `transcript_raw` (minimal cleanup) | 17.91% | 15.11% | 16.31% | 15.97% | 18.15% |
| `transcript_clean` (**gold standard**) | 17.53% | **14.76%** | 15.93% | 15.60% | 16.66% |
| `hf_raw` (dataset's normalization, broken) | 20.24% | 18.01% | 19.14% | 18.54% | 17.99% |
| `hf_clean` (dataset norm + our fix) | 18.07% | 15.76% | 16.94% | 16.40% | 17.61% |

The dataset's own `Normalised_Transcript` (`hf_raw`) is **2.7–3.3 pp worse** than the gold `Transcript` with correct normalization for the four conventional systems: it splits ordinals into characters (`"1st"` → `"one s t"`). Qwen3 is the exception (+1.3 pp): its richly punctuated verbatim output already disagrees with the raw reference's formatting, so the reference bug costs it less, and the bias isn't even uniform across models. **Always use `transcript_clean`.**

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

Parakeet and Qwen3 are far more robust on 60s+ clips: Whisper hallucinates during long pauses, while the TDT/LLM decoders do not. (The extreme buckets are tiny: n=4 for 0–5s, n=5 for 60s+, so a single clip moves them by 20+ pp; 87% of clips sit in 15–30s. Read the extremes qualitatively.)

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

YouTube auto-captions, evaluated on the 190 clips (19.3%) with available English captions via clip-aligned Jaccard matching, score **51.88% WER**: 3.8× worse than Whisper Medium on the same clips (13.67%). Not directly comparable to the main benchmark; kept for reference in [`archived_tasks/youtube_captions/`](archived_tasks/youtube_captions/).

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

Same check on Svarah, this time recording-clustered rather than speaker-clustered (the public release doesn't expose speaker IDs, so the recording tag embedded in each filename is the strictest unit available): paired bootstrap over 3,232 recordings, Holm-corrected across all 21 pairs (see [`statistics_transcript_clean.md`](results/svarah/analysis/statistics_transcript_clean.md)). 19 of the 21 pairwise differences come out significant. The two that don't: Whisper Medium vs. large-v3-turbo, and Parakeet-TDT vs. Qwen3.

<p align="center">
  <img src="results/svarah/analysis/wer_by_model.png" width="680" alt="Model ranking by corpus WER on Svarah (transcript_clean)">
</p>

**Key findings:**

1. Whisper Large takes the top spot on Svarah at 7.11%, roughly half TIE's error rate. That gap makes sense: Svarah is controlled read speech, while TIE is noisier scraped lecture audio.
2. Normalization matters a lot more here than it did on TIE, at least for the CTC/TDT/LLM models. Parakeet-TDT drops from 13.03% (`raw`) to 8.35% (`whisper_norm`), a 4.7pp swing. Why: Parakeet and Qwen3 transcribe fillers verbatim ("and uh", "mm hmm") on Svarah's spontaneous portions and spell digits differently on its read prompts. `transcript_clean` penalizes that faithful transcription, while `whisper_norm` strips the fillers and unifies numerals. Whisper models omit fillers by training anyway, so they barely move (7.89% → 7.69%).
3. The curated dataset really is cleaner than TIE, but only once you fix the classifier first. Among classifiable clips (references ≥4 words), Svarah's artifact share is 0.8% against TIE's 1.1%. Run the same classifier naively, though, and it reports 4.4% on Svarah: that's an artifact of the instrument, not the data. 23% of Svarah's clips are 1–2-word isolated-word items ("cat", "jump", sub-second audio), and any single missed word saturates recall and auto-flags the clip. On those flagged short clips the models actually disagree with each other (inter-hypothesis distance 0.89) rather than agreeing with each other and disagreeing with the reference, which is the opposite of what a true reference fault looks like: it's the signature of genuinely hard, decontextualized words ("tree"→"three", "left"→"lift"). The lesson: heuristic artifact detectors don't transfer across dataset designs unless you audit them first.

---

## Fine-tuning: pretrained vs. fine-tuned, across sizes

The Medium fine-tune (the strongest pretrained model here) came back with **no significant
gain** on the official TIE split. That raises a natural question: is Medium (769M) already
saturated by what 46.9h of TIE data can teach it (a **capacity ceiling**), or can the dataset
not support a fine-tuning gain at *any* model size? To find out, we ran the same protocol —
one official-split fine-tune, evaluated against its own pretrained baseline through an
identical decoding pipeline — on Whisper Tiny (39M) and Small (244M) as well.

**Setup:** Medium uses a full fine-tune (all 769M params) via `transformers` `Seq2SeqTrainer`
— bf16, epoch-based, early stopping on validation WER
([`finetune_medium.py`](finetune/finetune_medium.py)). Tiny/Small use an adaptation of a
step-based recipe (`max_steps=2000`, effective batch 32, fp16, best-checkpoint selection)
([`finetune_tiny_small.py`](finetune/finetune_tiny_small.py)) — a disclosed recipe
difference, not a bug; see the linked findings report for the full list of deltas. All three
compare fine-tuned vs. pretrained **decoded through the identical HF pipeline**, so the
comparison isolates fine-tuning from any decoding-engine effect.

Engine-controlled, `transcript_clean`, paired speaker-clustered bootstrap over 280 speakers,
Holm-corrected across this 3-test family (kept separate from the pretrained-model families in
[Results](#results) — those cover pretrained models only, and mixing in a different decoding
engine would confound fine-tuning with an engine change):

| Size | Params | Pretrained (HF) | Fine-tuned | Δ (paired) | 95% CI | p (Holm) |
|------|:------:|:---:|:---:|:---:|:---:|:---:|
| Whisper Tiny | 39M | 22.10% | 19.14% | **−2.96 pp** | [−6.35, +0.13] | 0.195 |
| Whisper Small | 244M | 17.38% | 16.21% | **−1.17 pp** | [−3.97, +1.21] | 0.774 |
| Whisper Medium | 769M | 14.42% | 14.61% | +0.20 pp | [−0.46, +1.03] | 0.774 |

**A capacity gradient, but not (yet) a significant one.** The point estimates trace exactly
the shape the capacity-ceiling hypothesis predicts — gain shrinks monotonically as capacity
grows, flipping to null at 769M — but none of the three deltas survives Holm correction at
985 test clips; the study is underpowered to confirm a ~3 pp effect at that sample size alone.

**Read past the headline number.** For both sizes, more individual clips got *worse* after
fine-tuning than got better (Tiny: 313 improved vs. 326 regressed; Small: 263 vs. 403). The
net corpus-level gain is driven largely by fine-tuning fixing a handful of severe
repetition-loop decoding failures — one Tiny sample dropped from 977.8% WER to 55.6% — not a
broad, uniform content-adaptation improvement; fine-tuning also *introduces* the same
pathology elsewhere (one Small sample went from 66.2% to 445.1%). Both training runs showed a
healthy learn-then-overfit trajectory (Tiny best checkpoint at step 600/2000, Small at
step 800/2000), which rules out a degenerate no-learning explanation.

**Why absolute WER stays elevated (14–22%) even after fine-tuning:** mostly a domain-difficulty
floor, not a fine-tuning shortfall. TIE is noisy, scraped lecture audio — Whisper Large scores
15.93% on TIE vs. **7.11% on the controlled-protocol Svarah benchmark**, literally half — and
that gap holds for every model regardless of fine-tuning.

> ⚠️ **Speaker overlap (disclosed).** TIE's official splits share speakers: **100% of test
> speakers, and 100% of test clips, come from speakers also seen in training**
> ([`speaker_overlap.md`](results/tie/analysis/speaker_overlap.md), via
> [`check_speaker_overlap.py`](finetune/check_speaker_overlap.py)). There is
> **no clip-level leakage**, but the comparison is *speaker-matched*, so any gain above partly
> reflects speaker adaptation rather than purely accent/content adaptation.

> **Long-clip decoding note.** On 60s+ clips the HF chunked pipeline scores much higher WER
> than the same weights under `openai-whisper` (long-form chunk stitching) — a decoding-pipeline
> artifact, not a fine-tuning effect. It hits pretrained and fine-tuned equally, so the
> head-to-head above stays fair, but it does inflate Std Dev/tail metrics for HF-pipeline runs.

**Bottom line:** directionally supportive of the capacity-ceiling explanation for Medium's
null result, but not conclusive on its own — 985 test clips is not enough to statistically
confirm a ~3pp effect after correcting for three comparisons, and part of the observed gain is
decoding-pathology-fixing rather than uniform improvement. Full methodology, training
trajectories, and per-sample breakdowns:
[`findings_tiny_small_ft.md`](results/tie/analysis/findings_tiny_small_ft.md),
[`finetune_comparison.md`](results/tie/analysis/finetune_comparison.md) (Medium),
[`finetune_comparison_small.md`](results/tie/analysis/finetune_comparison_small.md),
[`finetune_comparison_tiny.md`](results/tie/analysis/finetune_comparison_tiny.md).

<p align="center">
  <img src="results/tie/analysis/finetune_comparison.png" width="640" alt="Whisper Medium pretrained vs fine-tuned across all five modes">
</p>

---

## Evaluation Methodology

Five modes are available; which ones apply depends on the dataset's schema (TIE has both a gold and a dataset-provided reference, so all five apply; Svarah has only a gold reference, so only `transcript_raw`/`transcript_clean`/`whisper_norm` apply). All normalization is applied **symmetrically** to reference and hypothesis:

| Mode | Reference | Normalization | Purpose |
|------|-----------|:-------------:|---------|
| `transcript_raw` | gold (`Transcript` / `text`) | Minimal (case/punct/quotes) | Near-upper-bound baseline |
| `transcript_clean` | gold (`Transcript` / `text`) | Forward (full) | **Gold standard, primary metric** |
| `whisper_norm` | gold (`Transcript` / `text`) | OpenAI's `EnglishTextNormalizer` | Cross-checks the primary metric against a widely-used reference normalizer |
| `hf_raw` | `Normalised_Transcript` (TIE only) | Minimal | Quantifies dataset normalization errors |
| `hf_clean` | `Normalised_Transcript` (TIE only) | Forward (full) | Dataset norm + our fix |

**Forward normalization** (the `*_clean` modes): Unicode NFC → fix possessives (`"Bernoulli's"` → `"bernoulli s"`) → ordinals/cardinals to words (`"1st"` → `"first"`, `"100"` → `"one hundred"`) → lowercase → strip punctuation → collapse whitespace. Contractions are intentionally **left unexpanded** (`"don't"` → `"dont"`, applied to both sides) so the metric doesn't reward a rewrite neither transcript uses.

The `*_raw` modes apply **minimal cleanup** only (strip wrapping quotes, lowercase, remove punctuation), with no number/possessive handling.

**Why the dataset's `Normalised_Transcript` is unreliable:** it maps `"the 1st component"` → `"the one s t component"` (ordinal split into characters), affecting 50+ clips and inflating `hf_raw` WER by 2–3 pp. Use `transcript_clean`.

---

## Framework architecture

The benchmark is a **generalized multi-dataset framework**: one pipeline, driven by a
central registry, that runs identically on any dataset. Only dataset *loading* is
dataset-specific; everything after Stage 1 is dataset-agnostic.

```
DatasetSpec + ModelSpec (utils/registry.py, single source of truth)
        │
Stage 1  inference driver ──► results/<dataset>/stage1_raw_transcripts/wer_<model>_raw.csv   (GPU; immutable, committed)
        │
Stage 2  normalize_and_score.py --dataset X ──► results/<dataset>/stage2_processed/<mode>/    (CPU; WER + CER + hallucination)
        │
Stage 3  compare_all / statistics / error_analysis / entity_analysis ──► results/<dataset>/analysis/   (CPU)
        │
         paper figures (local paper workspace, not shipped) ──► paper/figures/
```

- `utils/registry.py` holds every model, dataset, evaluation mode, display name and colour. Nothing is defined anywhere else.
- `utils/datasets.py` is the dataset adapter: it validates that a dataset's declared columns actually exist, which catches provisional schemas early.
- Raw transcripts are the immutable source of truth: always committed, one CSV per (dataset, model). Any normalization or metric change recomputes Stage 2/3 straight from them, with no re-inference needed.

**Add a dataset** → append one `DatasetSpec` to `utils/registry.py` (HF id, column map, subgroup dims, applicable modes). No other file changes.
**Add a model** → append one `ModelSpec` (engine, checkpoint id, arch class, colour) and run its engine driver with `--model`.
**Add a metric** → add it in `utils/wer_compute.py` and surface it in `normalize_and_score.py` / the analysis scripts.

## Reproducing Results

Every Stage-1 run writes a `wer_<model>_manifest.json` beside its raw CSV recording the
model + pinned dataset revision, decode parameters, package versions, git commit, and host.
Decode settings and known nondeterminism are documented in
[`docs/DECODE_CONFIG.md`](docs/DECODE_CONFIG.md); the committed raw transcripts are the
reproducibility anchor.

**Analysis only (no GPU)**: recompute every table + figure from the committed transcripts:

```bash
git clone https://github.com/theshivam7/indian-asr-bench && cd indian-asr-bench
pip install -r requirements.txt
python normalize_and_score.py --dataset tie      # Stage 2 → results/tie/stage2_processed/
python analysis/compare_all.py --dataset tie     # Stage 3 tables + charts
python analysis/statistics.py --dataset tie      # cluster-bootstrap CIs + Holm-corrected paired tests
python analysis/error_analysis.py --dataset tie  # codified artifact taxonomy (+ instrument audit)
python analysis/compare_finetune.py              # fine-tuning report (TIE)
```

(Repeat with `--dataset svarah` for the second corpus. The publication-figure script lives in the local paper workspace and is not part of the shipped pipeline; every table and chart above regenerates from the committed transcripts alone.)

**Transcription (GPU)**: registry-driven drivers, `--model` / `--dataset`:

```bash
bash whisper_asr/setup.sh                                              # one env for all Whisper models
python whisper_asr/run_whisper.py --model large_v3_turbo --dataset tie # → results/tie/stage1_raw_transcripts/
python parakeet/wer_parakeet.py --model parakeet_ctc --dataset tie
python qwen3/wer_qwen3.py --dataset svarah
```

**On a cluster (NSCC / PBS Pro)**: one command submits every remaining experiment with the right parallelism and dependency chaining, printing the job IDs:

```bash
hf auth login                                           # once, Svarah is gated
PROJECT=<nscc_project_id> bash hpc/submit_all.sh        # add --setup to also create the conda envs
```

Or drive pieces individually: `qsub -P <id> -v DATASET=svarah hpc/run_pipeline.pbs` (full run), `qsub -P <id> -v DATASET=tie hpc/job_score.pbs` (CPU-only re-scoring), `qsub -P <id> -v DATASETS=tie,svarah hpc/job_figures.pbs` (combined figures). See [`hpc/README.md`](hpc/README.md).

**Fine-tuning (GPU)**: official split, per model size:

```bash
bash finetune/setup.sh
python finetune/finetune_medium.py                                    # Medium → models/whisper_medium_ft/
MODEL_NAME=medium_hf python finetune/evaluate_finetuned.py
MODEL_NAME=medium_ft python finetune/evaluate_finetuned.py

python finetune/finetune_tiny_small.py \
    --base-model openai/whisper-tiny --output-dir models/whisper_tiny_ft   # Tiny (or --base-model openai/whisper-small)
MODEL_NAME=tiny_hf python finetune/evaluate_finetuned.py
MODEL_NAME=tiny_ft MODEL_SOURCE=models/whisper_tiny_ft python finetune/evaluate_finetuned.py
```

Or on the cluster: `qsub -v SIZE=tiny hpc/job_finetune_size.pbs` (or `SIZE=small`) runs the Tiny/Small capacity-study fine-tune end to end.

Conda specs in [`environments/`](environments/); PBS jobs + the `submit_all.sh` one-shot submitter in [`hpc/`](hpc/)
(`job_whisper`, `job_parakeet`, `job_qwen3`, `job_svarah`, `job_new_models_tie`, `job_finetune`,
`job_medium_ft`, `job_finetune_size`, `job_score`, `job_figures`). All runs on a single **NVIDIA A100-40GB** (NSCC ASPIRE2A).

---

## Error Analysis

Dataset artifacts (clip/reference misalignment) are classified with a **full-corpus, multi-model consensus classifier** (every clip, not just a hand-reviewed tail) using per-clip reference-word recall and hypothesis/reference length ratio, averaged across all models. Clips with **<4-word references are excluded as unclassifiable** (`short_ref`): with an *n*-word reference, recall is quantized to multiples of 1/*n* and one wrong word crosses either threshold, so the signals carry no information there. Full analysis with evidence: [`results/tie/analysis/error_analysis_transcript_clean.md`](results/tie/analysis/error_analysis_transcript_clean.md) (TIE) and [`results/svarah/analysis/error_analysis_transcript_clean.md`](results/svarah/analysis/error_analysis_transcript_clean.md) (Svarah).

**Reference artifacts are rare in both corpora; what dominates each dataset's tail differs:**

| | TIE_shorts | Svarah |
|---|:---:|:---:|
| Artifact share (classifiable clips, refs ≥4 words) | **1.1%** (95% CI 0.6–2.0%) | **0.8%** (95% CI 0.6–1.0%) |
| Short-reference (<4 words) share of corpus | 0.1% (1 clip) | **23.0%** (1,530 clips) |
| Worst-20-per-model tail: artifacts | **62%** (95% CI 47–74%) | 4% |
| Worst-20-per-model tail: short-ref clips | 0% | **95%** |
| Per-model WER inflation from artifacts | ≈0.53–0.60 pp | ≈0.29–0.36 pp |

The original hand-analysis figure, ~70% of the worst-20 samples being dataset artifacts, still holds up as TIE's tail statistic. It was never wrong; it's just wrong to report it as if it applied to the whole corpus. On Svarah the curated dataset really is cleaner (0.8% vs 1.1%), but run the same classifier without care and it reports 4.4%. That's an instrument artifact, not a data artifact: Svarah's isolated-word items (sub-second clips like "cat", "jump") auto-flag on any single-word miss, yet on those clips the models disagree with each other (inter-hypothesis distance 0.89 vs 0.17 on TIE's true artifacts), which is the signature of genuinely hard decontextualized words, not reference faults. It's a fitting demonstration of the paper's whole thesis, applied to its own instrument: an artifact classifier tuned on one dataset design doesn't transfer to another unless you audit it first.

**Two independent lines of evidence that TIE's flagged clips are reference errors, not model errors:**

1. **Clip over-run**: the model transcribes the reference correctly **plus** real speech the clip cut off. Proof: a CTC model (Parakeet, which structurally cannot hallucinate), an LLM (Qwen3), and Whisper all emit the *same* extra words on these clips, real audio the reference omitted, not a hallucination. Example (TIE, `-2aOCNaOiLs`): REF "considered in problem forty five" → every model outputs "…forty five **let us do that**" (80% WER, model perfect).
2. **Inter-hypothesis agreement**: on flagged clips, models agree with *each other* (mean pairwise hypothesis distance ≈0.17–0.20) far more than they agree with the reference (≈0.98–1.0 WER against it), architecture-independent evidence the fault is in the reference, since these models share no decoder or training objective.

On Svarah the same check is applied honestly in reverse: its `clip_over_run` flags (14 long-ref clips, reference truncation and disfluency clean-up in spontaneous chunks) show the agree-with-each-other signature (0.19), but its residual `content_mismatch` flags (25 clips) do **not** (0.74), so Svarah's true reference-fault rate is, if anything, *below* the 0.8% headline. The agreement check acts as a built-in audit on the classifier itself.

**Other patterns (TIE, evidence in the doc):**

- SLOW speech dominates the tail: 38% of the data but 69% of the high-WER clips (1.8×). This comes from truncated reference windows on slow, self-correcting delivery, not from worse acoustics.
- Errors are U-shaped by duration: over-represented at 0–5s (12×) and 60s+ (4×), under-represented in the safe 15–30s middle.
- Hallucination is the biggest genuine failure mode. Whisper Large has the most WER>100% clips of any model (9 of its 20), which lines up with it also having the highest Std Dev.
- No female speaker shows up in any model's top-20 worst clips. The sample is small, but the pattern is consistent across models.

**Implication:** median WER (11.1% for Medium on TIE) is a more honest estimate of typical quality than corpus WER (14.8%); the ~3.5 pp gap is this rare-but-severe tail. Model *rankings* are essentially unaffected (all models hit the same artifacts equally); only the absolute numbers are inflated by a consistent ≈0.55 pp on TIE (≈0.3 pp on Svarah).

**Classifier validation:** the consensus classifier above is a heuristic, not ground truth. A blind, stratified human-annotation protocol (annotators see only the audio and reference, never the model output or predicted label) is implemented in [`analysis/validation/`](analysis/validation/) to measure its precision/recall against human judgment. See [`PROTOCOL.md`](analysis/validation/PROTOCOL.md) for the methodology; results pending the annotation pass.

---

## Limitations

Stated so the numbers above are read correctly:

- The artifact classifier hasn't been validated against human judgment yet. It's backed by inter-hypothesis agreement evidence and manual reading of flagged clips, but the blind human-annotation pass still needs to run.
- Svarah can only be clustered by recording, not by speaker, since the public release doesn't expose speaker IDs. That's 3,232 recording clusters versus the 117 true speakers in the dataset paper, and true speaker clustering would widen the confidence intervals further. TIE doesn't have this problem: its clusters are real speakers.
- The fine-tuning study is single-seed per size (one official-split run each, no multi-seed replication), and none of the three size-vs-pretrained deltas survives Holm correction at 985 test clips — read the capacity gradient (Tiny > Small > Medium) as suggestive, not statistically confirmed. Per-sample analysis also shows the net gain is partly driven by fixing a handful of severe repetition-loop decoding failures rather than a uniform improvement — see [Fine-tuning](#fine-tuning-pretrained-vs-fine-tuned-across-sizes).
- Training-data contamination is possible: NPTEL lectures are public and may already appear in Whisper's web-scraped training data. A small probe (grounded vs. free-decoding agreement with flawed references) turned up no memorization signal, but with n=10 it's a low-powered check.
- Stage-1 transcripts are single runs with temperature-fallback decoding ([`docs/DECODE_CONFIG.md`](docs/DECODE_CONFIG.md)). The committed raw CSVs, not re-decoding, are the reproducibility anchor.
- Some cells are just small: duration extremes sit at n=4–5, and TIE has only 58 female speakers. Read those numbers qualitatively, not as precise estimates.

---

## Citation

If you use this benchmark, the analyses, or the fine-tuned checkpoints, please cite:

```bibtex
@misc{sharma2026indianasrbench,
  title        = {Indian-ASR-Bench: Auditing WER Benchmarks for Indian-English ASR},
  author       = {Sharma, Shivam and Liu, Changsong},
  year         = {2026},
  howpublished = {\url{https://github.com/theshivam7/indian-asr-bench}},
}
```

Please also cite the datasets you use: [TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) and [Svarah](https://arxiv.org/abs/2305.15760) (Javed et al., Interspeech 2023).

---

## About

Built by **Shivam Sharma** (student at **IIT Madras**) during a research internship at **Nanyang Technological University (NTU), Singapore**. The project provides a reproducible, transparently-documented WER benchmark for Indian English speech across both found (lecture) and curated (read-speech) audio, including a deliberately-disclosed negative fine-tuning result, so that both the strong pretrained baselines and the limits of in-domain fine-tuning are visible to other researchers.

**Acknowledgements:** computing resources were provided by NSCC Singapore (ASPIRE2A, NVIDIA A100-40GB).

Contributions welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT: see [LICENSE](LICENSE). The datasets ([raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts), [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)) are under their own licenses; review the dataset cards before use.
