<h1 align="center">Indian-ASR-Bench</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Models-5%20ASR%20Systems-orange" />
  <img src="https://img.shields.io/badge/Samples-986-purple" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <a href="https://huggingface.co/datasets/raianand/TIE_shorts">
    <img src="https://img.shields.io/badge/Dataset-TIE__shorts-yellow?logo=huggingface" />
  </a>
  <a href="https://github.com/theshivam7/indian-asr-bench">
    <img src="https://img.shields.io/badge/GitHub-indian--asr--bench-black?logo=github" />
  </a>
</p>

<p align="center">
  <b>A rigorous WER benchmark for 5 ASR systems on Indian English academic speech,<br>
  with comprehensive analysis across models, regions, speech rates, and normalization strategies.</b>
</p>

<p align="center">
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#setup">Setup</a> &nbsp;·&nbsp;
  <a href="#reproducing-results">Reproduce</a> &nbsp;·&nbsp;
  <a href="#evaluation-methodology">Methodology</a> &nbsp;·&nbsp;
  <a href="#project-structure">Structure</a>
</p>

---

## Motivation

Automatic Speech Recognition benchmarks predominantly cover American and British English. Indian English — spoken by over a billion people with distinct phonological patterns, regional accents, and code-switching — remains severely under-evaluated. Academic lectures in particular combine rapid speech, technical vocabulary, and heavy male-speaker bias, making them a challenging and practically important domain.

This benchmark evaluates **5 state-of-the-art ASR systems** on the [TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) dataset (Talks in Indian English), a curated test set of 986 NPTEL academic lecture clips. We also investigate how normalization choices affect measured WER — a methodological question that is often overlooked but has a larger impact than model selection.

---

## Models

| Model | Parameters | Architecture | Reference |
|-------|:----------:|:------------:|-----------|
| **Whisper Base** | 74M | Encoder-Decoder | [openai/whisper-base](https://huggingface.co/openai/whisper-base) |
| **Whisper Medium** | 769M | Encoder-Decoder | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) |
| **Whisper Large** | ~1.5B | Encoder-Decoder | [openai/whisper-large](https://huggingface.co/openai/whisper-large) |
| **Parakeet-TDT-0.6B-v2** | 600M | CTC + TDT | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) |
| **Qwen3-ASR-1.7B** | 1.7B | LLM-based | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |

---

## Dataset

**[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts)** — 986 samples from the `test` split of the TIE (Talks in Indian English) dataset. NPTEL-style Indian English academic lectures.

| Attribute | Distribution |
|-----------|-------------|
| Gender | Male 94.1% (928), Female 5.9% (58) |
| Speech rate | FAST 41.9% (413), SLOW 37.8% (373), AVG 20.2% (199) |
| Region | SOUTH 36.7% (362), EAST 35.7% (352), NORTH 20.5% (202), WEST 7.0% (69) |
| Discipline | Engineering 70.1% (691), Non-Engineering 29.9% (294) |
| Reference words | ~52,178 (after normalization) |

---

## Results

### Primary Metric: `transcript_clean`

Forward normalization applied symmetrically to both reference and hypothesis, using the original `Transcript` column as ground truth (gold standard — see [Evaluation Methodology](#evaluation-methodology)).

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|
| **Whisper Medium** | **14.72%** | **15.39%** | **10.91%** | 15.92% | 31.58% | 38.46% |
| Parakeet-TDT-0.6B | 15.54% | 16.70% | 11.63% | 17.50% | 34.38% | 44.12% |
| Whisper Large | 15.88% | 16.83% | 11.36% | 19.27% | 35.21% | 48.94% |
| Qwen3-ASR-1.7B | 15.93% | 16.64% | **12.28%** | **15.88%** | **33.85%** | 44.90% |
| Whisper Base | 17.44% | 18.29% | 13.33% | 16.99% | 38.16% | 50.00% |

### Key Findings

1. **Whisper Medium (14.72%) is best overall** — 1.16 pp ahead of Parakeet, 1.16 pp ahead of Large.
2. **Parakeet-TDT-0.6B (15.54%) beats Whisper Large (15.88%)** — a 600M specialized model outperforms a ~1.5B general-purpose model.
3. **Qwen3-ASR-1.7B (15.93%)** ties Whisper Large, and has the **lowest standard deviation** (15.88%) of any model — most consistent predictions.
4. **Whisper Large underperforms on hard samples** — std dev 19.27% vs Medium's 15.92%; Large hallucinates on 75% of top-20 hardest samples.
5. **Parakeet and Qwen3 dominate long audio (60s+)**: 18–20% vs Whisper Large's 38%.
6. **Normalization causes a ~10–17 pp swing** — larger than any inter-model gap.

### Impact of Normalization

| Mode | Base | Medium | Large | Parakeet | Qwen3 |
|------|:----:|:------:|:-----:|:--------:|:-----:|
| `transcript_raw` (no normalization) | 27.95% | 24.14% | 25.62% | 28.09% | 33.16% |
| `transcript_clean` (**gold standard**) | **17.44%** | **14.72%** | **15.88%** | **15.54%** | **15.93%** |
| `hf_raw` (dataset's normalization, broken) | 31.76% | 29.83% | 30.95% | 33.85% | 36.36% |
| `hf_clean` (dataset norm + our fix) | 18.00% | 15.73% | 16.91% | 16.34% | 16.87% |

`hf_raw` is 3.8–5.7 pp **worse** than no normalization at all — the dataset's `Normalised_Transcript` column contains systematic errors (see [Normalization](#normalization-pipeline)).

Qwen3's large raw→clean gap (33.16% → 15.93%) reflects its rich punctuation output, which normalization fully corrects.

### Breakdown by Speech Rate

| Speech Rate | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:-----------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| FAST | 16.35% | **13.46%** | 13.77% | 14.30% | 14.76% | 413 |
| AVG | 15.89% | **13.41%** | 16.00% | 13.89% | 14.91% | 199 |
| SLOW | 19.85% | 17.21% | 18.69% | 18.23% | **18.14%** | 373 |

### Breakdown by Region

| Region | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| EAST | 16.78% | **13.92%** | 16.94% | 15.42% | 15.42% | 352 |
| NORTH | 17.01% | **14.72%** | 15.08% | 15.98% | 15.55% | 202 |
| SOUTH | 18.27% | **15.27%** | 15.58% | 15.57% | 16.57% | 362 |
| WEST | 17.29% | 15.40% | 14.98% | **14.76%** | 16.01% | 69 |

### Breakdown by Audio Duration

| Duration | Base | Medium | Large | Parakeet | Qwen3 |
|:--------:|:----:|:------:|:-----:|:--------:|:-----:|
| 0–5s | 25.00% | 25.00% | 25.00% | 40.00% | 30.00% |
| 5–15s | 24.72% | 21.23% | 24.89% | 23.79% | 23.19% |
| **15–30s** | 16.87% | **13.78%** | 14.73% | 14.90% | 15.20% |
| 30–60s | 19.63% | 19.80% | 22.31% | **18.93%** | 20.15% |
| **60s+** | 33.33% | 37.31% | 38.23% | **18.35%** | 20.49% |

Parakeet and Qwen3 are dramatically more robust on 60s+ clips (18–20% vs 37–38% for Whisper Large). Whisper hallucinates during long pauses; Parakeet-TDT and Qwen3 do not.

### Breakdown by Gender

| Gender | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| Female | 13.88% | 12.02% | 12.49% | **11.61%** | 13.17% | 58 |
| Male | 17.65% | **14.88%** | 16.09% | 15.78% | 16.09% | 927 |

### Breakdown by Discipline

| Discipline | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:----------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| Engineering | 17.92% | **15.06%** | 16.02% | 16.27% | 16.48% | 691 |
| Non-Engineering | 16.30% | 13.90% | 15.55% | **13.85%** | 14.63% | 294 |

### YouTube Captions (Archived)

As a baseline comparison, YouTube auto-captions were evaluated on the 190/986 samples (19.3%) with available English captions, using clip-aligned Jaccard matching. This is archived — the methodology is not directly comparable to the main benchmark.

| Evaluation | Corpus WER | Samples | Notes |
|------------|:----------:|:-------:|-------|
| YouTube captions (clip-aligned) | 51.88% | 190 | Sliding-window Jaccard alignment |
| Whisper Medium (same 190 samples) | 13.67% | 190 | Direct ASR evaluation |

3.8× worse than Whisper Medium on the same samples. Full methodology and results: [`archived_tasks/youtube_captions/`](archived_tasks/youtube_captions/).

---

## Evaluation Methodology

Four evaluation modes covering **2 reference sources × 2 normalization states**, all applied symmetrically (same normalization to both reference and hypothesis):

| Mode | Reference | Normalization | Purpose |
|------|-----------|:-------------:|---------|
| `transcript_raw` | `Transcript` | None | Upper-bound baseline |
| `transcript_clean` | `Transcript` | Forward | **Gold standard — primary metric** |
| `hf_raw` | `Normalised_Transcript` | None | Quantifies dataset normalization errors |
| `hf_clean` | `Normalised_Transcript` | Forward | HF normalization + our fix |

### Normalization Pipeline

Applied to both reference and hypothesis in `*_clean` modes:

| Step | Before | After |
|------|--------|-------|
| Unicode NFC | encoding artifacts | fixed |
| Expand contractions | `"don't"` | `"do not"` |
| Fix possessives | `"Bernoulli's"` | `"bernoulli s"` |
| Ordinals → words | `"1st"`, `"2nd"` | `"first"`, `"second"` |
| Cardinals → words | `"100"`, `"60,000"` | `"one hundred"`, `"sixty thousand"` |
| Lowercase | `"The Second"` | `"the second"` |
| Remove punctuation | `"hello, world."` | `"hello world"` |
| Normalize whitespace | `"too  many  spaces"` | `"too many spaces"` |

### Why the Dataset's `Normalised_Transcript` Is Wrong

```
Original Transcript:    "the 1st component is..."
Normalised_Transcript:  "the one s t component is..."   ← splits ordinal into characters
Our normalization:       "the first component is..."     ← correct
```

This systematic error affects 50+ samples and inflates `hf_raw` WER by 3.8–5.7 pp above even the un-normalized baseline. **Always use `transcript_clean` as the primary metric.**

---

## Setup

### Prerequisites

- Python 3.10
- Conda (Miniconda or Miniforge recommended)
- CUDA-capable GPU (tested on NVIDIA A100; any modern GPU works)

### Stage 2 / Analysis Only (No GPU)

```bash
git clone https://github.com/theshivam7/indian-asr-bench
cd indian-asr-bench
pip install -r requirements.txt
python normalize_and_score.py   # recompute WER from existing stage1 CSVs
python analysis/compare_all.py  # regenerate charts and breakdowns
```

### Stage 1: ASR Transcription (GPU Required)

Each model requires its own conda environment. Run from repo root:

```bash
# Whisper models (all share the same env)
bash task1_whisper_base/setup.sh        # creates 'whisper_base' env
bash task2_whisper_medium/setup.sh      # creates 'whisper_medium' env
bash task3_whisper_large/setup.sh       # creates 'whisper_large' env

# Parakeet-TDT (NeMo framework, CUDA 11.8)
bash task4_parakeet/setup.sh            # creates 'parakeet' env
# Alternative: conda env create -f environments/parakeet.yaml

# Qwen3-ASR
bash task5_qwen3_asr/setup.sh           # creates 'qwen3' env
```

### Environment Files

Pre-built conda environment specs in [`environments/`](environments/):

```
environments/
  whisper.yaml    — Whisper tasks (Base / Medium / Large)
  parakeet.yaml   — Parakeet-TDT-0.6B-v2 (NeMo, CUDA 11.8)
  qwen3.yaml      — Qwen3-ASR-1.7B
```

Usage: `conda env create -f environments/parakeet.yaml`

### HPC / NSCC

PBS job scripts are in [`hpc/`](hpc/). Configure via environment variables:

```bash
export WORKDIR=/path/to/indian-asr-bench
export PBS_PROJECT=your_project_id
export WHISPER_ENV=whisper
export PARAKEET_ENV=parakeet
export QWEN3_ENV=qwen3
export HF_CACHE=/path/to/hf/cache

qsub hpc/job_parakeet.pbs
qsub hpc/job_qwen3.pbs
```

See [`hpc/README.md`](hpc/README.md) for SLURM equivalents and full configuration.

---

## Reproducing Results

### Stage 1 — ASR Transcription

```bash
# From repo root, with the correct conda env active:
conda activate whisper_base
python task1_whisper_base/wer_whisper_base.py

conda activate whisper_medium
python task2_whisper_medium/wer_whisper_medium.py

conda activate whisper_large
python task3_whisper_large/wer_whisper_large.py

conda activate parakeet
python task4_parakeet/wer_parakeet.py

conda activate qwen3
python task5_qwen3_asr/wer_qwen3.py
```

Each script is **resumable** — if interrupted, re-running it picks up from the last checkpoint automatically.

Output: `results/stage1_raw_transcripts/wer_{model}_raw.csv`

### Stage 2 — Normalization + WER

```bash
python normalize_and_score.py
```

No GPU needed. Reads Stage 1 CSVs, applies all 4 evaluation modes, writes per-sample and summary CSVs to `results/stage2_processed/`.

### Stage 3 — Analysis + Charts

```bash
python analysis/compare_all.py
```

Generates breakdowns by region, speech rate, gender, discipline, duration, plus matplotlib charts. Output in `results/analysis/`.

### Hardware

All Stage 1 transcriptions were run on **NVIDIA A100-SXM4-40GB** (NSCC ASPIRE2A). Approximate wall-clock times per model (single GPU):

| Model | A100 Time |
|-------|:---------:|
| Whisper Base | ~12 min |
| Whisper Medium | ~35 min |
| Whisper Large | ~90 min |
| Parakeet-TDT-0.6B (batch=16) | ~8 min |
| Qwen3-ASR-1.7B | ~15 min |

---

## Fine-tuning Whisper Medium

Whisper Medium — the best pretrained model here — is fine-tuned on the dataset's **`train`** split and
re-evaluated on the **same `test` split** through the identical normalization + WER pipeline, so it slots
in as a **6th model** across all 4 evaluation modes and every breakdown.

### Method (best-practice full fine-tuning)

| Aspect | Choice |
|--------|--------|
| Strategy | **Full fine-tuning** (all 769M params) via HuggingFace `transformers` `Seq2SeqTrainer` |
| Splits | Train on `train` (3036), select checkpoint on `validation` (986), evaluate on `test` (986) — **no leakage** |
| Targets | `Transcript` (gold ground truth) |
| Precision | **bf16** (A100-native) + gradient checkpointing (`use_cache=False`) |
| Regularization | SpecAugment, `weight_decay=0.01`, LR `1e-5`, warmup 10% |
| Stopping | Epoch cap 10, **early stopping** (patience 2) on validation WER → guards both under- and over-fitting |
| Selection | `load_best_model_at_end` by validation WER (computed with the **same** normalization as the final metric) |
| Long audio | Clips >30s filtered for **training**; **inference** uses chunked long-form (`chunk_length_s=30`) so long clips are windowed like `openai-whisper` |

### Fair baseline (engine-controlled)

Whisper fine-tuning requires `transformers`, whose checkpoints can't be loaded by `openai-whisper`. To avoid
attributing a decoding/engine difference to fine-tuning, the **pretrained** Whisper Medium is *also*
transcribed through the same `transformers` chunked pipeline (`medium_hf`). The headline comparison is
**`medium_ft` vs `medium_hf`** (same engine); the original `openai-whisper` number is kept as a secondary
reference.

### Run order (NSCC)

```bash
git pull
bash task6_whisper_medium_ft/setup.sh                       # conda env 'whisper_medium_ft'
python task6_whisper_medium_ft/check_speaker_overlap.py      # pre-flight leakage disclosure (CPU)

export WORKDIR=$(pwd) PBS_PROJECT=<id> WHISPER_FT_ENV=whisper_medium_ft HF_CACHE=/scratch/hf_cache

JOBID=$(qsub hpc/job_finetune.pbs)                          # Stage 0: train → models/whisper_medium_ft/
qsub -W depend=afterok:$JOBID hpc/job_medium_ft.pbs         # Stage 1+2+3: transcribe + WER + analysis
```

Outputs: `results/stage1_raw_transcripts/wer_medium_{hf,ft}_raw.csv`, all 4 modes under
`results/stage2_processed/`, updated tables/charts (now with **Medium-HF** and **Medium-FT** columns), and
the dedicated **`results/analysis/finetune_comparison.{md,png}`**.

After the run, push `results/` to GitHub and upload the model to Hugging Face:

```bash
huggingface-cli upload <your-hf-username>/whisper-medium-tie-shorts models/whisper_medium_ft .
```

> Tunable via env vars: `FT_EPOCHS`, `FT_BATCH`, `FT_GRAD_ACCUM`, `FT_LR`, `FT_PATIENCE`.
> Smoke-test the code path with `MAX_TRAIN_SAMPLES=8 FT_EPOCHS=1 python task6_whisper_medium_ft/finetune.py`.

---

## Common Error Patterns

1. **Mathematical notation** — equations like `ds/dt = πr²H` have no standard spoken form; ASR models often misrecognize variable names
2. **SLOW speech hallucinations** — Whisper (especially Large) generates filler text during long pauses; Parakeet and Qwen3 are more robust
3. **Technical vocabulary** — domain terms (`gel permeation chromatography`, `sludge drying beds`) are frequently misrecognized across all models
4. **Very short clips (0–5s)** — a single misrecognized word can push WER to 100%+; all models perform poorly at this duration
5. **Engineering jargon and code-switching** — Hindi/regional language words in otherwise English lectures cause confusion

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, adding new model evaluations, and submitting pull requests.

---

## About

This benchmark was developed by **Shivam Sharma**, a student at **IIT Madras**, during a research internship at **Nanyang Technological University (NTU), Singapore**.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The dataset ([raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts)) is subject to its own license. Please review the dataset card before use.
