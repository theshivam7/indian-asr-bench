<h1 align="center">Indian-ASR-Bench</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Models-5%20ASR%20Systems-orange" />
  <img src="https://img.shields.io/badge/Dataset-Indian%20English-green" />
  <img src="https://img.shields.io/badge/Samples-986-purple" />
  <a href="https://github.com/theshivam7/indian-asr-bench"><img src="https://img.shields.io/badge/GitHub-theshivam7%2Findian--asr--bench-black?logo=github" /></a>
</p>

<p align="center">
  <b>Benchmarking 5 ASR systems on Indian English academic speech — with rigorous WER analysis across models, regions, speech rates, and normalization strategies.</b>
</p>

<p align="center">
  <a href="#key-results">Results</a> &nbsp;·&nbsp;
  <a href="#dataset">Dataset</a> &nbsp;·&nbsp;
  <a href="#evaluation-modes">Modes</a> &nbsp;·&nbsp;
  <a href="#pipeline-architecture">Pipeline</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick-start</a>
</p>

---

Word Error Rate (WER) evaluation of **5 ASR systems** on Indian English academic lectures from the TIE (Talks in Indian English) dataset, with comprehensive analysis of how normalization choices affect measured WER.

**ASR Systems evaluated:**
- OpenAI Whisper Base (74M parameters)
- OpenAI Whisper Medium (769M parameters)
- OpenAI Whisper Large (~1.5B parameters)
- NVIDIA Parakeet-TDT-0.6B-v2 (600M parameters)
- Qwen3-ASR-1.7B (1.7B parameters)

> **Archived:** YouTube caption evaluation (clip-aligned, 190/986 samples) is preserved in `archived_tasks/youtube_captions/` but excluded from the main benchmark — the methodology differs fundamentally from direct ASR evaluation.

---

## Key Results

**Primary metric: `transcript_clean`** — forward normalization applied symmetrically to both reference and hypothesis using the original `Transcript` column as ground truth (gold standard).

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 | Samples |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|:-------:|
| **Whisper Medium** | **14.72%** | **15.39%** | **10.91%** | 15.92% | 31.58% | 38.46% | 986 |
| Parakeet-TDT-0.6B | 15.54% | 16.70% | 11.63% | 17.50% | 34.38% | 44.12% | 986 |
| Whisper Large | 15.88% | 16.83% | 11.36% | 19.27% | 35.21% | 48.94% | 986 |
| Qwen3-ASR-1.7B | 15.93% | 16.64% | **12.28%** | **15.88%** | **33.85%** | 44.90% | 986 |
| Whisper Base | 17.44% | 18.29% | 13.33% | 16.99% | 38.16% | 50.00% | 986 |

### Key Findings

1. **Whisper Medium (14.72%)** is the best overall model — 1.16 pp ahead of Parakeet and 1.16 pp ahead of Large.
2. **Parakeet-TDT-0.6B (15.54%) beats Whisper Large (15.88%)** — a 0.6B specialized model outperforms a ~1.5B general-purpose model by 0.34 pp.
3. **Qwen3-ASR-1.7B (15.93%)** is essentially tied with Whisper Large, and has the **lowest standard deviation** (15.88%) of any model — most consistent predictions.
4. **Whisper Large underperforms its size** on Indian English, particularly on hard samples — 19.27% std dev vs Medium's 15.92%.
5. **Normalization choice causes a ~10 pp swing** — larger than any inter-model gap.

### Impact of Normalization

| Mode | Base | Medium | Large | Parakeet | Qwen3 |
|------|:----:|:------:|:-----:|:--------:|:-----:|
| `transcript_raw` (no normalization) | 27.95% | 24.14% | 25.62% | 28.09% | 33.16% |
| `transcript_clean` (**gold standard**) | **17.44%** | **14.72%** | **15.88%** | **15.54%** | **15.93%** |
| `hf_raw` (dataset's broken normalization) | 31.76% | 29.83% | 30.95% | 33.85% | 36.36% |
| `hf_clean` (dataset norm + our fix) | 18.00% | 15.73% | 16.91% | 16.34% | 16.87% |

**Key finding:** The dataset's `Normalised_Transcript` column inflates WER by 3.8–5.7 pp due to systematic errors (e.g. `"1st"` → `"one s t"`). The `hf_raw` mode is worse than no normalization at all.

Qwen3's `transcript_raw` (33.16%) is notably high — the model outputs text with heavy punctuation and casing that normalization fully corrects, explaining its competitive `transcript_clean` score.

---

### Breakdown by Speech Rate

| Speech Rate | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:-----------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| FAST | 16.35% | **13.46%** | 13.77% | 14.30% | 14.76% | 413 |
| AVG | 15.89% | **13.41%** | 16.00% | **13.89%** | 14.91% | 199 |
| SLOW | 19.85% | 17.21% | 18.69% | 18.23% | **18.14%** | 373 |

SLOW speech is consistently the hardest condition. Qwen3 and Parakeet lead on SLOW, while Medium dominates FAST and AVG.

### Breakdown by Region

| Region | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| EAST | 16.78% | **13.92%** | 16.94% | 15.42% | 15.42% | 352 |
| NORTH | 17.01% | **14.72%** | 15.08% | 15.98% | 15.55% | 202 |
| SOUTH | 18.27% | **15.27%** | 15.58% | 15.57% | 16.57% | 362 |
| WEST | 17.29% | 15.40% | **14.98%** | **14.76%** | 16.01% | 69 |

Moderate regional variation (~1.5 pp range). Parakeet leads for WEST; Medium leads for EAST, NORTH, SOUTH.

### Breakdown by Gender

| Gender | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| Female | 13.88% | 12.02% | 12.49% | **11.61%** | 13.17% | 58 |
| Male | 17.65% | **14.88%** | 16.09% | 15.78% | 16.09% | 927 |

Parakeet achieves the best female-speaker WER (11.61%). Dataset is 94% male — interpret carefully.

### Breakdown by Discipline

| Discipline | Base | Medium | Large | Parakeet | Qwen3 | Samples |
|:----------:|:----:|:------:|:-----:|:--------:|:-----:|:-------:|
| Engineering | 17.92% | **15.06%** | 16.02% | 16.27% | 16.48% | 691 |
| Non-Engineering | 16.30% | **13.90%** | 15.55% | **13.85%** | 14.63% | 294 |

Non-Engineering lectures are uniformly easier. Parakeet is best for Non-Engineering (13.85%).

### Breakdown by Audio Duration

| Duration | Base | Medium | Large | Parakeet | Qwen3 |
|:--------:|:----:|:------:|:-----:|:--------:|:-----:|
| 0–5s | 25.00% | 25.00% | 25.00% | 40.00% | 30.00% |
| 5–15s | 24.72% | 21.23% | 24.89% | 23.79% | 23.19% |
| **15–30s** | 16.87% | **13.78%** | 14.73% | 14.90% | 15.20% |
| 30–60s | 19.63% | 19.80% | 22.31% | **18.93%** | 20.15% |
| **60s+** | 33.33% | 37.31% | 38.23% | **18.35%** | 20.49% |

**Parakeet and Qwen3 dramatically outperform Whisper on 60s+ clips** — 18–20% vs 37–38%. Whisper hallucinates extensively on very long audio; Parakeet/Qwen3 are far more robust.

15–30s clips are the sweet spot for all models.

---

## Dataset

[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) — 986 samples from the `test` split. NPTEL-style Indian English academic lectures.

| Attribute | Distribution |
|-----------|-------------|
| Gender | Male 94.1% (928), Female 5.9% (58) |
| Speech rate | FAST 41.9% (413), SLOW 37.8% (373), AVG 20.2% (199) |
| Region | SOUTH 36.7% (362), EAST 35.7% (352), NORTH 20.5% (202), WEST 7.0% (69) |
| Discipline | Engineering 70.1% (691), Non-Engineering 29.9% (294) |
| Total reference words | ~52,178 (transcript_clean) |

---

## Evaluation Modes

4 modes covering 2 reference sources × 2 normalization states — all symmetric:

| Mode | Reference Source | Normalization | Purpose |
|------|-----------------|---------------|---------|
| `transcript_raw` | `Transcript` | None | Upper bound baseline |
| `transcript_clean` | `Transcript` | Forward | **Gold standard — paper primary** |
| `hf_raw` | `Normalised_Transcript` | None | Quantifies dataset normalization errors |
| `hf_clean` | `Normalised_Transcript` | Forward | HF normalization + our fix |

### Normalization Pipeline (`*_clean` modes)

Applied **symmetrically** to both reference and hypothesis:

| Step | Example Before | Example After |
|------|----------------|---------------|
| Unicode NFC | encoding artifacts | fixed |
| Expand contractions | `"don't"` | `"do not"` |
| Fix possessives | `"Bernoulli's"` | `"bernoulli s"` |
| Ordinals → words | `"1st"`, `"2nd"` | `"first"`, `"second"` |
| Cardinals → words | `"100"`, `"60,000"` | `"one hundred"`, `"sixty thousand"` |
| Lowercase | `"The Second"` | `"the second"` |
| Remove punctuation | `"hello, world."` | `"hello world"` |
| Normalize whitespace | `"too  many  spaces"` | `"too many spaces"` |

**Why the dataset's `Normalised_Transcript` is wrong:**

```
Original Transcript:    "the 1st component is..."
Normalised_Transcript:  "the one s t component is..."   ← splits "1st" into characters
Our normalization:       "the first component is..."     ← ordinal → word (correct)
```

---

## Pipeline Architecture

### Stage 1 — ASR Transcription (GPU, run once per model)

| Task | Script | Model | Est. Time (A100) |
|------|--------|-------|------------------|
| Task 1 | `task1_whisper_base/wer_whisper_base.py` | Whisper Base | ~12 min |
| Task 2 | `task2_whisper_medium/wer_whisper_medium.py` | Whisper Medium | ~35 min |
| Task 3 | `task3_whisper_large/wer_whisper_large.py` | Whisper Large | ~90 min |
| Task 4 | `task4_parakeet/wer_parakeet.py` | Parakeet-TDT-0.6B-v2 | ~8 min |
| Task 5 | `task5_qwen3_asr/wer_qwen3.py` | Qwen3-ASR-1.7B | ~15 min |

Output: `results/stage1_raw_transcripts/wer_{model}_raw.csv`

Each script is independently resumable via checkpointing — safe to restart after interruption.

### Stage 2 — Normalization + WER (CPU, re-run freely)

```bash
python normalize_and_score.py    # ~1 min, no GPU needed
python analysis/compare_all.py   # charts and breakdowns
```

### NSCC PBS Jobs

```bash
qsub job_parakeet.pbs   # Parakeet (3h walltime, g1 queue)
qsub job_qwen3.pbs      # Qwen3 (5h walltime, g1 queue)
```

---

## Quick Start

```bash
# Stage 1: Whisper (whisper conda env)
conda activate whisper
python task1_whisper_base/wer_whisper_base.py
python task2_whisper_medium/wer_whisper_medium.py
python task3_whisper_large/wer_whisper_large.py

# Stage 1: Parakeet
conda activate parakeet_env
python task4_parakeet/wer_parakeet.py

# Stage 1: Qwen3-ASR
conda activate qwen3_env
python task5_qwen3_asr/wer_qwen3.py

# Stage 2: Normalization + WER + charts (CPU only)
python normalize_and_score.py
python analysis/compare_all.py
```

---

## Results Folder Structure

```
results/
  stage1_raw_transcripts/       ← raw ASR outputs (read-only after run)
    wer_base_raw.csv
    wer_medium_raw.csv
    wer_large_raw.csv
    wer_parakeet_raw.csv
    wer_qwen3_raw.csv
  stage2_processed/             ← WER results per mode
    transcript_raw/
    transcript_clean/           ← gold standard
    hf_raw/
    hf_clean/
    wer_summary_all_models.csv
    wer_summary_all_models.md
    top_20_high_wer_{model}_{mode}.csv
  analysis/
    wer_summary.csv
    summary_report.md
    comparison_by_{region,speech_class,gender,discipline,duration}.csv
    wer_by_model_and_mode.png
    wer_distribution.png
    wer_by_{region,speech_class,duration}.png
```

## Project Structure

```
.
├── utils/
│   ├── normalize.py              # 4-mode normalization pipeline
│   ├── transcribe.py             # Audio processing + Whisper inference
│   ├── wer_compute.py            # Corpus + sample WER computation
│   └── io_helpers.py             # Dataset loading, I/O, checkpointing
├── task1_whisper_base/
├── task2_whisper_medium/
├── task3_whisper_large/
├── task4_parakeet/               # Parakeet-TDT-0.6B-v2
│   └── nemo_asr_environment.yaml # Reference conda env spec
├── task5_qwen3_asr/              # Qwen3-ASR-1.7B
├── archived_tasks/
│   ├── youtube_captions/         # YouTube caption experiment (archived)
│   └── audio_analysis/           # Audio sample analysis (archived)
├── normalize_and_score.py        # Stage 2: Normalization + WER
├── analysis/compare_all.py       # Cross-model charts and breakdowns
├── job_{base,medium,large,parakeet,qwen3}.pbs   # Individual NSCC PBS jobs
├── fix_{parakeet,qwen3}_env.pbs  # NSCC environment setup jobs
├── run_pipeline.pbs              # Full pipeline NSCC job
└── results/
```

## Tech Stack

- Python 3.10
- [openai-whisper](https://github.com/openai/whisper)
- [nemo_toolkit](https://github.com/NVIDIA/NeMo) — Parakeet-TDT
- [qwen-asr](https://github.com/QwenLM/Qwen3-ASR-Toolkit) — Qwen3-ASR
- [jiwer](https://github.com/jitsi/jiwer) — WER computation
- [num2words](https://github.com/savoirfairelinux/num2words) — number normalization
- HuggingFace Datasets, pandas, matplotlib, librosa, torch
