<h1 align="center">Indian-asr-bench</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Whisper-OpenAI-black?logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-Indian%20English-green" />
  <a href="https://github.com/theshivam7/indian-asr-bench"><img src="https://img.shields.io/badge/GitHub-theshivam7%2Findian--asr--bench-black?logo=github" /></a>
</p>

<p align="center">
  <b>Benchmarking ASR systems on Indian English academic speech — with rigorous WER analysis across models, regions, and normalization strategies.</b>
</p>

<p align="center">
  <a href="#key-results">Results</a> &nbsp;·&nbsp;
  <a href="#dataset">Dataset</a> &nbsp;·&nbsp;
  <a href="#pipeline-architecture">Pipeline</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick-start</a>
</p>

---

Word Error Rate (WER) evaluation of **5 ASR systems** on Indian English academic lectures from the TIE (Talks in Indian English) dataset, with a comprehensive analysis of how normalization choices affect measured WER.

**ASR Systems evaluated:**
- OpenAI Whisper Base (74M parameters) — completed
- OpenAI Whisper Medium (769M parameters) — completed, best Whisper result
- OpenAI Whisper Large (~1.5B parameters) — completed
- NVIDIA Parakeet-TDT-0.6B-v2 — **results pending (NSCC run)**
- Qwen3-ASR-1.7B — **results pending (NSCC run)**

> **Archived:** YouTube caption evaluation (clip-aligned, 190/986 samples) is preserved in `archived_tasks/youtube_captions/` but excluded from the main benchmark — the methodology differs fundamentally from direct ASR evaluation.

## Key Results

**Primary evaluation metric: `transcript_clean` mode** (gold standard — forward normalization applied symmetrically to both reference and hypothesis using original `Transcript` as ground truth)

| Model | Corpus WER | Mean WER | Median WER | Std Dev | Samples |
|-------|:----------:|:--------:|:----------:|:-------:|:-------:|
| **Whisper Medium** | **14.72%** | **15.39%** | **10.91%** | **15.92%** | 986 |
| Whisper Large | 15.88% | 16.83% | 11.36% | 19.27% | 986 |
| Whisper Base | 17.44% | 18.29% | 13.33% | 16.99% | 986 |
| Parakeet-TDT-0.6B | — | — | — | — | pending |
| Qwen3-ASR-1.7B | — | — | — | — | pending |

**Whisper Medium outperforms Whisper Large** — Large hallucinates on 75% of hard samples vs Medium's 40%, due to overconfidence on Indian-accented speech.

### Impact of Normalization

| Mode | Base | Medium | Large | Parakeet | Qwen3 |
|------|:----:|:------:|:-----:|:--------:|:-----:|
| `transcript_raw` (no normalization) | 27.95% | 24.14% | 25.62% | — | — |
| `transcript_clean` (gold standard) | **17.44%** | **14.72%** | **15.88%** | — | — |
| `hf_raw` (dataset's normalization, broken) | 31.76% | 29.83% | 30.95% | — | — |
| `hf_clean` (dataset normalization + our fix) | 18.00% | 15.73% | 16.91% | — | — |

**Key finding:** Normalization causes a ~10 pp swing for Whisper — larger than any model difference.

The dataset's `Normalised_Transcript` inflates WER by 3.8–5.7 pp due to systematic errors (e.g. `"1st"` → `"one s t"`).

### Breakdown by Speech Rate (transcript_clean, all 986 samples)

| Speech Rate | Base | Medium | Large | Samples |
|:-----------:|:----:|:------:|:-----:|:-------:|
| FAST | 16.35% | **13.46%** | 13.77% | 413 |
| AVG | 15.89% | **13.41%** | 16.00% | 199 |
| SLOW | 19.85% | **17.21%** | 18.69% | 373 |

### Breakdown by Region (transcript_clean, all 986 samples)

| Region | Base | Medium | Large | Samples |
|:------:|:----:|:------:|:-----:|:-------:|
| EAST | 16.78% | **13.92%** | 16.94% | 352 |
| NORTH | 17.01% | **14.72%** | 15.08% | 202 |
| SOUTH | 18.27% | **15.27%** | 15.58% | 362 |
| WEST | 17.29% | **15.40%** | 14.98% | 69 |

### Breakdown by Audio Duration (transcript_clean, all 986 samples)

| Duration | Base | Medium | Large |
|:--------:|:----:|:------:|:-----:|
| 0–5s | 25.0% | 25.0% | 25.0% |
| 5–15s | 24.72% | 21.23% | 24.89% |
| **15–30s** | **16.87%** | **13.78%** | **14.73%** |
| 30–60s | 19.63% | 19.80% | 22.31% |
| 60s+ | 33.33% | 37.31% | 38.23% |

15–30s clips show best performance. 60s+ clips degrade sharply for all models.

---

## Dataset

[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) — 986 samples from the `test` split. NPTEL-style Indian English academic lectures.

| Attribute | Distribution |
|-----------|-------------|
| Gender | Male 94.1% (928), Female 5.9% (58) |
| Speech rate | FAST 41.9% (413), SLOW 37.8% (373), AVG 20.2% (199) |
| Region | SOUTH 36.7% (362), EAST 35.7% (352), NORTH 20.5% (202), WEST 7.0% (69) |
| Discipline | Engineering 70.1% (691), Non-Engineering 29.9% (294) |
| Total reference words | 52,178 (transcript_clean) |

---

## Evaluation Modes

4 modes covering 2 reference sources × 2 normalization states — all symmetric:

| Mode | Reference Source | Before Norm | After Norm | Purpose |
|------|-----------------|-------------|------------|---------|
| `transcript_raw` | `Transcript` | as-is | as-is | Upper bound baseline |
| `transcript_clean` | `Transcript` | `Transcript` | normalized | **Gold standard — paper primary** |
| `hf_raw` | `Normalised_Transcript` | as-is | as-is | Quantifies dataset normalization errors |
| `hf_clean` | `Normalised_Transcript` | `Normalised_Transcript` | normalized | HF normalization + our fix |

### Normalization Pipeline (`*_clean` modes)

Applied **symmetrically** to both reference and hypothesis. Standard forward normalization — everything converted to lowercase word form, no symbols.

| Step | Before | After | Why |
|------|--------|-------|-----|
| Unicode NFC | `"café"` | `"café"` | fix encoding artifacts |
| Expand contractions | `"don't"` | `"do not"` | avoid penalizing correct transcriptions |
| Fix possessives | `"Bernoulli's"` | `"bernoulli s"` | remove apostrophe, keep the s |
| Ordinals → words | `"1st"`, `"2nd"` | `"first"`, `"second"` | standard word form for WER |
| Cardinals → words | `"100"`, `"60,000"` | `"one hundred"`, `"sixty thousand"` | digit/word mismatch causes false errors |
| Decimals → words | `"3.14"` | `"three point one four"` | consistent spoken form |
| Lowercase | `"The Second"` | `"the second"` | case differences are not speech errors |
| Remove punctuation | `"hello, world."` | `"hello world"` | punctuation is editorial, not spoken |
| Normalize whitespace | `"too  many  spaces"` | `"too many spaces"` | clean tokenization |

**What we deliberately do NOT normalize:**
- Greek letter names (`sigma`, `rho`, `pi`) — they are spoken as words
- Variable names (`d0`, `s1`) — domain-specific identifiers
- The word `"point"` — kept as a word, not converted to `.`
- Filler words (`uh`, `um`) — removing selectively on one side would be unfair
- Abbreviations (`MOSFET`, `CPU`, `NPTEL`) — kept as-is

**Why the dataset's `Normalised_Transcript` is wrong:**

```
Original Transcript:    "the 1st component is..."
Normalised_Transcript:  "the one s t component is..."   ← WRONG: splits "1st" into characters
Our normalization:       "the first component is..."     ← CORRECT: ordinal → word
```

This error affects ~50+ samples and inflates `hf_raw` WER by 3.8–5.7 pp vs `transcript_raw`. The `hf_raw` mode (31.76% for Base) is *worse* than no normalization at all (27.95%).

---

## Pipeline Architecture

### Stage 1 — ASR Transcription (GPU, run once per model)

| Task | Script | Model | Est. Time (A100) |
|------|--------|-------|------------------|
| Task 1 | `task1_whisper_base/wer_whisper_base.py` | Whisper Base | ~12 min |
| Task 2 | `task2_whisper_medium/wer_whisper_medium.py` | Whisper Medium | ~35 min |
| Task 3 | `task3_whisper_large/wer_whisper_large.py` | Whisper Large | ~90 min |
| Task 4 | `task4_parakeet/wer_parakeet.py` | Parakeet-TDT-0.6B-v2 | ~20 min |
| Task 5 | `task5_qwen3_asr/wer_qwen3.py` | Qwen3-ASR-1.7B | ~30 min |

Output: `results/stage1_raw_transcripts/wer_{model}_raw.csv`

Each script is independently resumable via checkpointing — safe to restart after interruption.

### Stage 2 — Normalization + WER (CPU, re-run freely)

```bash
python normalize_and_score.py    # ~1 min, no GPU
python analysis/compare_all.py   # charts and breakdowns
```

### Automated NSCC Run (individual jobs)

```bash
qsub job_base.pbs
qsub job_medium.pbs
qsub job_large.pbs
qsub job_parakeet.pbs
qsub job_qwen3.pbs
```

Or all at once with the pipeline script (requires all conda envs set up):

```bash
qsub run_pipeline.pbs
```

---

## Quick Start

```bash
# Stage 1: Whisper (whisper conda env)
conda activate whisper
pip install openai-whisper torch librosa jiwer pandas tqdm numpy num2words datasets
python task1_whisper_base/wer_whisper_base.py
python task2_whisper_medium/wer_whisper_medium.py
python task3_whisper_large/wer_whisper_large.py

# Stage 1: Parakeet (parakeet_env conda env)
conda activate parakeet_env
bash task4_parakeet/setup.sh
python task4_parakeet/wer_parakeet.py

# Stage 1: Qwen3-ASR (qwen3_env conda env)
conda activate qwen3_env
bash task5_qwen3_asr/setup.sh
python task5_qwen3_asr/wer_qwen3.py

# Stage 2: Normalization + WER + charts (CPU only, any env with pandas/jiwer)
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
    wer_parakeet_raw.csv         ← pending
    wer_qwen3_raw.csv            ← pending
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

## CSV Column Schema

| Column | Description |
|--------|-------------|
| `model` | ASR model (`base`, `medium`, `large`, `parakeet`, `qwen3`) |
| `mode` | Evaluation mode |
| `reference_source` | Dataset column used as reference |
| `reference_raw` | Reference **before** normalization |
| `reference` | Reference **after** normalization (used for WER) |
| `hypothesis_raw` | ASR output **before** normalization |
| `hypothesis` | ASR output **after** normalization (used for WER) |
| `wer` | Per-sample WER |

## Project Structure

```
.
├── utils/
│   ├── normalize.py              # 4-mode normalization
│   ├── transcribe.py             # Audio processing + Whisper inference
│   ├── wer_compute.py            # WER computation
│   └── io_helpers.py             # Dataset loading, I/O, checkpointing
├── task1_whisper_base/
├── task2_whisper_medium/
├── task3_whisper_large/
├── task4_parakeet/               # Parakeet-TDT-0.6B-v2
├── task5_qwen3_asr/              # Qwen3-ASR-1.7B
├── archived_tasks/
│   ├── youtube_captions/         # YouTube caption experiment (archived)
│   └── audio_analysis/           # Whisper audio sample analysis (archived)
├── normalize_and_score.py        # Stage 2: Normalization + WER
├── analysis/compare_all.py       # Cross-model charts and breakdowns
├── job_{base,medium,large,parakeet,qwen3}.pbs   # Individual NSCC jobs
├── run_pipeline.pbs              # Full pipeline NSCC job
└── results/
```

## Tech Stack

- Python 3.10+
- [openai-whisper](https://github.com/openai/whisper)
- [nemo_toolkit](https://github.com/NVIDIA/NeMo) — for Parakeet-TDT
- [qwen-asr](https://github.com/QwenLM/Qwen3-ASR-Toolkit) — for Qwen3-ASR
- [jiwer](https://github.com/jitsi/jiwer)
- [num2words](https://github.com/savoirfairelinux/num2words)
- HuggingFace Datasets, pandas, matplotlib, librosa, torch
