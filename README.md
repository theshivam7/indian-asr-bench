<h1 align="center">Indian-ASR-Bench</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
  <img src="https://img.shields.io/badge/Datasets-TIE__shorts%20+%20Svarah%20+%20AESRC-orange" />
  <img src="https://img.shields.io/badge/Models-9%20pretrained%20+%203%20fine--tuned-blue" />
  <a href="https://huggingface.co/theshivam7">
    <img src="https://img.shields.io/badge/Hugging%20Face-models%20%26%20datasets-yellow?logo=huggingface" />
  </a>
</p>

<p align="center">
  <b>A reproducible Word Error Rate benchmark for ASR on Indian English speech:<br>
  three datasets, nine models each, up to five normalization modes, and a fine-tuning capacity study across model sizes.</b>
</p>

<p align="center">
  <a href="#key-features">Features</a> &nbsp;·&nbsp;
  <a href="#datasets">Datasets</a> &nbsp;·&nbsp;
  <a href="#models">Models</a> &nbsp;·&nbsp;
  <a href="#pipeline">Pipeline</a> &nbsp;·&nbsp;
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#usage">Usage</a> &nbsp;·&nbsp;
  <a href="SUMMARY.md">Full analysis</a>
</p>

---

## Overview

Most ASR benchmarks focus on American and British English. Indian English is spoken by over a
billion people and gets far less evaluation attention, and academic lecture speech makes it
harder still: fast delivery, dense technical vocabulary, real-world recording conditions.

This project runs the same nine ASR systems across three Indian-English corpora under an
identical, registry-driven pipeline, then asks how much of the reported WER is the model versus
the evaluation choices around it: reference field, normalizer, and dataset artifacts.

One number to start with: the reference and normalizer you score against can move a model as
much as swapping the model itself. Full detail in [Normalization](SUMMARY.md#normalization).

## Key features

- Three datasets share one pipeline: TIE_shorts (scraped lecture speech), Svarah (curated read speech), and the AESRC2020 Indian subset (short prompted speech), all scored identically.
- Nine pretrained models run head to head: Whisper across five sizes, both Parakeet variants (TDT and CTC), and Qwen3-ASR.
- Up to five normalization modes apply symmetrically to reference and hypothesis, so ranking artifacts from text cleanup are visible instead of hidden.
- Significance testing uses a speaker- or recording-clustered paired bootstrap, Holm-corrected across every pairwise model comparison, and is run under both normalizers rather than only the primary one.
- A cross-model consensus classifier flags reference/audio mismatches from agreement patterns across all nine models, without hand review.
- Split design is treated as an evaluation-validity property: TIE's official splits are shown to be speaker-entangled, and the fine-tuning capacity study (Tiny, Small, Medium) runs on AESRC, whose test set is natively speaker-disjoint from training.
- Every table and chart regenerates on CPU from the committed Stage-1 transcripts; no GPU or re-transcription needed.

---

## Datasets

| Dataset | Type | Test clips | Link |
|---|---|:---:|---|
| TIE_shorts | Scraped NPTEL lecture audio | 986 | [HF Hub](https://huggingface.co/datasets/raianand/TIE_shorts) |
| Svarah | Curated read-speech prompts | 6,656 | [HF Hub](https://huggingface.co/datasets/ai4bharat/Svarah) |
| AESRC2020 (Indian subset) | Short prompted read speech | 1,731 | [HF Hub](https://huggingface.co/datasets/pengyizhou/accented_english) |

Full split sizes, durations, and demographic breakdowns (gender, region, speech rate, age, native
language): [SUMMARY.md, Datasets](SUMMARY.md#datasets).

---

## Models

| Model | Params | Architecture |
|---|:---:|:---:|
| Whisper Tiny / Base / Small / Medium / Large-v3 / large-v3-turbo | 39M–1.5B | Encoder-Decoder |
| Parakeet-TDT-0.6B-v2 | 600M | CTC + TDT |
| Parakeet-CTC-1.1B | 1.1B | CTC |
| Qwen3-ASR-1.7B | 1.7B | LLM-based |

All nine run as-is on all three datasets; that is the headline benchmark. Three more (Whisper
Tiny/Small/Medium, fine-tuned on AESRC) are published on the [Hugging Face
Hub](https://huggingface.co/theshivam7) and analyzed in [Fine-tuning and split
design](SUMMARY.md#fine-tuning-and-split-design).
Full model table with parameter counts and links: [SUMMARY.md, Models](SUMMARY.md#models).

---

## Pipeline

One registry-driven pipeline runs identically on every dataset. Only the loading step is dataset-specific.

```mermaid
flowchart LR
    R(["Registry<br/>models · datasets · modes"]) --> S1

    subgraph GPU["GPU · once per model"]
        S1["Stage 1<br/>Transcribe"]
    end

    subgraph CPU["CPU · fully reproducible"]
        S2["Stage 2<br/>Normalize + Score"] --> S3["Stage 3<br/>Analyze"]
    end

    S1 -- "raw transcripts<br/>(committed)" --> S2
    S3 --> O(["Tables, stats,<br/>charts"])
```

Stage 1 is committed and immutable, the reproducibility anchor. Any normalization or metric
change re-runs Stages 2 and 3 straight from those committed transcripts, no re-inference needed.
Adding a dataset or model is a one-line registry entry. Stage table and decode-config detail:
[SUMMARY.md, Pipeline in detail](SUMMARY.md#pipeline-in-detail).

---

## Results

Corpus WER under `transcript_clean` (gold, symmetric normalization), best model per dataset:

| Dataset | Best model | Corpus WER | Runner-up |
|---|---|:---:|---|
| TIE_shorts | Whisper Medium | **14.76%** | Parakeet-TDT-0.6B-v2 (15.60%) |
| Svarah | Whisper Large-v3 | **7.11%** | Whisper Medium (7.89%) |
| AESRC2020 (Indian) | Whisper Large-v3 | **5.20%** | Qwen3-ASR-1.7B (5.23%) |

A few things stood out across all three datasets:

- Bigger is not always better: on TIE, WER falls from Tiny to Medium, then rises again at Large-v3, and a smaller model wins outright.
- The median clip beats corpus WER by 3 to 12 pp; a small tail of severe misses, largely reference artifacts, pulls the average up.
- The normalizer changes conclusions, not just numbers: 5 of 36 Holm-corrected pairwise verdicts on TIE flip depending on which normalizer is used, against 0 of 36 on either curated corpus. What drives it is how tightly the leaderboard is packed, not how far WER moves.
- Fine-tuning showed a significant gain on AESRC (speaker-disjoint test set) for Small and Medium, real domain or accent adaptation rather than memorization. The same question is unanswerable on TIE, whose official splits put every test speaker in training.

Full leaderboards, confidence intervals, significance tests, demographic breakdowns, normalization
sensitivity, error-artifact analysis, and the complete fine-tuning study:
**[→ SUMMARY.md](SUMMARY.md)**

---

## Installation

```bash
git clone https://github.com/theshivam7/indian-asr-bench && cd indian-asr-bench
pip install -r requirements.txt
```

Requires Python 3.10+. That is everything needed to reproduce every table and chart, because
Stage 1 transcripts are committed.

Re-transcribing or fine-tuning needs GPU environments, and **each engine needs its own**:
openai-whisper, NeMo and the Qwen3 stack cannot coexist in one environment, and the fine-tuning
stack pins newer versions than any of them. That is why `whisper_asr/`, `parakeet/`, `qwen3/` and
`finetune/` each ship a separate `requirements.txt` whose pins deliberately disagree with each
other. Install them with the matching `setup.sh`, and do not align the versions across files.
Full conda specs are in [`environments/`](environments/).

---

## Usage

**Analysis only (no GPU).** Recompute every table and chart from the committed transcripts:

```bash
python normalize_and_score.py --dataset tie      # Stage 2
python analysis/compare_all.py --dataset tie     # Stage 3 tables + charts
python analysis/statistics.py --dataset tie      # cluster-bootstrap CIs + Holm-corrected tests
python analysis/error_analysis.py --dataset tie  # artifact taxonomy + instrument audit
python analysis/compare_finetune.py --dataset aesrc   # fine-tuning report
```

Repeat with `--dataset svarah` or `--dataset aesrc` for the other two corpora.

Significance is reported under both normalizers. `analysis/statistics.py` defaults to the
pre-registered primary mode; pass `--mode whisper_norm` to reproduce the cross-normalizer
comparison that shows 5 of 36 TIE verdicts flipping:

```bash
python analysis/statistics.py --dataset tie --mode whisper_norm
```

**Transcription (GPU).** Registry-driven drivers with `--model` and `--dataset`:

```bash
bash whisper_asr/setup.sh                                              # one env for all Whisper models
python whisper_asr/run_whisper.py --model large_v3_turbo --dataset tie
python parakeet/wer_parakeet.py --model parakeet_ctc --dataset tie
python qwen3/wer_qwen3.py --dataset svarah
```

**On a cluster (NSCC / PBS Pro).** One command submits every remaining experiment with dependency chaining:

```bash
hf auth login                                           # once, Svarah is gated
PROJECT=<nscc_project_id> bash hpc/submit_all.sh        # add --setup to also create the conda envs
```

Or drive pieces individually: `qsub -P <id> -v DATASET=svarah hpc/run_pipeline.pbs` (full run),
`qsub -P <id> -v DATASET=tie hpc/job_score.pbs` (CPU-only re-scoring). See [`hpc/README.md`](hpc/README.md).

**Fine-tuning (GPU).** AESRC's official split, per model size. AESRC is used because its test
speakers are disjoint from training, which is what makes a measured gain attributable:

```bash
bash finetune/setup.sh
python finetune/finetune_tiny_small.py --dataset aesrc --base-model openai/whisper-tiny --output-dir models/whisper_tiny_aesrc_ft
DATASET=aesrc MODEL_NAME=tiny_hf MODEL_SOURCE=openai/whisper-tiny python finetune/evaluate_finetuned.py
DATASET=aesrc MODEL_NAME=tiny_aesrc_ft MODEL_SOURCE=models/whisper_tiny_aesrc_ft python finetune/evaluate_finetuned.py
```

Or on the cluster: `qsub -v SIZE=tiny,DATASET=aesrc hpc/job_finetune_size.pbs`.

**Multi-seed fine-tuning (GPU).** One seed cannot separate a real effect from run-to-run noise,
so each size can be trained across several seeds and reported as mean and standard deviation:

```bash
bash finetune/run_seeds.sh --size tiny --dataset aesrc     # 6 seeds by default
python analysis/compare_seeds.py --dataset aesrc           # mean, SD, min, max per size
```

Or on the cluster: `qsub -v SIZE=tiny hpc/job_finetune_seeds.pbs`. The runner skips seeds that
are already scored, so resubmitting after a walltime kill resumes rather than restarts. The
across-seed spread is reported separately from the within-run bootstrap CI, because they measure
different things and pooling them would misstate both.

**Efficiency benchmarking (GPU).** Real-time factor, latency percentiles, throughput, and peak
GPU memory, measured on a seeded clip subset so every model sees identical audio:

```bash
python whisper_asr/run_whisper.py --model medium --dataset tie --efficiency
python parakeet/wer_parakeet.py --model parakeet --dataset tie --efficiency
python qwen3/wer_qwen3.py --dataset tie --efficiency
python analysis/compare_efficiency.py --dataset tie        # merge into one table
```

Or on the cluster: `qsub -v ENGINE=whisper,MODEL=medium hpc/job_efficiency.pbs`. Keep `--clips`
and `--seed` identical across models; the aggregator refuses to present runs measured on
different subsets or different GPUs as a single comparable table.

Conda specs live in [`environments/`](environments/), PBS jobs in [`hpc/`](hpc/). All runs used a
single NVIDIA A100-40GB (NSCC ASPIRE2A).

---

## Repository structure

```
indian-asr-bench/
├── utils/               registry, normalization, WER computation, dataset loading, efficiency probes
├── whisper_asr/         Whisper transcription driver (--efficiency for speed/memory)
├── parakeet/            NeMo Parakeet transcription driver (--efficiency)
├── qwen3/               Qwen3-ASR transcription driver (--efficiency)
├── finetune/            fine-tuning, multi-seed runner, evaluation scripts
├── analysis/            Stage 3: comparisons, statistics, error analysis, efficiency, seeds, human-eval protocol
├── results/<dataset>/   stage1_raw_transcripts/, stage2_processed/, analysis/
├── hpc/                 PBS job scripts + runbook for NSCC
├── environments/        conda env specs per engine
├── tests/               pytest suite
└── archived_tasks/      exploratory work kept for reference (YouTube captions, etc.)
```

---

## Author

Shivam Sharma

Developed during a research internship at Nanyang Technological University (NTU), Singapore,
under the supervision of Liu Changsong.

---

<p align="center">
  MIT licensed, see <a href="LICENSE">LICENSE</a>. Datasets (<a href="https://huggingface.co/datasets/raianand/TIE_shorts">TIE_shorts</a>, <a href="https://huggingface.co/datasets/ai4bharat/Svarah">Svarah</a>, <a href="https://huggingface.co/datasets/pengyizhou/accented_english">accented_english</a>) keep their own licenses.<br>
  Full results and analysis: <a href="SUMMARY.md">SUMMARY.md</a>. Contributions welcome, see <a href="CONTRIBUTING.md">CONTRIBUTING.md</a>.
</p>
