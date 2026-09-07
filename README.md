<h1 align="center">Indian-ASR-Bench</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
  <img src="https://img.shields.io/badge/Datasets-TIE__shorts%20+%20Svarah%20+%20AESRC-orange" />
  <img src="https://img.shields.io/badge/Models-9%20ASR%20systems-blue" />
  <a href="https://huggingface.co/theshivam7">
    <img src="https://img.shields.io/badge/Hugging%20Face-checkpoints-yellow?logo=huggingface" />
  </a>
</p>

<p align="center">
  <b>A reproducible Word Error Rate benchmark for ASR on Indian English speech:<br>
  three corpora, nine systems, five normalization modes, clustered significance tests,
  a validated reference-artifact classifier, and one throughput protocol for every system.</b>
</p>

<p align="center">
  <a href="#key-features">Features</a> &nbsp;|&nbsp;
  <a href="#datasets">Datasets</a> &nbsp;|&nbsp;
  <a href="#models">Models</a> &nbsp;|&nbsp;
  <a href="#pipeline">Pipeline</a> &nbsp;|&nbsp;
  <a href="#results">Results</a> &nbsp;|&nbsp;
  <a href="#installation">Installation</a> &nbsp;|&nbsp;
  <a href="#usage">Usage</a> &nbsp;|&nbsp;
  <a href="SUMMARY.md">Full analysis</a>
</p>

<p align="center">
  <img src="results/benchmark_overview.png" alt="Corpus WER with 95% cluster-bootstrap confidence intervals for nine ASR systems on TIE_shorts, Svarah and AESRC2020 (Indian)" width="100%">
</p>

<p align="center">
  <sub>The same nine systems, reordered by every corpus. Whisper Medium wins on scraped lecture
  audio and Large-v3 on both curated corpora; on TIE the top five overlap inside their confidence
  intervals. Regenerate with <code>python analysis/make_overview_figure.py</code>.</sub>
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
- Nine pretrained models run head to head: Whisper across six sizes (Tiny, Base, Small, Medium, Large-v3, large-v3-turbo), both Parakeet variants (TDT and CTC), and Qwen3-ASR.
- Up to five normalization modes apply symmetrically to reference and hypothesis, so ranking artifacts from text cleanup are visible instead of hidden.
- Significance testing uses a speaker- or recording-clustered paired bootstrap, Holm-corrected across every pairwise model comparison, and is run under both normalizers rather than only the primary one.
- A cross-model consensus classifier flags reference/audio mismatches from agreement patterns across all nine models. A human review of the 49 hardest TIE clips checks it.
- Inference cost sits next to accuracy: a 512-clip quality-gated batch sweep measures offline throughput, GPU utilization, memory and power for all nine systems on all three corpora on one A100-40GB, one CUDA runtime, one provenance digest.
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
| Whisper Tiny / Base / Small / Medium / Large-v3 / large-v3-turbo | 39M-1.5B | Encoder-Decoder |
| Parakeet-TDT-0.6B-v2 | 600M | CTC + TDT |
| Parakeet-CTC-1.1B | 1.1B | CTC |
| Qwen3-ASR-1.7B | 1.7B | LLM-based |

All nine run as-is on all three datasets; that is the benchmark. Full model table with parameter
counts and links: [SUMMARY.md, Models](SUMMARY.md#models).

Fine-tuned Whisper checkpoints from an exploratory study are also published on the
[Hugging Face Hub](https://huggingface.co/theshivam7). See
[Exploratory: fine-tuning](#exploratory-fine-tuning) below.

---

## Pipeline

One registry-driven pipeline runs identically on every dataset. Only the loading step is dataset-specific.

```mermaid
flowchart LR
    R(["<b>Registry</b><br/>9 models, 3 datasets, 5 modes"])

    subgraph GPU ["GPU, once per model"]
        direction TB
        S1["<b>Stage 1</b><br/>Transcribe"]
    end

    subgraph CPU ["CPU, re-runs from disk"]
        direction TB
        S2["<b>Stage 2</b><br/>Normalize + score"]
        S3["<b>Stage 3</b><br/>Stats, artifacts, charts"]
        S2 --> S3
    end

    R --> S1
    S1 -- "raw transcripts<br/><i>committed, immutable</i>" --> S2
    S3 --> O(["Leaderboards, CIs<br/>Taxonomy, Figures"])

    style R fill:#0072B2,stroke:#004c77,color:#fff
    style O fill:#009E73,stroke:#00674c,color:#fff
    style S1 fill:#D55E00,stroke:#8f3f00,color:#fff
    style S2 fill:#F0E442,stroke:#b3aa00,color:#111
    style S3 fill:#F0E442,stroke:#b3aa00,color:#111
```

Stage 1 is committed and immutable, the reproducibility anchor. Any normalization or metric
change re-runs Stages 2 and 3 straight from those committed transcripts, no re-inference needed.
That is why the hero chart above and every table in [SUMMARY.md](SUMMARY.md) rebuild on a laptop
in minutes. Adding a dataset or model is a one-line registry entry.

Per-clip Stage-2 output is not tracked in git (about 200 MB that rebuilds in minutes). Only the
per-dataset summary CSV is committed, and CI checks that a fresh rebuild matches it byte for byte.
Stage table and decode-config detail: [SUMMARY.md, Pipeline in detail](SUMMARY.md#pipeline-in-detail).

---

## Results

Corpus WER under `transcript_clean` (gold, symmetric normalization), best model per dataset:

| Dataset | Best model | Corpus WER | Runner-up |
|---|---|:---:|---|
| TIE_shorts | Whisper Medium | **14.76%** | Parakeet-TDT-0.6B-v2 (15.60%) |
| Svarah | Whisper Large-v3 | **7.11%** | Whisper Medium (7.89%) |
| AESRC2020 (Indian) | Whisper Large-v3 | **5.20%** | Qwen3-ASR-1.7B (5.23%) |

What stood out:

- Bigger is not always better: on TIE, WER falls from Tiny to Medium, then rises again at Large-v3, and a smaller model wins outright.
- The median clip beats corpus WER by 3 to 12 pp; a small tail of severe misses, largely reference artifacts, pulls the average up.
- The normalizer changes conclusions, not just numbers: 6 of 36 Holm-corrected pairwise verdicts on TIE flip depending on which normalizer is used, against 0 of 36 on either curated corpus. What drives it is how tightly the leaderboard is packed, not how far WER moves.
- Human review of TIE's 49 hardest clips (WER above 40% for at least 3 of 4 strong models): correcting the reference drops mean WER on that subset from 64.8% to 17.0%, and 46 of 49 clips trace to a bad reference, not a model failure. See [Classifier validation (human review)](SUMMARY.md#classifier-validation-human-review).
- Batching reorders the cost ranking, so the measurement protocol decides the conclusion. Qwen3-ASR is near the bottom at batch 1 and reaches parity with the best Whisper by batch 128, with the highest GPU utilization of anything tested (83.8% mean SM on TIE).
- Whisper gains least from batching because its short-form path pads every clip to 30 seconds. On Svarah, Whisper Tiny runs at 1.8% mean GPU utilization, and the padded window inflates its RTFx by up to 6.6x on the short-clip corpora. The aggregator reports both numbers.
- The pre-registered 0.10 pp quality gate never rejects a Whisper configuration and only ever binds on Parakeet and Qwen3. It understates Parakeet-CTC on TIE by 7.5x, and the aggregator now reports that cost for every model. See [Inference efficiency](SUMMARY.md#inference-efficiency).

Full leaderboards, confidence intervals, significance tests, demographic breakdowns, normalization
sensitivity, error-artifact analysis, and the throughput panel: **[SUMMARY.md](SUMMARY.md)**.

### Human review status

| Corpus | Review | Clips |
|---|---|:---:|
| TIE_shorts | complete (single annotator, non-blind) | 49 |
| Svarah | pending, sheet built, not yet annotated | 60 |
| AESRC2020 (Indian) | pending, sheet built, not yet annotated | 28 |

Nothing in the published numbers depends on the two pending reviews. Sheets and audio extraction
live in `analysis/<dataset>_validation/`.

### Exploratory: fine-tuning

An exploratory fine-tuning study is included for completeness. It is not part of the paper's core
claims. Whisper Tiny, Small and Medium were fine-tuned on AESRC's Indian training split, whose
test speakers are disjoint from training, and each size was retrained from six seeds. All 18 runs
improve on their own pretrained baseline (6-seed mean deltas of -6.85, -1.65 and -1.22 pp), but no
seed-level significance test exists yet, so this is informal evidence. The Tiny delta is measured
against an HF-pipeline baseline that scores 17.45% where openai-whisper scores 13.66% with the
same weights, so counted from the leaderboard number the Tiny gain is about 3 pp, not 7. Details,
tables and caveats: [Fine-tuning and split design (exploratory)](SUMMARY.md#fine-tuning-and-split-design-exploratory).
A matching study on TIE is archived, not reported, because TIE's official split places every test
speaker in training ([why](archived_tasks/tie_finetuning/README.md)).

---

## Installation

```bash
git clone https://github.com/theshivam7/indian-asr-bench && cd indian-asr-bench
pip install -r requirements.txt
```

Requires Python 3.10 or newer. That is everything needed to reproduce every table and chart,
because Stage 1 transcripts are committed.

### What you need

| Task | Hardware | Environment |
|---|---|---|
| Reproduce every table and figure | any laptop, CPU only, a few minutes | `requirements.txt` |
| Re-transcribe one dataset with one engine | one CUDA GPU (Large-v3 at batch 1 peaks near 4 GB in the sweep; all published runs used a 40 GB A100) | the engine's own env, see below |
| Throughput sweep as published | one A100-40GB (batch 128 on Large-v3 peaks at 38 GB) | `throughput/setup_*.sh` |
| Fine-tuning | one 40 GB A100 is what the published runs used; smaller cards were not tested | `finetune/setup.sh` |

Re-transcribing or fine-tuning needs GPU environments, and **each engine needs its own**:
openai-whisper, NeMo and the Qwen3 stack cannot coexist in one environment, and the fine-tuning
stack pins newer versions than any of them. That is why `whisper_asr/`, `parakeet/`, `qwen3/` and
`finetune/` each ship a separate `requirements.txt` whose pins deliberately disagree with each
other. Install them with the matching `setup.sh`, and do not align the versions across files.

Full conda specs are in [`environments/`](environments/). The exact package sets the published
results were produced with are in [`environments/resolved/`](environments/resolved/), captured
from the cluster. Use those if the two engine `.yaml` files fail to solve, which they now do.

### Tested environment

The committed results were produced on Python 3.10.20, linux-64, one NVIDIA A100-SXM4-40GB per
job, torch 2.5.1. The engine environments used CUDA 11.8; the throughput environments used
CUDA 12.4. Per-run package versions are recorded in
`results/<dataset>/stage1_raw_transcripts/*_manifest.json`. The CPU analysis path is tested in CI
on Ubuntu with Python 3.10 and 3.12. The full list, and where the pinned files differ from what
actually ran, is in [SUMMARY.md, Tested environment](SUMMARY.md#tested-environment).

### Known limits of reproduction

- Stage 2 and 3 reproduce every number bit for bit from the committed transcripts. Stage 1 does not: openai-whisper's temperature fallback is stochastic, so a fresh transcription can differ on a few hard clips.
- `environments/parakeet.yaml` and `environments/qwen3.yaml` no longer solve because of conda channel drift. Rebuild those two from [`environments/resolved/`](environments/resolved/).
- The engine environments need CUDA 11.8 and the throughput environments need CUDA 12.4. One machine can host both only if its driver supports both runtimes.
- The requirements files were captured after the runs. They pin numpy 1.26.4 and jiwer 3.1.0 for Whisper, while the run manifests record numpy 2.2.6 and jiwer 4.0.0. This does not change any published number, because scoring runs in the top-level environment and CI checks the rebuild.
- The throughput sweep was measured on A100-40GB nodes. Results on any other GPU are a different experiment and the aggregator refuses to merge them.

---

## Usage

Every command takes `--dataset {tie,svarah,aesrc}`. Only Stage 1 needs a GPU; everything
else recomputes on CPU from the committed transcripts.

### Reproduce the results (no GPU)

Run Stage 2 first on a fresh clone. The per-clip CSVs it writes are what every `analysis/`
script reads.

```bash
python normalize_and_score.py       --dataset tie    # Stage 2: normalize + score (run first)
python analysis/compare_all.py      --dataset tie    # Stage 3: tables + charts
python analysis/statistics.py       --dataset tie    # cluster-bootstrap CIs, Holm-corrected
python analysis/error_analysis.py   --dataset tie    # artifact taxonomy + instrument audit
python analysis/compare_throughput.py --dataset tie  # throughput panel from the committed sweep
```

`statistics.py` defaults to the pre-registered primary mode. Add `--mode whisper_norm` to
reproduce the cross-normalizer comparison where 6 of 36 TIE verdicts flip. A table-by-table map
from every number in SUMMARY.md to the command that produces it is in
[SUMMARY.md, Reproducing each table](SUMMARY.md#reproducing-each-table).

<details>
<summary><b>Expected output</b> from the first command</summary>

Corpus WER per model across every applicable mode, then the summary path. The numbers
below are the committed ones, so a fresh checkout reproduces them exactly:

```
model            transcript_raw  transcript_clean  hf_raw  hf_clean  whisper_norm
...
medium                   15.11             14.76   18.01     15.76         14.48
parakeet                 15.97             15.60   18.54     16.40         15.17
tiny                     19.79             19.43   22.20     20.07         19.01

Saved summary to results/tie/stage2_processed
Done.
```

Anything other than `14.76` for `medium` under `transcript_clean` means something has
drifted; CI checks exactly this on every push.
</details>

### Transcribe with a model (GPU)

```bash
bash whisper_asr/setup.sh                          # one env for all Whisper sizes
python whisper_asr/run_whisper.py  --model large_v3_turbo --dataset tie
python parakeet/wer_parakeet.py    --model parakeet_ctc   --dataset tie
python qwen3/wer_qwen3.py                                 --dataset svarah
```

Each engine needs its own environment; see [Installation](#installation). Run
`python scripts/smoke_test.py --dataset <ds>` first: it exercises the exact loading and decode
paths on a few clips and fails in seconds if the environment is wrong.

### Measure throughput (GPU)

The quality-gated batch sweep is pre-registered in
[INFERENCE_EFFICIENCY_PROTOCOL.md](INFERENCE_EFFICIENCY_PROTOCOL.md). Each engine runs in its own
CUDA 12.4 environment and the aggregator refuses to merge runs whose workload, hardware, software
or provenance differ.

```bash
bash throughput/setup_whisper.sh && bash throughput/setup_native.sh parakeet && bash throughput/setup_native.sh qwen3
PROJECT=<project_id> bash hpc/submit_throughput.sh                     # nine one-GPU jobs (PBS)
python analysis/compare_throughput.py --dataset tie --require-complete  # aggregate, gate on all nine
```

The committed sweep is complete: 9 models x 3 corpora x 8 batch sizes. Two reporting caveats are
documented in [Inference efficiency](SUMMARY.md#inference-efficiency): the quality gate only ever
binds on NeMo and Qwen3 entries, and RTFx does not account for Whisper's fixed 30-second padding
window, so the aggregator reports `gate_cost_x` and `padded_rtfx_audio_s_per_s` next to the
headline numbers.

### Fine-tune and replicate across seeds (GPU, exploratory)

```bash
bash finetune/setup.sh
python finetune/finetune_tiny_small.py --dataset aesrc \
    --base-model openai/whisper-tiny --output-dir models/whisper_tiny_aesrc_ft
DATASET=aesrc MODEL_NAME=tiny_hf       MODEL_SOURCE=openai/whisper-tiny            python finetune/evaluate_finetuned.py
DATASET=aesrc MODEL_NAME=tiny_aesrc_ft MODEL_SOURCE=models/whisper_tiny_aesrc_ft   python finetune/evaluate_finetuned.py
bash finetune/run_seeds.sh --size tiny --dataset aesrc   # seeds 42-47, resumable
python analysis/compare_seeds.py --dataset aesrc --mode all
```

### On a cluster (PBS Pro or SLURM)

```bash
hf auth login                                     # once, Svarah is gated
PROJECT=<project_id> bash hpc/submit_all.sh       # --setup also creates the conda envs
```

The scripts in [`hpc/`](hpc/) are PBS Pro and were run on a single A100-40GB node. Every path
and environment name is an overridable variable, and [`hpc/README.md`](hpc/README.md) has a
PBS-to-SLURM table for other schedulers.

---

## Data availability

- **In this repo:** every Stage-1 transcript with its run manifest, the per-dataset Stage-2 summary, every analysis table and figure, the raw throughput sweep JSONs, and the human-review sheets.
- **On the Hugging Face Hub:** the exploratory fine-tuned checkpoints, listed in [SUMMARY.md, Models](SUMMARY.md#models).
- **You download yourself:** the three datasets, from their Hugging Face pages above. Svarah is gated and needs `hf auth login`. This repo never redistributes audio.
- **AESRC2020:** the mirror states no licence and the corpus is Datatang's. Access for this research was confirmed through our advisor; redistribution or commercial use would need separately clarified terms.

---

## Repository structure

```
indian-asr-bench/
├── normalize_and_score.py   Stage 2 entry point: normalize + score from committed transcripts
├── utils/               registry, normalization, WER computation, dataset loading, throughput protocol
├── whisper_asr/         Whisper transcription driver
├── parakeet/            NeMo Parakeet transcription driver
├── qwen3/               Qwen3-ASR transcription driver
├── throughput/          quality-gated batched-throughput drivers and environments
├── finetune/            exploratory fine-tuning, multi-seed runner, evaluation scripts
├── analysis/            Stage 3: comparisons, statistics, error analysis, throughput, seeds
│   ├── tie_validation/      human review of TIE's 49 hardest clips (complete)
│   ├── svarah_validation/   review sheet for Svarah (pending)
│   └── aesrc_validation/    review sheet for AESRC (pending)
├── results/<dataset>/   stage1_raw_transcripts/, stage2_processed/ (summary only), analysis/, throughput/
├── hpc/                 PBS job scripts, with a SLURM translation table
├── environments/        conda env specs per engine, plus the resolved package sets
├── scripts/             smoke test
├── tests/               pytest regression suite
└── archived_tasks/      exploratory work SUMMARY.md still cites (TIE fine-tuning, YouTube captions)
```

---

## Authors

**Shivam Sharma**, Nanyang Technological University, Singapore.
[`@theshivam7`](https://github.com/theshivam7) on GitHub, [`theshivam7`](https://huggingface.co/theshivam7) on Hugging Face.

**Changsong Liu**, Nanyang Technological University, Singapore. Supervisor.

Built during a research internship at NTU Singapore. Compute was provided by the National
Supercomputing Centre (NSCC) Singapore.

---

<p align="center">
  <a href="SUMMARY.md"><b>Full results and analysis</b></a>
  &nbsp;|&nbsp;
  <a href="CONTRIBUTING.md"><b>Contributing</b></a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/theshivam7"><b>Models</b></a>
  &nbsp;|&nbsp;
  <a href="https://github.com/theshivam7/indian-asr-bench/releases"><b>Releases</b></a>
</p>

<p align="center">
  <sub>
    Code released under the <a href="LICENSE">MIT License</a>.
    Each dataset keeps its own terms:
    <a href="https://huggingface.co/datasets/raianand/TIE_shorts">TIE_shorts</a>,
    <a href="https://huggingface.co/datasets/ai4bharat/Svarah">Svarah</a>,
    <a href="https://huggingface.co/datasets/pengyizhou/accented_english">accented_english</a>.
  </sub>
</p>
