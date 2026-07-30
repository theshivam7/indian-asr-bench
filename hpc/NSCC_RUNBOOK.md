# NSCC (ASPIRE2A) Runbook

Concrete, copy-paste steps for running the benchmark on NSCC. Tuned for the real
constraints observed on the cluster:

- **`$HOME` is quota-limited** → all heavy I/O (HF dataset + model cache, fine-tune
  weights, the repo itself) must live on **`/scratch`** (petabytes free).
- **Conda envs are named envs in `~/.conda/envs`**: `whisper`, `parakeet`, `qwen3`,
  `whisper_medium_ft`. They deliberately do NOT live on scratch. Scratch is subject to
  an automatic purge of files that have not been accessed recently, and it deletes
  inside an env directory as readily as anywhere else. A purged env loses
  `lib/python3.10/encodings/` and every later command dies with a bare
  `init_fs_encoding: failed to get the Python codec of the filesystem encoding`,
  which names neither conda nor the purge. Envs are small; keep them in home and keep
  only data on scratch.
- Scheduler is **PBS Pro**; GPU is on compute nodes (`cuda/11.8.0` module).

## Never run compute on the login node: always `qsub -I` first

The login node (`asp2a-login-*`) is a **shared entry point for every NSCC user**
across NUS/NTU/A*STAR/SUTD/NEA; it's for `ssh`, `git`, editing, and submitting
jobs, nothing else. NSCC's own guidance is explicit: *"Avoid running heavy load
tasks in login nodes"* / *"Avoid transfer massive amount of data on login node."*
A fair-share monitor kills offending processes automatically and **more than 3
violations can auto-block the account.**

This applies to *anything* that loads a model or transcribes audio, including a
"quick" one-off probe script, not just full pipeline runs. **Before running any
python command that does real work (model inference, audio decoding, anything
beyond reading a CSV), grab an interactive compute-node session first:**

```bash
# CPU interactive session (audio probes, quick checks, no GPU needed)
qsub -I -P $PROJECT -l select=1:ncpus=8:mem=32gb -l walltime=00:30:00 -q normal

# GPU interactive session (model inference / timing probes)
qsub -I -P $PROJECT -l select=1:ngpus=1 -l walltime=00:30:00 -q normal
```

Wait for the shell prompt to switch to a **compute-node hostname** (anything
other than `asp2a-login-*`) before running anything; that's the signal you're
off the login node. Then `cd` back into the repo, `conda activate` the env, and
run the command. `exit` when done to free the node back to the queue.

For anything expected to run longer than a few minutes, or that's part of the
regular pipeline, submit a real batch job (`hpc/job_*.pbs` via `qsub`, see
below) instead of holding an interactive session open.

## 0. One-time setup

```bash
# work on scratch to avoid the HOME quota (recommended)
export SCRATCH=/scratch/users/ntu/$USER
export HF_CACHE=$SCRATCH/hf_cache
mkdir -p "$HF_CACHE" "$SCRATCH/models"

# clone (or move) the repo onto scratch, then cd in
cd $SCRATCH && git clone https://github.com/theshivam7/indian-asr-bench && cd indian-asr-bench

# authenticate for the gated Svarah dataset (token is stored under HF_CACHE)
HF_HOME=$HF_CACHE hf auth login       # or: export HF_TOKEN=hf_xxx
```

### Environments

Four named envs are needed: `whisper`, `parakeet`, `qwen3`, `whisper_medium_ft`. The
submitter creates any that are missing and makes sure the scoring env has
`whisper_normalizer`, which the `whisper_norm` mode needs and older envs predate:

```bash
PROJECT=<nscc_project_id> bash hpc/submit_all.sh --setup
```

By hand, point conda's package and pip caches at scratch first. Both default into
`$HOME` and will exhaust the home quota partway through an install:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/scratch/users/ntu/$USER/conda_pkgs
export PIP_CACHE_DIR=/scratch/users/ntu/$USER/pip_cache

conda env create -n whisper -f environments/whisper.yaml   # if absent
bash finetune/setup.sh whisper_medium_ft                   # if absent
```

`environments/parakeet.yaml` and `environments/qwen3.yaml` no longer solve. Channel
drift has made their MKL / llvm-openmp / mkl_random build hashes mutually
unsatisfiable, and the solver fails rather than picking substitutes. Build those two
from the pip requirements instead, which is what the CI-equivalent path does:

```bash
conda create -n parakeet python=3.10 -y && conda run -n parakeet pip install -r parakeet/requirements.txt
conda create -n qwen3    python=3.10 -y && conda run -n qwen3    pip install -r qwen3/requirements.txt
```

Verify before submitting anything, because a broken env costs a full queue wait to
discover:

```bash
for e in whisper parakeet qwen3 whisper_medium_ft; do printf "%-18s " "$e"; conda run -n $e python -c "import torch,datasets;print('torch',torch.__version__,'datasets',datasets.__version__)"; done
conda run -n parakeet python -c "import nemo.collections.asr; print('nemo ok')"
```

All four must report `datasets 4.8.5` or later. Version 3.x cannot read a cache that
4.x wrote and fails with `Feature type 'List' not found`, which reads like a dataset
problem rather than a version mismatch.

## 1. Submit in phases (one GPU job at a time)

Each phase is a single GPU job (ft-aesrc submits three, internally chained). Submit
a phase, wait for it to finish (`qstat -u $USER`), then submit the next. Result dirs
are separate per phase, so no phase overwrites another's output.

```bash
export PROJECT=<nscc_project_id>            # required
export SCRATCH=/scratch/users/ntu/$USER
export HF_CACHE=$SCRATCH/hf_cache

# Phase 1: TIE new models (turbo + parakeet_ctc), ~5h. Rescores TIE.
bash hpc/submit_all.sh --phase 1

# Phase 2: Svarah full 7-model benchmark, ~10h (largest download → scratch cache).
bash hpc/submit_all.sh --phase 2

# Phase 3: AESRC2020 Indian subset: 9 pretrained models on the 1,731-clip test split,
# ~8h incl. the first-run parquet download (~1.7GB, split-scoped) into $HF_CACHE.
bash hpc/submit_all.sh --phase 3

# Phase ft-aesrc: fine-tune tiny/small/medium on AESRC's Indian train split (12,820
# clips, 17.5h; first run downloads ~14.4GB of train parquet to $HF_CACHE). Three GPU
# jobs, auto-chained so only one runs at a time. Weights go to $SCRATCH/models/.
bash hpc/submit_all.sh --phase ft-aesrc
```

After ft-aesrc finishes, run ONE scoring pass (CPU job) to score everything and build
the pretrained-vs-fine-tuned reports under `results/aesrc/analysis/`:

```bash
qsub -P $PROJECT -v DATASET=aesrc,WHISPER_ENV=whisper hpc/job_score.pbs
```

Prefer to fire everything at once (queue permitting)? `bash hpc/submit_all.sh --phase all`
submits both with correct `afterok` chaining and runs the combined figures job last.

The script auto-forwards the scratch paths (`HF_CACHE`, `FT_OUTPUT_DIR`) and your
env locations to every job, and prints the job IDs.

TIE fine-tuning (Whisper Medium official split, and the Tiny/Small capacity study) is
submitted separately; see [Fine-tuning (standalone)](README.md#fine-tuning-standalone)
in the main HPC README, and `hpc/job_finetune_size.pbs` (`-v SIZE=tiny|small`) for
the capacity study. AESRC fine-tuning uses the same job with `DATASET=aesrc`
(`-v SIZE=tiny|small|medium,DATASET=aesrc`), or `--phase ft-aesrc` above for all three.

Three more standalone jobs, not covered by `submit_all.sh` phases:

```bash
# Speaker-leakage disclosure (CPU-only, minutes)
qsub -P $PROJECT -v DATASET=tie hpc/job_speaker_overlap.pbs

# Inference-efficiency benchmark: one model per submission (engines can't share an env)
qsub -P $PROJECT -v ENGINE=parakeet,MODEL=parakeet hpc/job_efficiency.pbs
# then: python analysis/compare_efficiency.py --dataset tie

# Multi-seed fine-tuning capacity study (gdev walltime cap: chunk seeds if >2h/seed at this size)
qsub -P $PROJECT -v SIZE=tiny,DATASET=aesrc hpc/job_finetune_seeds.pbs
# then: python analysis/compare_seeds.py --dataset aesrc
```

## 2. After the jobs finish

The committed raw transcripts + newly produced ones give you everything; the CPU
stages already ran inside each job. To rebuild just the cross-dataset figures or
re-score after a code change (no GPU):

```bash
qsub -P $PROJECT -v DATASET=tie,WHISPER_ENV=whisper    hpc/job_score.pbs   # rescore + analyse TIE
qsub -P $PROJECT -v DATASET=svarah,WHISPER_ENV=whisper hpc/job_score.pbs   # rescore + analyse Svarah
qsub -P $PROJECT -v DATASET=aesrc,WHISPER_ENV=whisper  hpc/job_score.pbs   # rescore + analyse AESRC (Indian)
qsub -P $PROJECT -v DATASETS=tie,svarah,aesrc hpc/job_figures.pbs
```

Then commit the updated `results/**` and push (`paper/` is local-only, not tracked).

## Troubleshooting

- **NSCC fair-share violation email** ("we have terminated all your processes on
  the login node"): means real compute (model inference, audio decoding) ran
  directly at the `asp2a-login-*` shell prompt instead of inside a job. See
  [Never run compute on the login node](#never-run-compute-on-the-login-node-always-qsub--i-first)
  above: request `qsub -I` first, every time, even for a 30-second test. This
  counts toward an automatic account block after 3 violations, so treat it as a
  hard stop, not a warning.
- **`Disk quota exceeded`**: you wrote to `$HOME`. This occurs in three DIFFERENT
  places, each needing its own redirect (all handled automatically by
  `hpc/job_*.pbs` and `submit_all.sh`, but relevant if you run things by hand):
  `HF_HOME`/`HF_DATASETS_CACHE` (HF datasets/models), `XDG_CACHE_HOME`
  (openai-whisper's own model cache, which does NOT follow `HF_HOME`), and
  `CONDA_PKGS_DIRS` (conda's own package/repodata cache, which affects `conda install`).
  Make sure `SCRATCH`, `HF_CACHE`, and the repo itself are under `/scratch`.
  Fine-tune weights default to `$SCRATCH/models/...` via the submitter.
- **`torch.cuda.is_available()` is `False`**: first check WHERE you're testing it:
  the login node has no GPU at all (`nvidia-smi` fails there), so this is *expected*
  and uninformative interactively. The real test is inside a submitted job: every
  GPU job here prints `torch sees CUDA: True/False` near the top of its log and now
  **hard-exits** if it's `False`, so check `logs/<job>.log`. If it genuinely fails
  inside a job: a pip-installed `torch` (bare or `+cu118`) does not reliably see the
  GPU on this cluster: the env must use conda's `pytorch-cuda=11.8` build, and the
  exact build string must be pinned (`pytorch::pytorch=2.5.1=py3.10_cuda11.8_cudnn9.1.0_0`)
  or the solver can silently substitute conda-forge's CPU-only build instead.
- **`ffmpeg: No such file or directory` during Whisper transcription**: openai-whisper
  shells out to `ffmpeg` per clip; if missing, every failing clip silently gets an
  EMPTY hypothesis (~100% WER for that clip) instead of crashing the job. Verify with
  `conda run -n whisper which ffmpeg`; if empty,
  `conda install -n whisper -c conda-forge ffmpeg`.
- **Svarah download 401/403** happens when the HF token isn't visible to the job. Re-run
  `HF_HOME=$HF_CACHE hf auth login`, or `export HF_TOKEN=...` before submitting.
- **`whisper_normalizer` ImportError during scoring**: run the `pip install` line
  in step 0 into the `whisper` env.
- **Svarah column mismatch warning**: the loader prints the real feature names on
  first load (`SVARAH` spec is `verified=False`). If they differ, fix the column
  names in `utils/registry.py` and flip `verified=True`, then re-run `job_svarah`.
