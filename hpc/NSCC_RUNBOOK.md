# NSCC (ASPIRE2A) Runbook

Concrete, copy-paste steps for running the benchmark on NSCC. Tuned for the real
constraints observed on the cluster:

- **`$HOME` is over quota** → all heavy I/O (HF dataset + model cache, fine-tune
  weights) must live on **`/scratch`** (petabytes free).
- Conda envs are **prefix envs on scratch**: `/scratch/users/ntu/$USER/envs/{whisper,parakeet_env,qwen3_env}`,
  plus the named `whisper_medium_ft` in `~/.conda/envs`.
- Scheduler is **PBS Pro**; GPU is on compute nodes (`cuda/11.8.0` module).

## 0. One-time setup

```bash
# work on scratch to avoid the HOME quota (recommended)
export SCRATCH=/scratch/users/ntu/$USER
export HF_CACHE=$SCRATCH/hf_cache
mkdir -p "$HF_CACHE" "$SCRATCH/models"

# clone (or move) the repo onto scratch, then cd in
cd $SCRATCH && git clone https://github.com/theshivam7/indian-asr-bench && cd indian-asr-bench

# authenticate for the gated Svarah dataset (token is stored under HF_CACHE)
HF_HOME=$HF_CACHE huggingface-cli login       # or: export HF_TOKEN=hf_xxx
```

### Environments

You already have `whisper`, `parakeet_env`, `qwen3_env` (on scratch) and
`whisper_medium_ft` (in home). The submitter creates any that are **missing** and,
crucially, makes sure the scoring env has `whisper_normalizer` (needed by the
`whisper_norm` mode — older envs predate it):

```bash
PROJECT=<nscc_project_id> bash hpc/submit_all.sh --setup
```

Or do it by hand (prefix envs, on scratch):

```bash
source $(conda info --base)/etc/profile.d/conda.sh
# create only what's missing:
conda env create -p $SCRATCH/envs/whisper      -f environments/whisper.yaml    # if absent
conda env create -p $SCRATCH/envs/parakeet_env -f environments/parakeet.yaml   # if absent
conda env create -p $SCRATCH/envs/qwen3_env    -f environments/qwen3.yaml      # if absent
bash finetune/setup.sh whisper_medium_ft                        # if absent
# ensure the scoring env has the Whisper normalizer:
conda run -p $SCRATCH/envs/whisper pip install whisper_normalizer==0.1.0
```

## 1. Submit in phases (one GPU job at a time)

Each phase is a single GPU job. Submit a phase, wait for it to finish
(`qstat -u $USER`), then submit the next. Result dirs are separate per phase, so
nothing clobbers anything.

```bash
export PROJECT=<nscc_project_id>            # required
export SCRATCH=/scratch/users/ntu/$USER
export HF_CACHE=$SCRATCH/hf_cache

# Phase 1 — TIE new models (turbo + parakeet_ctc), ~5h. Rescores TIE.
bash hpc/submit_all.sh --phase 1

# Phase 2 — Svarah full 7-model benchmark, ~10h (largest download → scratch cache).
bash hpc/submit_all.sh --phase 2
```

Prefer to fire everything at once (queue permitting)? `bash hpc/submit_all.sh --phase all`
submits both with correct `afterok` chaining and runs the combined figures job last.

The script auto-forwards the scratch paths (`HF_CACHE`, `FT_OUTPUT_DIR`) and your
env locations to every job, and prints the job IDs.

Fine-tuning (Whisper Medium official split, and the Tiny/Small capacity study) is
submitted separately — see [Fine-tuning (standalone)](README.md#fine-tuning-standalone)
in the main HPC README, and `hpc/job_finetune_size.pbs` (`-v SIZE=tiny|small`) for
the capacity study.

## 2. After the jobs finish

The committed raw transcripts + newly produced ones give you everything; the CPU
stages already ran inside each job. To rebuild just the cross-dataset figures or
re-score after a code change (no GPU):

```bash
qsub -P $PROJECT -v DATASET=tie    hpc/job_score.pbs      # rescore + analyse TIE
qsub -P $PROJECT -v DATASET=svarah hpc/job_score.pbs      # rescore + analyse Svarah
qsub -P $PROJECT -v DATASETS=tie,svarah hpc/job_figures.pbs
```

Then commit the updated `results/**` and `paper/figures/**` and push.

## Troubleshooting

- **`Disk quota exceeded`** — you wrote to `$HOME`. This bites in three DIFFERENT
  places, each needing its own redirect (all handled automatically by
  `hpc/job_*.pbs` and `submit_all.sh`, but relevant if you run things by hand):
  `HF_HOME`/`HF_DATASETS_CACHE` (HF datasets/models), `XDG_CACHE_HOME`
  (openai-whisper's own model cache — does NOT follow `HF_HOME`), and
  `CONDA_PKGS_DIRS` (conda's own package/repodata cache — bites `conda install`).
  Make sure `SCRATCH`, `HF_CACHE`, and the repo itself are under `/scratch`.
  Fine-tune weights default to `$SCRATCH/models/...` via the submitter.
- **`torch.cuda.is_available()` is `False`** — first check WHERE you're testing it:
  the login node has no GPU at all (`nvidia-smi` fails there), so this is *expected*
  and uninformative interactively. The real test is inside a submitted job — every
  GPU job here prints `torch sees CUDA: True/False` near the top of its log and now
  **hard-exits** if it's `False`, so check `logs/<job>.log`. If it genuinely fails
  inside a job: a pip-installed `torch` (bare or `+cu118`) does not reliably see the
  GPU on this cluster — the env must use conda's `pytorch-cuda=11.8` build, and the
  exact build string must be pinned (`pytorch::pytorch=2.5.1=py3.10_cuda11.8_cudnn9.1.0_0`)
  or the solver can silently substitute conda-forge's CPU-only build instead.
- **`ffmpeg: No such file or directory` during Whisper transcription** — openai-whisper
  shells out to `ffmpeg` per clip; if missing, every failing clip silently gets an
  EMPTY hypothesis (~100% WER for that clip) instead of crashing the job. Verify with
  `conda run -p $SCRATCH/envs/whisper which ffmpeg`; if empty,
  `conda install -p $SCRATCH/envs/whisper -c conda-forge ffmpeg`.
- **Svarah download 401/403** — the HF token isn't visible to the job. Re-run
  `HF_HOME=$HF_CACHE huggingface-cli login`, or `export HF_TOKEN=...` before submitting.
- **`whisper_normalizer` ImportError during scoring** — run the `pip install` line
  in step 0 into the `whisper` env.
- **Svarah column mismatch warning** — the loader prints the real feature names on
  first load (`SVARAH` spec is `verified=False`). If they differ, fix the column
  names in `utils/registry.py` and flip `verified=True`, then re-run `job_svarah`.
