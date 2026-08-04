# HPC Job Scripts

PBS Pro job scripts for running Indian-ASR-Bench on HPC clusters (developed on
**NSCC ASPIRE2A**, NVIDIA A100-40GB, CUDA 11.8).

## Configuration

Every script reads environment variables with sensible defaults. The NSCC project
id must be passed to `qsub` on the command line (`-P`), because `#PBS` directive
lines are not shell-expanded.

```bash
export WORKDIR=/path/to/indian-asr-bench
export HF_CACHE=/scratch/$USER/hf_cache   # keep off the quota-limited $HOME
export CONDA_BASE=$(conda info --base)
export CUDA_MODULE=cuda/11.8.0            # `module avail cuda` to check the name
# conda env names (defaults shown):
export WHISPER_ENV=whisper PARAKEET_ENV=parakeet QWEN3_ENV=qwen3 WHISPER_FT_ENV=whisper_medium_ft
```

## One-shot submitter (recommended)

`submit_all.sh` creates the environments (optional) and submits every remaining
experiment with the right parallelism + dependency chaining, printing the job ids.

```bash
# from the repo root, on a login node:
hf auth login                             # once; Svarah is a gated HF dataset
PROJECT=<nscc_project_id> bash hpc/submit_all.sh --phase all   # submit everything, correctly chained
PROJECT=<nscc_project_id> bash hpc/submit_all.sh --setup       # also create missing conda envs first
```

Without `--phase`, the submitter only submits **phase 1** (see `hpc/NSCC_RUNBOOK.md` for the
phased submission flow, and the `--after <job_id>` flag for chaining phases you submit
separately). Phases: `1` (TIE new models), `2` (Svarah), `3` (AESRC Indian pretrained
benchmark), `ft-aesrc` (tiny/small/medium fine-tune on AESRC, three serially-chained jobs).

Dependency graph (`-->` = PBS `afterok`):

```
job_new_models_tie ──┐              (GPU ~5h)   writes results/tie
                     ├─> job_figures (CPU)      writes paper/figures
job_svarah ──────────┘              (GPU ~10h)  writes results/svarah
```

TIE-new-models and Svarah run **in parallel** (separate result dirs); a final CPU
figures job rebuilds the cross-dataset plots once both are done. To submit a
single phase on its own, use `--phase 1|2` instead of `all`.

## Environments

| conda env | created from | used by |
|-----------|--------------|---------|
| `whisper` | `environments/whisper.yaml` | Whisper inference **+ all CPU scoring/analysis/figures** (has `whisper_normalizer`) |
| `parakeet` | `environments/parakeet.yaml` | Parakeet-TDT / Parakeet-CTC (NeMo) |
| `qwen3` | `environments/qwen3.yaml` | Qwen3-ASR |
| `whisper_medium_ft` | `bash finetune/setup.sh` | fine-tuning (HF `transformers`, `datasets==4.8.5`) |

## Individual jobs

```bash
qsub -P <id> -v MODEL=large_v3_turbo,DATASET=tie hpc/job_whisper.pbs   # one Whisper model on one dataset
qsub -P <id> -v DATASET=svarah                    hpc/job_parakeet.pbs # (parakeet/qwen3 read DATASET too)
qsub -P <id> -v DATASET=svarah                    hpc/job_qwen3.pbs
qsub -P <id> -v DATASET=tie                        hpc/job_score.pbs   # CPU-only rescore + analysis (no GPU)
qsub -P <id> -v DATASETS=tie,svarah                hpc/job_figures.pbs # CPU-only combined figures
qsub -P <id> -v DATASET=svarah                     hpc/run_pipeline.pbs # full from-scratch 7-model run
```

`DATASET` accepts any registry key (`tie`, `svarah`, `aesrc`); the AESRC spec filters
to the Indian accent subset on load.

Bundled multi-step jobs: `job_new_models_tie.pbs` (turbo + parakeet_ctc on TIE,
then rescore + analyse), `job_svarah.pbs` (all 7 models on Svarah → Stage 2/3 +
NEER), `job_aesrc.pbs` (9 pretrained models on the AESRC Indian test split → Stage 2/3),
`job_finetune_size.pbs` (capacity-study fine-tune: `-v SIZE=tiny|small` for TIE,
`-v SIZE=tiny|small|medium,DATASET=aesrc` for AESRC).

```bash
qsub -P <id> -v DATASET=tie                          hpc/job_speaker_overlap.pbs # CPU-only train/test speaker-leakage audit
qsub -P <id> -v ENGINE=parakeet,MODEL=parakeet        hpc/job_efficiency.pbs      # RTF/latency/peak-GPU, one model per submission
qsub -P <id> -v SIZE=tiny,DATASET=aesrc               hpc/job_finetune_seeds.pbs  # multi-seed capacity study (default: seeds 42-47)
```

`job_finetune_seeds.pbs` has its full usage (including the SEEDS-quoting gotcha) documented in its
own header comment; see also `analysis/compare_seeds.py` to aggregate the resulting per-seed tables.

## Fine-tuning (standalone)

```bash
JOBID=$(qsub -P <id> hpc/job_finetune.pbs)                  # Stage 0: train → models/whisper_medium_ft/
qsub -P <id> -W depend=afterok:$JOBID hpc/job_medium_ft.pbs # Stage 1+2+3: transcribe test, WER, analysis
```

Overridable training knobs: `FT_EPOCHS`, `FT_BATCH`, `FT_GRAD_ACCUM`, `FT_LR`, `FT_PATIENCE`.

## Notes

- **Reuse over recompute:** the raw TIE/Svarah transcripts are committed, so you
  normally only need `job_new_models_tie` (2 new models) + `job_svarah`, not a
  full re-run. After a pure normalization/metric change, `job_score.pbs`
  (CPU) recomputes everything from the committed transcripts with no GPU.
- Scripts are resumable: re-submitting picks up from the last checkpoint.
- **Run provenance:** each Stage-1 run writes
  `results/<dataset>/stage1_raw_transcripts/wer_<model>_manifest.json` recording the
  pinned dataset revision, package versions, host, timing and the code commit. The
  commit comes from `GIT_COMMIT`, which `submit_all.sh` resolves on the login node and
  forwards, because `git` is not on PATH on every compute node. Manifests written
  before that was added carry an empty `git_commit`; for those runs, provenance is the
  commit that added the transcript to git history. If you submit a job by hand rather
  than through `submit_all.sh`, pass `-v GIT_COMMIT=$(git rev-parse HEAD),...` so the
  field is not blank.

## Adapting to SLURM

| PBS | SLURM |
|-----|-------|
| `#PBS -N name` | `#SBATCH --job-name=name` |
| `#PBS -l select=1:ngpus=1:ncpus=8:mem=32gb` | `#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=32G` |
| `#PBS -l walltime=3:00:00` | `#SBATCH --time=03:00:00` |
| `#PBS -P project_id` | `#SBATCH --account=project_id` |
| `#PBS -o logfile` | `#SBATCH --output=logfile` |
| `qsub -W depend=afterok:$JID` | `sbatch --dependency=afterok:$JID` |
