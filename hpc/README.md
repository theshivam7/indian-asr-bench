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
huggingface-cli login                             # once — Svarah is a gated HF dataset
PROJECT=<nscc_project_id> bash hpc/submit_all.sh --phase all   # submit everything, correctly chained
PROJECT=<nscc_project_id> bash hpc/submit_all.sh --setup       # also create missing conda envs first
```

Without `--phase`, the submitter only submits **phase 1** (see `hpc/NSCC_RUNBOOK.md` for the
phased 1/2/3 submission flow, and the `--after <job_id>` flag for chaining phases you submit
separately). Two additional phases cover the fine-tuning replicates:

- `--phase seeds` — disjoint-FT seed replicates 43/44 (`job_finetune_disjoint_seed.pbs` ×2)
  plus ONE chained `job_score.pbs` rescore after both (avoids racing on shared Stage-2/3 files).
- `--phase sizematch` — the **size-matched speaker-overlapping control** (`job_finetune_sizematch.pbs`
  ×3, seeds 42/43/44, 567 random train clips each) plus one chained rescore. This is the control
  that separates the training-set-size effect from the speaker-disjointness effect (the disjoint
  filter leaves only 567/7200 clips, so the two are confounded in the disjoint runs).

Dependency graph (`-->` = PBS `afterok`):

```
job_new_models_tie ──┐              (GPU ~5h)   writes results/tie
                     ├─> job_figures (CPU)      writes paper/figures
job_svarah ──────────┘   ^          (GPU ~10h)  writes results/svarah
     │                   │
job_finetune_disjoint ───┘          (GPU ~10h)  writes results/tie (rescore)
     (afterok job_new_models_tie)

job_finetune_disjoint_seed (SEED=43) ─┐
job_finetune_disjoint_seed (SEED=44) ─┴─> job_score (CPU rescore, afterok both)

job_finetune_sizematch (SEED=42) ─┐
job_finetune_sizematch (SEED=43) ─┼─> job_score (CPU rescore, afterok all three)
job_finetune_sizematch (SEED=44) ─┘
```

TIE-new-models and Svarah run **in parallel** (disjoint result dirs); the disjoint
fine-tune is serialized after the TIE job (both rescore `results/tie`); a final CPU
figures job rebuilds the cross-dataset plots once everything is done. To submit a
single phase on its own (e.g. only Svarah), use `--phase 1|2|3` instead of `all`.

## Environments

| conda env | created from | used by |
|-----------|--------------|---------|
| `whisper` | `environments/whisper.yaml` | Whisper inference **+ all CPU scoring/analysis/figures** (has `whisper_normalizer`) |
| `parakeet` | `environments/parakeet.yaml` | Parakeet-TDT / Parakeet-CTC (NeMo) |
| `qwen3` | `environments/qwen3.yaml` | Qwen3-ASR |
| `whisper_medium_ft` | `bash task6_whisper_medium_ft/setup.sh` | fine-tuning (HF `transformers`, `datasets==4.8.5`) |

## Individual jobs

```bash
qsub -P <id> -v MODEL=large_v3_turbo,DATASET=tie hpc/job_whisper.pbs   # one Whisper model on one dataset
qsub -P <id> -v DATASET=svarah                    hpc/job_parakeet.pbs # (parakeet/qwen3 read DATASET too)
qsub -P <id> -v DATASET=svarah                    hpc/job_qwen3.pbs
qsub -P <id> -v DATASET=tie                        hpc/job_score.pbs   # CPU-only rescore + analysis (no GPU)
qsub -P <id> -v DATASETS=tie,svarah                hpc/job_figures.pbs # CPU-only combined figures
qsub -P <id> -v DATASET=svarah                     hpc/run_pipeline.pbs # full from-scratch 7-model run
```

Bundled multi-step jobs: `job_new_models_tie.pbs` (turbo + parakeet_ctc on TIE,
then rescore + analyse), `job_svarah.pbs` (all 7 models on Svarah → Stage 2/3 +
NEER), `job_finetune_disjoint.pbs` (speaker-disjoint fine-tune → transcribe →
rescore → FT report), `job_finetune_disjoint_seed.pbs` (one extra disjoint seed,
`-v SEED=43|44`; scoring deferred), `job_finetune_sizematch.pbs` (size-matched
control, `-v SEED=42|43|44`; scoring deferred), `job_score_disjoint_seed42.pbs`
(one-off: transcribe + score an existing seed-42 disjoint checkpoint without
retraining).

## Fine-tuning (standalone)

```bash
JOBID=$(qsub -P <id> hpc/job_finetune.pbs)                  # Stage 0: train → models/whisper_medium_ft/
qsub -P <id> -W depend=afterok:$JOBID hpc/job_medium_ft.pbs # Stage 1+2+3: transcribe test, WER, analysis
```

Overridable training knobs: `FT_EPOCHS`, `FT_BATCH`, `FT_GRAD_ACCUM`, `FT_LR`,
`FT_PATIENCE`; `FT_SPEAKER_DISJOINT=1` + `FT_OUTPUT_DIR=...` for the disjoint variant;
`FT_SIZE_MATCHED=567` + `FT_SEED=<s>` for the size-matched control.

## Notes

- **Reuse over recompute:** the 7 original TIE raw transcripts are committed, so you
  normally only need `job_new_models_tie` (2 new models) + `job_svarah` + the disjoint
  FT — not a full re-run. After a pure normalization/metric change, `job_score.pbs`
  (CPU) recomputes everything from the committed transcripts with no GPU.
- Scripts are resumable — re-submitting picks up from the last checkpoint.

## Adapting to SLURM

| PBS | SLURM |
|-----|-------|
| `#PBS -N name` | `#SBATCH --job-name=name` |
| `#PBS -l select=1:ngpus=1:ncpus=8:mem=32gb` | `#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=32G` |
| `#PBS -l walltime=3:00:00` | `#SBATCH --time=03:00:00` |
| `#PBS -P project_id` | `#SBATCH --account=project_id` |
| `#PBS -o logfile` | `#SBATCH --output=logfile` |
| `qsub -W depend=afterok:$JID` | `sbatch --dependency=afterok:$JID` |
