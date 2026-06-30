# HPC Job Scripts

PBS Pro job scripts for running Indian-ASR-Bench on HPC clusters.

## Configuration

All scripts use environment variables with sensible defaults. Set them before submitting:

```bash
export WORKDIR=/path/to/indian-asr-bench
export PBS_PROJECT=your_project_id
export CONDA_BASE=/path/to/conda
export WHISPER_ENV=whisper          # conda env name for Whisper tasks
export PARAKEET_ENV=parakeet        # conda env name for Parakeet
export QWEN3_ENV=qwen3              # conda env name for Qwen3
export HF_CACHE=/path/to/hf/cache
export CUDA_MODULE=cuda/11.8.0      # adjust to your cluster's module name
```

## Submitting Individual Jobs

```bash
cd /path/to/indian-asr-bench

qsub hpc/job_base.pbs       # Whisper Base     (~3h on A100)
qsub hpc/job_medium.pbs     # Whisper Medium   (~4h on A100)
qsub hpc/job_large.pbs      # Whisper Large    (~6h on A100)
qsub hpc/job_parakeet.pbs   # Parakeet-TDT     (~3h on A100)
qsub hpc/job_qwen3.pbs      # Qwen3-ASR        (~5h on A100)
```

## Full Pipeline (Single Job)

Runs all 5 models sequentially in one PBS job (~10h):

```bash
qsub hpc/run_pipeline.pbs
```

## Fine-tuning Whisper Medium

Two jobs, chained so evaluation auto-starts when training succeeds:

```bash
export WHISPER_FT_ENV=whisper_medium_ft      # conda env from task6_whisper_medium_ft/setup.sh

JOBID=$(qsub hpc/job_finetune.pbs)           # Stage 0: train (~8h on A100) → models/whisper_medium_ft/
qsub -W depend=afterok:$JOBID hpc/job_medium_ft.pbs   # Stage 1+2+3: transcribe test, WER, analysis (~2-3h)
```

- `job_finetune.pbs` — full fine-tune of Whisper Medium on the `train` split. Resumable (auto-detects the
  latest checkpoint). Overridable via `FT_EPOCHS`, `FT_BATCH`, `FT_GRAD_ACCUM`, `FT_LR`, `FT_PATIENCE`.
- `job_medium_ft.pbs` — transcribes the `test` split with both the same-engine pretrained baseline
  (`MODEL_NAME=medium_hf`) and the fine-tuned model (`MODEL_NAME=medium_ft`), then runs
  `normalize_and_score.py`, `analysis/compare_all.py`, and `analysis/compare_finetune.py`.

## Adapting to SLURM

Replace PBS directives (`#PBS`) with SLURM equivalents (`#SBATCH`):

| PBS | SLURM |
|-----|-------|
| `#PBS -N name` | `#SBATCH --job-name=name` |
| `#PBS -l select=1:ngpus=1:ncpus=8:mem=32gb` | `#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=32G` |
| `#PBS -l walltime=3:00:00` | `#SBATCH --time=03:00:00` |
| `#PBS -P project_id` | `#SBATCH --account=project_id` |
| `#PBS -o logfile` | `#SBATCH --output=logfile` |

## Notes

- Scripts are resumable: if interrupted, re-submitting picks up from last checkpoint.
- CUDA module name (`CUDA_MODULE`) varies by cluster. Run `module avail cuda` to see options.
- These scripts were developed on NSCC ASPIRE2A (NVIDIA A100) with CUDA 11.8.
