#!/bin/bash
# ============================================================================
# Indian-ASR-Bench — NSCC (ASPIRE2A / PBS Pro) submitter.
#
# Tuned for the real cluster layout: conda envs live as PREFIX envs on /scratch,
# and $HOME is quota-limited, so ALL heavy I/O (HF dataset + model cache, fine-tune
# outputs) is forced onto /scratch.
#
# Runs in PHASES so you never need more than one GPU job at a time (respects queue
# limits / storage). Phases write to separate result dirs, so you can also run
# them in parallel if your allocation allows (PHASE=all).
#
#   PHASE 1      job_new_models_tie        GPU ~5h   turbo + parakeet_ctc on TIE, rescore
#   PHASE 2      job_svarah                GPU ~10h  all 7 models on Svarah (biggest download)
#
# USAGE (from the repo root on a login node):
#     PROJECT=<nscc_project_id> bash hpc/submit_all.sh --phase 1     # then 2
#     PROJECT=<nscc_project_id> bash hpc/submit_all.sh --setup       # create/verify envs only
#     PROJECT=<nscc_project_id> bash hpc/submit_all.sh --phase all   # submit both, chained by afterok
#
# Phase 2 (Svarah) writes to a separate results dir — safe to submit while Phase 1
# is still running:
#     PROJECT=<id> bash hpc/submit_all.sh --phase 2
#
# Svarah is a GATED HF dataset — authenticate ONCE (writes token under HF cache):
#     export HF_CACHE=/scratch/users/ntu/$USER/hf_cache
#     HF_HOME=$HF_CACHE huggingface-cli login          # or: export HF_TOKEN=hf_xxx
# ============================================================================
set -euo pipefail

# ---- storage: default everything to /scratch (HOME is over quota) -----------
SCRATCH="${SCRATCH:-/scratch/users/ntu/$USER}"
WORKDIR="${WORKDIR:-$(pwd)}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"
# conda's own package/repodata cache defaults to $HOME/.conda/pkgs — redirect it too,
# or `conda install`/`conda env create` fail with the same HOME disk-quota error.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SCRATCH/conda_pkgs}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /app/apps/miniforge3/25.3.1)}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"

# ---- conda envs: real NSCC prefix-path envs (override if yours differ) -------
WHISPER_ENV="${WHISPER_ENV:-$SCRATCH/envs/whisper}"
PARAKEET_ENV="${PARAKEET_ENV:-$SCRATCH/envs/parakeet_env}"
QWEN3_ENV="${QWEN3_ENV:-$SCRATCH/envs/qwen3_env}"
WHISPER_FT_ENV="${WHISPER_FT_ENV:-whisper_medium_ft}"   # named env in ~/.conda/envs

PROJECT="${PROJECT:-${PBS_PROJECT:-}}"

# ---- args -------------------------------------------------------------------
PHASE="1"
DO_SETUP=0
AFTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --setup) DO_SETUP=1 ;;
    --phase) shift; PHASE="${1:-1}" ;;
    --phase=*) PHASE="${1#*=}" ;;
    --after) shift; AFTER="${1:-}" ;;
    --after=*) AFTER="${1#*=}" ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

if [ ! -d "$WORKDIR/hpc" ]; then
  echo "ERROR: run from the repo root (WORKDIR=$WORKDIR has no hpc/)." >&2; exit 1
fi

echo "=================================================================="
echo " Indian-ASR-Bench NSCC submitter   (phase: $PHASE)"
echo "   WORKDIR=$WORKDIR"
echo "   SCRATCH=$SCRATCH"
echo "   HF_CACHE=$HF_CACHE   CONDA_BASE=$CONDA_BASE   CUDA_MODULE=$CUDA_MODULE"
echo "   whisper=$WHISPER_ENV"
echo "   parakeet=$PARAKEET_ENV"
echo "   qwen3=$QWEN3_ENV"
echo "   whisper_ft=$WHISPER_FT_ENV"
echo "=================================================================="
mkdir -p "$HF_CACHE" "$WORKDIR/logs"

# ---- optional: create/verify the conda envs (idempotent) --------------------
if [ "$DO_SETUP" = 1 ]; then
  echo ">>> --setup: creating any MISSING envs and verifying deps ..."
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  _env_ok() { conda run -p "$1" python -c "import sys" >/dev/null 2>&1 || conda run -n "$1" python -c "import sys" >/dev/null 2>&1; }

  # Whisper (+ all CPU scoring/analysis) — create as a prefix env on scratch if absent.
  _env_ok "$WHISPER_ENV" || conda env create -p "$WHISPER_ENV" -f environments/whisper.yaml
  _env_ok "$PARAKEET_ENV" || conda env create -p "$PARAKEET_ENV" -f environments/parakeet.yaml
  _env_ok "$QWEN3_ENV" || conda env create -p "$QWEN3_ENV" -f environments/qwen3.yaml
  _env_ok "$WHISPER_FT_ENV" || bash finetune/setup.sh "$WHISPER_FT_ENV"

  # CRITICAL: the scoring env must have whisper_normalizer (whisper_norm mode). Existing
  # envs predate its addition to whisper.yaml, so install it explicitly.
  conda run -p "$WHISPER_ENV" python -c "import whisper_normalizer" 2>/dev/null \
    || conda run -p "$WHISPER_ENV" pip install whisper_normalizer==0.1.0 \
    || conda run -n "$WHISPER_ENV" pip install whisper_normalizer==0.1.0
  echo ">>> envs ready. (Re-run without --setup, or add --phase N, to submit.)"
  [ "$PHASE" = "none" ] && exit 0
fi

if [ -z "$PROJECT" ]; then
  echo "ERROR: set PROJECT to your NSCC project id, e.g.  PROJECT=12345678 bash hpc/submit_all.sh --phase 1" >&2
  exit 1
fi

# ---- vars forwarded to every job -------------------------------------------
VARS="WORKDIR=${WORKDIR},HF_CACHE=${HF_CACHE},CONDA_BASE=${CONDA_BASE},CUDA_MODULE=${CUDA_MODULE}"
VARS="${VARS},WHISPER_ENV=${WHISPER_ENV},PARAKEET_ENV=${PARAKEET_ENV},QWEN3_ENV=${QWEN3_ENV},WHISPER_FT_ENV=${WHISPER_FT_ENV}"
# Keep fine-tune model weights off HOME (quota): write them onto scratch.
VARS="${VARS},FT_OUTPUT_DIR=${SCRATCH}/models/whisper_medium_ft"
[ -n "${HF_TOKEN:-}" ] && VARS="${VARS},HF_TOKEN=${HF_TOKEN}"

qsub_job() { qsub -P "$PROJECT" -v "$VARS" "$@"; }

submit_phase1() { local d="${1:-}"; qsub_job ${d:+-W depend=afterok:$d} hpc/job_new_models_tie.pbs; }
submit_phase2() { local d="${1:-}"; qsub_job ${d:+-W depend=afterok:$d} hpc/job_svarah.pbs; }
submit_figs()   { qsub_job -W depend=afterok"$1" hpc/job_figures.pbs; }

case "$PHASE" in
  1) J=$(submit_phase1 "$AFTER");    echo "  [phase 1] TIE new models : $J${AFTER:+ (afterok $AFTER)}" ;;
  2) J=$(submit_phase2 "$AFTER");    echo "  [phase 2] Svarah         : $J${AFTER:+ (afterok $AFTER)}" ;;
  all)
     J1=$(submit_phase1)
     J2=$(submit_phase2)                       # parallel with phase 1 (separate result dirs)
     F=$(submit_figs ":${J1}:${J2}")
     echo "  [phase 1] TIE new models : $J1"
     echo "  [phase 2] Svarah         : $J2 (parallel)"
     echo "  [final ] combined figs   : $F  (afterok $J1,$J2)" ;;
  *) echo "ERROR: --phase must be 1, 2, or all" >&2; exit 1 ;;
esac

echo "------------------------------------------------------------------"
echo "Track: qstat -u \"$USER\"    Logs: $WORKDIR/logs/*.log"
echo "When this phase finishes, run the next:  bash hpc/submit_all.sh --phase <next>"
echo "=================================================================="
