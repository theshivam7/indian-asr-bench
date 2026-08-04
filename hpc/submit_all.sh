#!/bin/bash
# ============================================================================
# Indian-ASR-Bench, NSCC (ASPIRE2A / PBS Pro) submitter.
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
#   PHASE 3      job_aesrc                 GPU ~8h   9 pretrained models on AESRC Indian test
#   PHASE ft-aesrc  job_finetune_size x3   GPU ~5h each  tiny/small/medium fine-tune on AESRC,
#                                          submitted serially (afterany-chained)
#
# USAGE (from the repo root on a login node):
#     PROJECT=<nscc_project_id> bash hpc/submit_all.sh --phase 1     # then 2, 3, ft-aesrc
#     PROJECT=<nscc_project_id> bash hpc/submit_all.sh --setup       # create/verify envs only
#     PROJECT=<nscc_project_id> bash hpc/submit_all.sh --phase all   # submit 1+2, chained by afterok
#
# Phases write to separate results dirs, safe to submit while another phase runs:
#     PROJECT=<id> bash hpc/submit_all.sh --phase 2
#     PROJECT=<id> bash hpc/submit_all.sh --phase 3
#
# Svarah is a GATED HF dataset, authenticate ONCE (writes token under HF cache):
#     export HF_CACHE=/scratch/users/ntu/$USER/hf_cache
#     HF_HOME=$HF_CACHE huggingface-cli login          # or: export HF_TOKEN=hf_xxx
# ============================================================================
set -euo pipefail

# ---- storage: default everything to /scratch (HOME is over quota) -----------
SCRATCH="${SCRATCH:-/scratch/users/ntu/$USER}"
WORKDIR="${WORKDIR:-$(pwd)}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"
# conda's own package/repodata cache defaults to $HOME/.conda/pkgs, redirect it too,
# or `conda install`/`conda env create` fail with the same HOME disk-quota error.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SCRATCH/conda_pkgs}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /app/apps/miniforge3/25.3.1)}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"

# ---- conda envs: named envs in ~/.conda/envs (override if yours differ) ------
# Deliberately NOT on scratch. Scratch is auto-purged by inactivity, and the purge
# deletes inside an env as readily as anywhere else; losing lib/python3.10/encodings/
# leaves every command failing with a bare init_fs_encoding error that names neither
# conda nor the purge. Envs are small, so they live in home and only data goes to
# scratch. Prefix paths still work if passed explicitly.
WHISPER_ENV="${WHISPER_ENV:-whisper}"
PARAKEET_ENV="${PARAKEET_ENV:-parakeet}"
QWEN3_ENV="${QWEN3_ENV:-qwen3}"
WHISPER_FT_ENV="${WHISPER_FT_ENV:-whisper_medium_ft}"

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

  # An env may be given either as a name or as an absolute prefix path; conda needs
  # -n for the first and -p for the second, so dispatch on the leading slash.
  _flag() { case "$1" in /*) echo "-p" ;; *) echo "-n" ;; esac; }

  # Whisper (+ all CPU scoring/analysis).
  _env_ok "$WHISPER_ENV" || conda env create "$(_flag "$WHISPER_ENV")" "$WHISPER_ENV" -f environments/whisper.yaml

  # parakeet and qwen3 are built from pip, not from their yaml. Channel drift has made
  # the MKL / llvm-openmp / mkl_random build hashes in those two specs mutually
  # unsatisfiable, so `conda env create -f` fails outright. The yaml files are kept as
  # a record of the exact versions the published results were produced with.
  _env_ok "$PARAKEET_ENV" || { conda create "$(_flag "$PARAKEET_ENV")" "$PARAKEET_ENV" python=3.10 -y \
      && conda run "$(_flag "$PARAKEET_ENV")" "$PARAKEET_ENV" pip install -r parakeet/requirements.txt; }
  _env_ok "$QWEN3_ENV" || { conda create "$(_flag "$QWEN3_ENV")" "$QWEN3_ENV" python=3.10 -y \
      && conda run "$(_flag "$QWEN3_ENV")" "$QWEN3_ENV" pip install -r qwen3/requirements.txt; }

  _env_ok "$WHISPER_FT_ENV" || bash finetune/setup.sh "$WHISPER_FT_ENV"

  # CRITICAL: the scoring env must have whisper_normalizer (whisper_norm mode). Existing
  # envs predate its addition to whisper.yaml, so install it explicitly.
  # Both envs need it, not just the scoring env: run_seeds.sh scores each seed inside
  # the fine-tuning env, so a missing package there kills a seed after its GPU time
  # has already been spent on training and transcription.
  for _e in "$WHISPER_ENV" "$WHISPER_FT_ENV"; do
    conda run "$(_flag "$_e")" "$_e" python -c "import whisper_normalizer" 2>/dev/null \
      || conda run "$(_flag "$_e")" "$_e" pip install whisper_normalizer==0.1.0
  done

  # datasets 3.x cannot read a cache written by 4.x and dies with "Feature type 'List'
  # not found", which reads as a dataset problem rather than a version mismatch. Catch
  # it here rather than after a job has queued, run, and failed.
  for _e in "$WHISPER_ENV" "$PARAKEET_ENV" "$QWEN3_ENV" "$WHISPER_FT_ENV"; do
    conda run "$(_flag "$_e")" "$_e" python -c "import datasets,sys; v=tuple(int(x) for x in datasets.__version__.split('.')[:1]); sys.exit(0 if v[0]>=4 else 1)" \
      || echo "  [WARN] $_e has datasets <4: it cannot read the shared 4.x cache. Fix: conda run $(_flag "$_e") $_e pip install -U 'datasets==4.8.5'"
  done
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
# Resolve the commit HERE, on the login node, and forward it: utils.io_helpers stamps it
# into every Stage-1 manifest, and `git` is not on PATH on every compute node. Without
# this the manifests' git_commit field is silently blank, which is how every run before
# this change lost its code provenance.
GIT_COMMIT="$(git -C "${WORKDIR:-.}" rev-parse HEAD 2>/dev/null || true)"
if [ -n "${GIT_COMMIT}" ]; then
  VARS="${VARS},GIT_COMMIT=${GIT_COMMIT}"
else
  echo "WARNING: could not resolve a git commit; Stage-1 manifests will have no provenance." >&2
fi

qsub_job() { qsub -P "$PROJECT" -v "$VARS" "$@"; }

submit_phase1() { local d="${1:-}"; qsub_job ${d:+-W depend=afterok:$d} hpc/job_new_models_tie.pbs; }
submit_phase2() { local d="${1:-}"; qsub_job ${d:+-W depend=afterok:$d} hpc/job_svarah.pbs; }
submit_phase3() { local d="${1:-}"; qsub_job ${d:+-W depend=afterok:$d} hpc/job_aesrc.pbs; }
submit_figs()   { qsub_job -W depend=afterok"$1" hpc/job_figures.pbs; }

# Three fine-tune jobs (tiny -> small -> medium) on AESRC, chained afterany so only one
# GPU job runs at a time. Each needs its own SIZE/FT_SIZE_OUTPUT, so VARS is per-job.
submit_ft_aesrc() {
  local dep="${1:-}" size jid prev="" depflag
  for size in tiny small medium; do
    if [ -n "$prev" ]; then depflag="-W depend=afterany:$prev";
    elif [ -n "$dep" ]; then depflag="-W depend=afterok:$dep";
    else depflag=""; fi
    jid=$(qsub -P "$PROJECT" \
          -v "${VARS},SIZE=${size},DATASET=aesrc,FT_SIZE_OUTPUT=${SCRATCH}/models/whisper_${size}_aesrc_ft" \
          $depflag hpc/job_finetune_size.pbs)
    echo "  [ft-aesrc] ${size} fine-tune : $jid${prev:+ (afterany $prev)}"
    prev="$jid"
  done
}

case "$PHASE" in
  1) J=$(submit_phase1 "$AFTER");    echo "  [phase 1] TIE new models : $J${AFTER:+ (afterok $AFTER)}" ;;
  2) J=$(submit_phase2 "$AFTER");    echo "  [phase 2] Svarah         : $J${AFTER:+ (afterok $AFTER)}" ;;
  3) J=$(submit_phase3 "$AFTER");    echo "  [phase 3] AESRC Indian   : $J${AFTER:+ (afterok $AFTER)}" ;;
  ft-aesrc) submit_ft_aesrc "$AFTER" ;;
  all)
     J1=$(submit_phase1)
     J2=$(submit_phase2)                       # parallel with phase 1 (separate result dirs)
     F=$(submit_figs ":${J1}:${J2}")
     echo "  [phase 1] TIE new models : $J1"
     echo "  [phase 2] Svarah         : $J2 (parallel)"
     echo "  [final ] combined figs   : $F  (afterok $J1,$J2)" ;;
  *) echo "ERROR: --phase must be 1, 2, 3, ft-aesrc, or all" >&2; exit 1 ;;
esac

echo "------------------------------------------------------------------"
echo "Track: qstat -u \"$USER\"    Logs: $WORKDIR/logs/*.log"
echo "When this phase finishes, run the next:  bash hpc/submit_all.sh --phase <next>"
echo "=================================================================="
