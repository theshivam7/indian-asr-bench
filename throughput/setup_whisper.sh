#!/bin/bash
set -euo pipefail
# Run from the repository root on an NSCC login node. Installation is allowed on
# login nodes; model inference is not.

SCRATCH_DIR=${SCRATCH:-/scratch/users/ntu/${USER}}
ENV_NAME=${1:-${WHISPER_THROUGHPUT_ENV:-${SCRATCH_DIR}/envs/whisper_throughput}}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-${SCRATCH_DIR}/conda_pkgs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${SCRATCH_DIR}/pip_cache}
export TMPDIR=${TMPDIR:-${SCRATCH_DIR}/tmp}
mkdir -p "${SCRATCH_DIR}/envs" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${TMPDIR}"

case "${ENV_NAME}" in
  /*) ENV_FLAG=-p ;;
  *)  ENV_FLAG=-n ;;
esac

if ! conda run "${ENV_FLAG}" "${ENV_NAME}" python --version >/dev/null 2>&1; then
  conda create "${ENV_FLAG}" "${ENV_NAME}" python=3.10 -y
fi

# Install file decoding independently, then let the pinned pip requirements use
# the same PyTorch 2.5.1 CUDA wheel as the dedicated native throughput environments.
# This also repairs an environment left half-created by an interrupted install.
conda install "${ENV_FLAG}" "${ENV_NAME}" -y "conda-forge::ffmpeg" -c conda-forge
conda run --no-capture-output "${ENV_FLAG}" "${ENV_NAME}" \
  python -m pip install "torch==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu124
conda run --no-capture-output "${ENV_FLAG}" "${ENV_NAME}" \
  python -m pip install -r throughput/requirements-whisper.txt
conda run --no-capture-output "${ENV_FLAG}" "${ENV_NAME}" python -m pip check
conda run "${ENV_FLAG}" "${ENV_NAME}" ffmpeg -version >/dev/null
conda run --no-capture-output "${ENV_FLAG}" "${ENV_NAME}" python -c \
  "import torch, transformers; assert torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"

echo "Ready: conda activate ${ENV_NAME}"
