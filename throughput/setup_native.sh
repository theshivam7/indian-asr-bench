#!/bin/bash
set -euo pipefail
# Build a dedicated CUDA-12.4 environment for either native throughput engine.
# Keeping these environments separate from the Stage-1 environments makes the
# measured software stack reproducible without changing previously published runs.

ENGINE=${1:?usage: bash throughput/setup_native.sh parakeet|qwen3 [conda_env_or_prefix]}
SCRATCH_DIR=${SCRATCH:-/scratch/users/ntu/${USER}}

case "${ENGINE}" in
  parakeet)
    DEFAULT_ENV=${SCRATCH_DIR}/envs/parakeet_throughput
    REQUIREMENTS=parakeet/requirements.txt
    ENGINE_CHECK="import nemo.collections.asr"
    ;;
  qwen3)
    DEFAULT_ENV=${SCRATCH_DIR}/envs/qwen3_throughput
    REQUIREMENTS=qwen3/requirements.txt
    ENGINE_CHECK="import qwen_asr"
    ;;
  *)
    echo "[FATAL] ENGINE must be parakeet or qwen3 (got '${ENGINE}')" >&2
    exit 2
    ;;
esac

ENV_NAME=${2:-${DEFAULT_ENV}}
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

# Install the identical PyTorch/CUDA build used by the Whisper throughput env
# before resolving the engine requirements. torch==2.5.1 accepts 2.5.1+cu124,
# so the requirements file preserves this explicitly selected wheel.
conda run --no-capture-output "${ENV_FLAG}" "${ENV_NAME}" \
  python -m pip install "torch==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu124
conda run --no-capture-output "${ENV_FLAG}" "${ENV_NAME}" \
  python -m pip install -r "${REQUIREMENTS}"
conda run --no-capture-output "${ENV_FLAG}" "${ENV_NAME}" python -m pip check
conda run --no-capture-output "${ENV_FLAG}" "${ENV_NAME}" python -c \
  "import torch; ${ENGINE_CHECK}; assert torch.version.cuda == '12.4', torch.version.cuda; print('engine', '${ENGINE}', 'torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"

echo "Ready: conda activate ${ENV_NAME}"
