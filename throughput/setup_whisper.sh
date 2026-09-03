#!/bin/bash
set -euo pipefail
# Run from the repository root on an NSCC login node. Installation is allowed on
# login nodes; model inference is not.

ENV_NAME=${1:-whisper_throughput}

if ! conda run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

# Install file decoding independently, then let the pinned pip requirements use
# the same working PyTorch 2.5.1 CUDA wheel as the native-engine environments.
# This also repairs an environment left half-created by an interrupted install.
conda install -n "${ENV_NAME}" -y "conda-forge::ffmpeg" -c conda-forge
conda run -n "${ENV_NAME}" python -m pip install -r throughput/requirements-whisper.txt
conda run -n "${ENV_NAME}" python -m pip check
conda run -n "${ENV_NAME}" ffmpeg -version >/dev/null
conda run -n "${ENV_NAME}" python -c \
  "import torch, transformers; assert torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"

echo "Ready: conda activate ${ENV_NAME}"
