#!/bin/bash
set -e
# Run from the repository root on an NSCC login node. Installation is allowed on
# login nodes; model inference is not.

ENV_NAME=${1:-whisper_throughput}

conda create -n "${ENV_NAME}" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
conda install -y \
  "pytorch::pytorch=2.5.1=py3.10_cuda11.8_cudnn9.1.0_0" \
  "pytorch::torchaudio=2.5.1=py310_cu118" \
  "pytorch::pytorch-cuda=11.8" \
  "conda-forge::ffmpeg" \
  -c pytorch -c nvidia
pip install -r throughput/requirements-whisper.txt

echo "Ready: conda activate ${ENV_NAME}"
