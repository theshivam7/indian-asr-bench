#!/bin/bash
set -e
# Run from repo root: bash task5_qwen3_asr/setup.sh

ENV_NAME="${1:-qwen3}"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing PyTorch cu118 via conda ==="
# Build string pinned exactly, not just the version: an unpinned "pytorch==2.5.1"
# lets the solver silently substitute a CPU-only build from a channel other than
# `pytorch` (confirmed to happen on NSCC when conda-forge is also on the channel
# list) even with pytorch-cuda present as a constraint.
conda install -y "pytorch::pytorch=2.5.1=py3.10_cuda11.8_cudnn9.1.0_0" \
    "pytorch::torchaudio=2.5.1=py310_cu118" "pytorch::pytorch-cuda=11.8" \
    -c pytorch -c nvidia

echo "=== Installing remaining dependencies ==="
pip install -r task5_qwen3_asr/requirements.txt

echo ""
echo "Environment ready. Activate with:"
echo "  conda activate $ENV_NAME"
echo "Run benchmark from repo root:"
echo "  python task5_qwen3_asr/wer_qwen3.py"
