#!/bin/bash
set -e
# Run from repo root: bash task5_qwen3_asr/setup.sh

ENV_NAME="${1:-qwen3}"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing PyTorch cu118 via conda ==="
conda install -y "pytorch==2.5.1" "torchaudio==2.5.1" "pytorch-cuda=11.8" \
    -c pytorch -c nvidia

echo "=== Installing remaining dependencies ==="
pip install -r task5_qwen3_asr/requirements.txt

echo ""
echo "Environment ready. Activate with:"
echo "  conda activate $ENV_NAME"
echo "Run benchmark from repo root:"
echo "  python task5_qwen3_asr/wer_qwen3.py"
