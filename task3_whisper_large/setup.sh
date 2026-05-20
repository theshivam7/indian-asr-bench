#!/bin/bash
set -e
# Run from repo root: bash task3_whisper_large/setup.sh

ENV_NAME="${1:-whisper_large}"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing ffmpeg ==="
conda install -c conda-forge ffmpeg -y

echo "=== Installing Python dependencies ==="
pip install -r task3_whisper_large/requirements.txt

echo ""
echo "Environment ready. Activate with:"
echo "  conda activate $ENV_NAME"
echo "Run benchmark from repo root:"
echo "  python task3_whisper_large/wer_whisper_large.py"
