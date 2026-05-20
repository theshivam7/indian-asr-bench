#!/bin/bash
set -e
# Run from repo root: bash task2_whisper_medium/setup.sh

ENV_NAME="${1:-whisper_medium}"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing ffmpeg ==="
conda install -c conda-forge ffmpeg -y

echo "=== Installing Python dependencies ==="
pip install -r task2_whisper_medium/requirements.txt

echo ""
echo "Environment ready. Activate with:"
echo "  conda activate $ENV_NAME"
echo "Run benchmark from repo root:"
echo "  python task2_whisper_medium/wer_whisper_medium.py"
