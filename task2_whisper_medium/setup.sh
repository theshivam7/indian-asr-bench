#!/bin/bash
set -e

ENV_NAME="tie_wer_medium"

echo "=== Creating Miniconda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Installing ffmpeg (required by whisper) ==="
conda install -c conda-forge ffmpeg -y

echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
echo "Run with:      python wer_whisper_medium.py"
