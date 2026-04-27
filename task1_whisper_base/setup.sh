#!/bin/bash
set -e

ENV_NAME="tie_wer_base"

echo "=== Creating Miniconda environment: $ENV_NAME whisper_base ==="
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
echo "Run with:      python wer_whisper_base.py"
