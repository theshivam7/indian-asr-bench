#!/bin/bash
set -e
# Run from repo root: bash whisper_asr/setup.sh
# One env for all openai-whisper models (base / medium / large / large_v3_turbo).

ENV_NAME="${1:-whisper}"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing ffmpeg ==="
conda install -c conda-forge ffmpeg -y

echo "=== Installing Python dependencies ==="
pip install -r whisper_asr/requirements.txt

echo ""
echo "Environment ready. Activate with:"
echo "  conda activate $ENV_NAME"
echo "Run e.g.:  python whisper_asr/run_whisper.py --model large_v3_turbo --dataset svarah"
