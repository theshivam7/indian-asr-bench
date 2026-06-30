#!/bin/bash
set -e
# Run from repo root: bash task6_whisper_medium_ft/setup.sh

ENV_NAME="${1:-whisper_medium_ft}"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing ffmpeg ==="
conda install -c conda-forge ffmpeg -y

echo "=== Installing Python dependencies ==="
pip install -r task6_whisper_medium_ft/requirements.txt

echo ""
echo "Environment ready. Activate with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Pre-flight leakage check (CPU):"
echo "  python task6_whisper_medium_ft/check_speaker_overlap.py"
echo ""
echo "Fine-tune (Stage 0):"
echo "  python task6_whisper_medium_ft/finetune.py"
echo ""
echo "Transcribe test split (Stage 1) — baseline then fine-tuned:"
echo "  MODEL_NAME=medium_hf python task6_whisper_medium_ft/wer_whisper_medium_ft.py"
echo "  MODEL_NAME=medium_ft python task6_whisper_medium_ft/wer_whisper_medium_ft.py"
