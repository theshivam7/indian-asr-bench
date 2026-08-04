#!/bin/bash
set -e
# Run from repo root: bash finetune/setup.sh

ENV_NAME="${1:-whisper_medium_ft}"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing Python dependencies ==="
# No conda-forge ffmpeg/torchcodec here on purpose: audio is decoded via soundfile
# (utils.io_helpers.decode_audio_value), which handles this dataset's WAV/FLAC clips
# natively. Letting conda touch this env after pip has already installed a CUDA torch
# build risks conda silently resolving in a conflicting CPU-only `pytorch` package that
# shadows it in site-packages (torch.cuda.is_available() would go False with no error).
pip install -r finetune/requirements.txt

echo ""
echo "Environment ready. Activate with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Pre-flight leakage check (CPU):"
echo "  python finetune/check_speaker_overlap.py"
echo ""
echo "Fine-tune (Stage 0):"
echo "  python finetune/finetune_medium.py"
echo ""
echo "Transcribe test split (Stage 1), baseline then fine-tuned:"
echo "  MODEL_NAME=medium_hf python finetune/evaluate_finetuned.py"
echo "  MODEL_NAME=medium_ft python finetune/evaluate_finetuned.py"
