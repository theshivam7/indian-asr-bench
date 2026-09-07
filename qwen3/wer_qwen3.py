"""
Stage 1: ASR transcription. Qwen3-ASR-1.7B (LLM-based).

Usage:
    python qwen3/wer_qwen3.py --dataset tie
    python qwen3/wer_qwen3.py --dataset svarah

Writes results/<dataset>/stage1_raw_transcripts/wer_qwen3_raw.csv.
Uses the shared utils.inference_loop (dataset-aware, resumable, SIGTERM-safe).
"""

import argparse
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.efficiency import cudnn_disabled, timed
from utils.transcribe import temp_wavs

MODEL_KEY = "qwen3"


def transcribe_qwen3(model, sample: dict, audio_col: str) -> str:
    try:
        with temp_wavs([sample], audio_col) as (path,):
            r = model.transcribe(audio=path, language="English")[0]
            return (r.text if hasattr(r, "text") else str(r)).strip()
    except Exception as e:
        raise RuntimeError("Qwen3-ASR transcription failed") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tie")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")

    from qwen_asr import Qwen3ASRModel
    from utils.registry import MODEL_BY_KEY, get_dataset
    from utils.inference_loop import run_transcription

    model_id = MODEL_BY_KEY[MODEL_KEY].model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device} ...")
    load_timing = []
    with cudnn_disabled(), timed(load_timing):
        if device == "cuda":
            model = Qwen3ASRModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto", max_new_tokens=512)
        else:
            model = Qwen3ASRModel.from_pretrained(model_id, device_map="cpu", max_new_tokens=512)
    print(f"Model loaded in {load_timing[0]:.1f}s.\n")

    audio_col = get_dataset(args.dataset).audio_col
    decode_kwargs = {"language": "English", "max_new_tokens": 512}

    run_transcription(MODEL_KEY, args.dataset,
                      transcribe_one=lambda s: transcribe_qwen3(model, s, audio_col),
                      manifest_extra={"decode_kwargs": decode_kwargs})
    print("Done.")


if __name__ == "__main__":
    main()
