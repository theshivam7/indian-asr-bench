"""
Stage 1: ASR transcription — Qwen3-ASR-1.7B (LLM-based).

Usage:
    python task5_qwen3_asr/wer_qwen3.py --dataset tie
    python task5_qwen3_asr/wer_qwen3.py --dataset svarah

Writes results/<dataset>/stage1_raw_transcripts/wer_qwen3_raw.csv.
Uses the shared utils.inference_loop (dataset-aware, resumable, SIGTERM-safe).
"""

import argparse
import os
import sys
import tempfile
import warnings

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import audio_to_wav_16k

MODEL_KEY = "qwen3"


def transcribe_qwen3(model, sample: dict, audio_col: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        audio_to_wav_16k(sample[audio_col], tmp_path)
        results = model.transcribe(audio=tmp_path, language="English")
        r = results[0]
        return (r.text if hasattr(r, "text") else str(r)).strip()
    except Exception as e:
        print(f"  [WARN] transcription failed: {e}", flush=True)
        return ""
    finally:
        os.unlink(tmp_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tie")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    torch.backends.cudnn.enabled = False

    from qwen_asr import Qwen3ASRModel
    from utils.registry import MODEL_BY_KEY, get_dataset
    from utils.inference_loop import run_transcription

    model_id = MODEL_BY_KEY[MODEL_KEY].model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device} ...")
    if device == "cuda":
        model = Qwen3ASRModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto", max_new_tokens=512)
    else:
        model = Qwen3ASRModel.from_pretrained(model_id, device_map="cpu", max_new_tokens=512)
    print("Model loaded.\n")

    audio_col = get_dataset(args.dataset).audio_col
    run_transcription(MODEL_KEY, args.dataset,
                      transcribe_one=lambda s: transcribe_qwen3(model, s, audio_col))
    print("Done.")


if __name__ == "__main__":
    main()
