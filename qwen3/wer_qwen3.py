"""
Stage 1: ASR transcription. Qwen3-ASR-1.7B (LLM-based).

Usage:
    python qwen3/wer_qwen3.py --dataset tie
    python qwen3/wer_qwen3.py --dataset svarah
    python qwen3/wer_qwen3.py --dataset tie --efficiency   # timing, not transcripts

Writes results/<dataset>/stage1_raw_transcripts/wer_qwen3_raw.csv, or with
--efficiency, results/<dataset>/efficiency/efficiency_qwen3.json (see
utils/efficiency.py for the measurement protocol).
Uses the shared utils.inference_loop (dataset-aware, resumable, SIGTERM-safe).
"""

import argparse
import os
import sys
import tempfile
import warnings

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.efficiency import (DEFAULT_CLIPS, DEFAULT_SEED, DEFAULT_WARMUP, count_parameters,
                              run_efficiency_benchmark, timed)
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
        raise RuntimeError("Qwen3-ASR transcription failed") from e
    finally:
        os.unlink(tmp_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tie")
    ap.add_argument("--efficiency", action="store_true",
                    help="measure speed/memory on a seeded clip subset instead of transcribing the split")
    ap.add_argument("--clips", type=int, default=DEFAULT_CLIPS,
                    help="--efficiency: number of measured clips (default %(default)s)")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help="--efficiency: untimed warmup clips (default %(default)s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="--efficiency: subset seed, keep identical across models (default %(default)s)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    torch.backends.cudnn.enabled = False

    from qwen_asr import Qwen3ASRModel
    from utils.registry import MODEL_BY_KEY, get_dataset
    from utils.inference_loop import run_transcription

    model_id = MODEL_BY_KEY[MODEL_KEY].model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device} ...")
    load_timing = []
    with timed(load_timing):
        if device == "cuda":
            model = Qwen3ASRModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto", max_new_tokens=512)
        else:
            model = Qwen3ASRModel.from_pretrained(model_id, device_map="cpu", max_new_tokens=512)
    print(f"Model loaded in {load_timing[0]:.1f}s.\n")

    audio_col = get_dataset(args.dataset).audio_col
    decode_kwargs = {"language": "English", "max_new_tokens": 512}

    if args.efficiency:
        run_efficiency_benchmark(
            MODEL_KEY, args.dataset,
            lambda s: transcribe_qwen3(model, s, audio_col),
            n_clips=args.clips, warmup=args.warmup, seed=args.seed,
            model_load_seconds=load_timing[0], param_count=count_parameters(model),
            extra={"decode_kwargs": decode_kwargs},
        )
        print("Done.")
        return

    run_transcription(MODEL_KEY, args.dataset,
                      transcribe_one=lambda s: transcribe_qwen3(model, s, audio_col),
                      manifest_extra={"decode_kwargs": decode_kwargs})
    print("Done.")


if __name__ == "__main__":
    main()
