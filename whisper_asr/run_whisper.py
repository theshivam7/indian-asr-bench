"""
Stage 1: ASR transcription — Whisper family (openai-whisper engine).

One driver for all openai-whisper checkpoints, replacing the former identical
task1/2/3 scripts. The model set and checkpoint ids come from utils.registry.

Usage:
    python whisper_asr/run_whisper.py --model base            # TIE (default)
    python whisper_asr/run_whisper.py --model large_v3_turbo --dataset svarah
    python whisper_asr/run_whisper.py --model base --efficiency   # timing, not transcripts

Writes results/<dataset>/stage1_raw_transcripts/wer_<model>_raw.csv, or with
--efficiency, results/<dataset>/efficiency/efficiency_<model>.json (see
utils/efficiency.py for the measurement protocol).
Do NOT re-run unless you need new transcriptions; then run normalize_and_score.py.
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import whisper

from utils.efficiency import (DEFAULT_CLIPS, DEFAULT_SEED, DEFAULT_WARMUP, count_parameters,
                              run_efficiency_benchmark, timed)
from utils.registry import MODEL_BY_KEY, MODEL_SPECS, get_dataset
from utils.transcribe import transcribe_sample
from utils.inference_loop import run_transcription

warnings.filterwarnings("ignore")

WHISPER_MODELS = tuple(m.key for m in MODEL_SPECS if m.engine == "openai_whisper")


def main() -> None:
    ap = argparse.ArgumentParser(description="Whisper (openai-whisper) transcription driver.")
    ap.add_argument("--model", required=True, choices=WHISPER_MODELS,
                    help=f"openai-whisper model key ({', '.join(WHISPER_MODELS)})")
    ap.add_argument("--dataset", default="tie", help="dataset key (tie, svarah, ...)")
    ap.add_argument("--efficiency", action="store_true",
                    help="measure speed/memory on a seeded clip subset instead of transcribing the split")
    ap.add_argument("--clips", type=int, default=DEFAULT_CLIPS,
                    help="--efficiency: number of measured clips (default %(default)s)")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help="--efficiency: untimed warmup clips (default %(default)s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="--efficiency: subset seed, keep identical across models (default %(default)s)")
    args = ap.parse_args()

    spec = MODEL_BY_KEY[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading whisper '{spec.model_id}' ({spec.display}) on {device} ...")
    load_timing: list[float] = []
    with timed(load_timing):
        model = whisper.load_model(spec.model_id, device=device)
    print(f"Model loaded in {load_timing[0]:.1f}s.\n")

    transcribe_kw = {"language": "en"}
    if device == "cpu":
        transcribe_kw["fp16"] = False

    audio_col = get_dataset(args.dataset).audio_col

    def transcribe_one(sample: dict) -> str:
        return transcribe_sample(model, sample, transcribe_kw, audio_col)

    if args.efficiency:
        run_efficiency_benchmark(
            args.model, args.dataset, transcribe_one,
            n_clips=args.clips, warmup=args.warmup, seed=args.seed,
            model_load_seconds=load_timing[0], param_count=count_parameters(model),
            extra={"decode_kwargs": {**transcribe_kw, "engine_defaults": "openai-whisper"}},
        )
        print("\nDone.")
        return

    run_transcription(
        args.model, args.dataset,
        transcribe_one=transcribe_one,
        # openai-whisper defaults apply for everything not listed here: greedy decoding with
        # temperature fallback (0.0->1.0, stochastic above 0) and condition_on_previous_text=True.
        manifest_extra={"decode_kwargs": {**transcribe_kw, "engine_defaults": "openai-whisper"}},
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
