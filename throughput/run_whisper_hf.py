"""Batched Whisper benchmark using Hugging Face's official pipeline API.

OpenAI's reference ``model.transcribe`` API is single-file and remains the
batch-1 latency implementation in ``whisper_asr/run_whisper.py``.  Transformers
is used here solely to expose a documented, common PyTorch batch interface; the
runtime is recorded explicitly so these results are never mislabeled as the
OpenAI reference implementation.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from throughput.common_cli import parser, run_kwargs
from utils.throughput import run_throughput_benchmark

MODELS = ("tiny", "base", "small", "medium", "large", "large_v3_turbo")
HF_IDS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "large_v3_turbo": "openai/whisper-large-v3-turbo",
}


def main() -> None:
    args = parser("Whisper HF offline throughput", MODELS).parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model_id = HF_IDS[args.model]
    print(f"Loading {model_id} in float16 on cuda:0 ...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        .to("cuda:0")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_id)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=0,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t0

    def transcribe(paths: list[str]) -> list[str]:
        with torch.inference_mode():
            outputs = asr(
                paths,
                batch_size=len(paths),
                num_workers=0,
                return_timestamps=False,
                generate_kwargs={
                    "language": "english",
                    "task": "transcribe",
                    "do_sample": False,
                    "num_beams": 1,
                },
            )
        if isinstance(outputs, dict):
            outputs = [outputs]
        return [str(o.get("text", "")).strip() for o in outputs]

    run_throughput_benchmark(
        args.model,
        args.dataset,
        transcribe,
        model=model,
        model_load_seconds=load_seconds,
        runtime="huggingface_transformers_whisper_pipeline",
        runtime_config={
            "dtype": "float16",
            "device": "cuda:0",
            "attention": "sdpa",
            "language": "english",
            "task": "transcribe",
            "input_policy": "short_form_at_most_30_seconds_no_external_chunking",
            "timestamps": False,
            "do_sample": False,
            "num_beams": 1,
            "num_workers": 0,
            "checkpoint": model_id,
        },
        **run_kwargs(args),
    )


if __name__ == "__main__":
    main()
