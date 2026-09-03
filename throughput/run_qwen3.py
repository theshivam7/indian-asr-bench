"""Official qwen-asr Transformers backend batch-throughput benchmark."""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from throughput.common_cli import parser, run_kwargs
from utils.registry import MODEL_BY_KEY
from utils.throughput import parse_batch_sizes, run_throughput_benchmark


def main() -> None:
    args = parser("Qwen3-ASR Transformers offline throughput", ("qwen3",)).parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    warnings.filterwarnings("ignore")

    from qwen_asr import Qwen3ASRModel

    sizes = parse_batch_sizes(args.batch_sizes)
    model_id = MODEL_BY_KEY[args.model].model_id
    print(f"Loading {model_id} in bfloat16 on cuda:0 ...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=max(sizes),
        max_new_tokens=512,
        attn_implementation="sdpa",
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t0

    def transcribe(paths: list[str]) -> list[str]:
        with torch.inference_mode():
            outputs = model.transcribe(
                audio=paths,
                language=["English"] * len(paths),
                return_time_stamps=False,
            )
        return [(o.text if hasattr(o, "text") else str(o)).strip() for o in outputs]

    run_throughput_benchmark(
        args.model,
        args.dataset,
        transcribe,
        model=model,
        model_load_seconds=load_seconds,
        runtime="qwen_asr_transformers_backend",
        runtime_config={
            "dtype": "bfloat16",
            "device_map": "cuda:0",
            "attention": "sdpa",
            "language": "English",
            "timestamps": False,
            "max_new_tokens": 512,
            "max_inference_batch_size": max(sizes),
            "checkpoint": model_id,
        },
        **run_kwargs(args),
    )


if __name__ == "__main__":
    main()
