"""Native NeMo batch-throughput benchmark for the two Parakeet checkpoints."""

import logging
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from throughput.common_cli import parser, run_kwargs
from utils.registry import MODEL_BY_KEY
from utils.throughput import run_throughput_benchmark

MODELS = ("parakeet", "parakeet_ctc")


def main() -> None:
    args = parser("Parakeet NeMo offline throughput", MODELS).parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    logging.getLogger("nemo_logger").setLevel(logging.WARNING)
    logging.getLogger("nemo").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")

    import nemo.collections.asr as nemo_asr

    model_id = MODEL_BY_KEY[args.model].model_id
    print(f"Loading {model_id} on cuda:0 ...", flush=True)
    # This avoids the known NSCC CUDNN_STATUS_NOT_INITIALIZED during the LSTM
    # weight-transfer step. cuDNN is restored before any measured inference.
    original_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_id).cuda().eval()
    finally:
        torch.backends.cudnn.enabled = original_cudnn
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t0

    def transcribe(paths: list[str]) -> list[str]:
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True),
        ):
            outputs = model.transcribe(
                audio=paths,
                batch_size=len(paths),
                num_workers=0,
                timestamps=False,
                verbose=False,
            )
        return [(o.text if hasattr(o, "text") else str(o)).strip() for o in outputs]

    run_throughput_benchmark(
        args.model,
        args.dataset,
        transcribe,
        model=model,
        model_load_seconds=load_seconds,
        runtime="nvidia_nemo_native_transcribe",
        runtime_config={
            "dtype": "automatic_mixed_precision_float16",
            "device": "cuda:0",
            "decoder": "checkpoint_default_greedy",
            "timestamps": False,
            "num_workers": 0,
            "verbose": False,
            "duration_presort": "external_common_order",
            "cudnn_disabled_during_load_only": True,
            "cudnn_enabled_during_inference": bool(torch.backends.cudnn.enabled),
            "checkpoint": model_id,
        },
        **run_kwargs(args),
    )


if __name__ == "__main__":
    main()
