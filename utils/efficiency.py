"""GPU measurement helpers shared by the Stage-1 drivers and the throughput protocol.

Each engine lives in its own conda env (openai-whisper, NeMo and Qwen3 cannot be
imported into one process), so nothing here imports an engine, only torch when it
is present. ``utils.throughput`` builds the offline batch protocol on these
helpers; the Stage-1 drivers use ``timed`` for model-load time and
``cudnn_disabled`` around model load.
"""

import hashlib
import os
import platform
import subprocess
import time
from contextlib import contextmanager

import numpy as np

# torch is present in every engine env but not in the CPU-only analysis venv. Guard
# the import so `python -c "import utils.efficiency"` works without the GPU stack.
try:
    import torch
except ImportError:  # pragma: no cover - exercised only in a torch-free env
    torch = None

MIB = 1024 * 1024


# ============================================================================
# Device and timing probes
# ============================================================================

def cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def synchronize_device() -> None:
    """Block until all queued CUDA work has finished (no-op on CPU).

    CUDA kernel launches are asynchronous, so a timer that does not synchronize
    measures how fast Python can enqueue work, not how fast the GPU runs it.
    """
    if cuda_available():
        torch.cuda.synchronize()


@contextmanager
def timed(sink: list):
    """Time a region in seconds (CUDA-synchronized) and append it to `sink`."""
    synchronize_device()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        synchronize_device()
        sink.append(time.perf_counter() - t0)


@contextmanager
def cudnn_disabled():
    """Turn cuDNN off inside the block and restore the previous setting after.

    NeMo and Qwen3 model loads can hit CUDNN_STATUS_NOT_INITIALIZED on some
    clusters. Restoring the setting matters: leaving cuDNN off would run every
    clip on a different backend from the Whisper drivers, which never touch it.
    """
    if torch is None:
        yield
        return
    original = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = original


def count_parameters(model) -> int | None:
    """Total parameter count for a loaded model, or None if it cannot be read.

    Engines wrap their nn.Module differently (openai-whisper and NeMo expose
    ``.parameters()`` directly, LLM wrappers often hide it one attribute down),
    so try the common shapes rather than special-casing each engine here.
    """
    for attr in (None, "model", "module", "_model"):
        obj = model if attr is None else getattr(model, attr, None)
        params = getattr(obj, "parameters", None) if obj is not None else None
        if not callable(params):
            continue
        try:
            return int(sum(p.numel() for p in params()))
        except Exception:
            continue
    return None


def _nvidia_driver_version() -> str:
    """Driver version from nvidia-smi, or "" when it is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
    except Exception:
        return ""


def hardware_provenance() -> dict:
    """Everything needed to interpret a timing number later.

    RTF and latency are properties of a (model, hardware, software) triple, not of
    a model. Two models measured on different GPUs must never be put in the same
    table, so record the GPU/driver/torch identity next to every measurement and
    let the aggregator check that they agree.
    """
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "device": "cuda" if cuda_available() else "cpu",
    }
    if torch is None:
        info["torch"] = ""
        return info
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda or ""
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        info["cudnn"] = None
    if cuda_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_total_mem_mib"] = round(props.total_memory / MIB, 1)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
        info["nvidia_driver"] = _nvidia_driver_version()
    return info


# ============================================================================
# Subset selection
# ============================================================================

def select_subset(ds, n_clips: int, seed: int):
    """Deterministic clip subset of an eval split. Returns (subset_ds, indices).

    Same dataset + same seed + same n_clips gives the same row indices in every
    engine env and on every machine (numpy's PCG64 stream is stable across
    platforms), which is what makes the per-model runs comparable at all.
    Indices are sorted so the subset preserves dataset order.

    ``flatten_indices()`` is required: ``select()`` leaves a lazy indices overlay,
    and raw arrow access elsewhere in the pipeline indexes physical rows, which
    would misalign with logical rows under an overlay (same trap documented in
    utils/datasets.py:_apply_row_filter).
    """
    n_clips = int(n_clips)
    if n_clips < 1:
        raise ValueError("n_clips must be positive")
    n = len(ds)
    if n < 1:
        raise ValueError("cannot select a benchmark subset from an empty dataset")
    k = min(n_clips, n)
    rng = np.random.default_rng(seed)
    indices = sorted(int(i) for i in rng.choice(n, size=k, replace=False))
    return ds.select(indices).flatten_indices(), indices


def subset_fingerprint(ids: list[str]) -> str:
    """Short stable hash of the measured clip IDs.

    Two models are only comparable if they were timed on the same audio. The
    fingerprint travels in each JSON so the aggregator can refuse to silently
    mix subsets instead of the reader discovering it in review.
    """
    h = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return h[:12]
