"""Shared efficiency-measurement helpers for the Stage-1 engines.

The accuracy benchmark answers "which model is most correct". The paper also
claims that small specialized models stay *competitive*, which is a cost claim,
so it needs cost numbers measured under one protocol: real-time factor, per-clip
latency percentiles, throughput, peak GPU memory, model load time and parameter
count.

Each engine lives in its own conda env (openai-whisper, NeMo and Qwen3 cannot be
imported into one process), so a single script can never benchmark all nine
models. This module is therefore engine-agnostic: it imports no engine, only
torch (optionally) plus the dataset adapter. Every driver calls
``run_efficiency_benchmark(...)`` with the same per-clip callable it already uses
for transcription, and writes one JSON per model. ``analysis/compare_efficiency.py``
merges those JSONs on CPU into the paper table.

Measurement protocol, stated once here because the numbers are meaningless
without it:

  * **Timed region** = the driver's own per-clip transcribe callable, which for
    every engine includes its audio preprocessing (decode + temp 16 kHz WAV
    write). That cost is real and is paid identically by all three engines, so
    leaving it in keeps the comparison honest rather than flattering whichever
    engine has the cheapest frontend.
  * **Batch size 1**, the interactive/streaming setting. Per-clip latency is only
    defined at batch 1, and throughput here is therefore *latency-bound*
    throughput, not the peak a batched server would reach. Engines whose driver
    batches in production (Parakeet) will look worse on throughput than they
    would when batched: say so in the paper rather than quietly batching one
    engine and not the others.
  * ``torch.cuda.synchronize()`` brackets every timed region. Without it the
    timer measures kernel *launches*, not kernel execution.
  * **Warmup clips are untimed.** The first call into a CUDA model compiles
    kernels, allocates workspaces and warms the caching allocator, which can cost
    seconds and would poison the mean.
  * **Fixed seeded subset** of the eval split, identical for every model, with a
    fingerprint recorded in each JSON so the aggregator can prove all models were
    measured on the same audio.

RTF convention: ``rtf = processing_seconds / audio_seconds``. Lower is faster;
below 1.0 is faster than real time. Reported both as an aggregate (total over
total, what a batch job experiences) and as a per-clip median.

Output: results/<dataset>/efficiency/efficiency_<model>.json  (summary)
        results/<dataset>/efficiency/efficiency_<model>_clips.csv  (per clip)
"""

import json
import hashlib
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.datasets import load_eval, extract_ids
from utils.io_helpers import (
    efficiency_dir,
    positive_float,
    probe_audio_duration,
    sample_id,
    text_value,
)
from utils.registry import MODEL_BY_KEY

# torch is present in every engine env but not in the CPU-only analysis venv, and this
# module is imported by nothing on the CPU path today. Guard the import anyway so a
# laptop smoke test (`python -c "import utils.efficiency"`) works without the GPU stack.
try:
    import torch
except ImportError:  # pragma: no cover - exercised only in a torch-free env
    torch = None

MIB = 1024 * 1024
DEFAULT_CLIPS = 200
DEFAULT_WARMUP = 3
DEFAULT_SEED = 42

# Batch size is fixed rather than exposed as a flag: a cross-engine latency table is
# only comparable if every engine ran at the same batch size, and per-clip latency
# percentiles are undefined for batched inference.
BATCH_SIZE = 1


# ============================================================================
# Device, timing and memory probes
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
    """Time a region in seconds (CUDA-synchronized) and append it to `sink`.

    Used for both the per-clip inference calls and the one-shot model load, so
    every duration in the report comes from the same clock and the same
    synchronization discipline.
    """
    synchronize_device()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        synchronize_device()
        sink.append(time.perf_counter() - t0)


def reset_peak_gpu_memory() -> None:
    """Zero the peak-memory counters, keeping already-resident weights counted.

    Called after warmup so the reported peak reflects steady-state inference
    (weights + activations) rather than one-off warmup allocations.
    """
    if cuda_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory() -> dict:
    """Peak GPU memory since the last reset, in MiB. Empty dict on CPU.

    Both figures are reported because they answer different questions:
    ``allocated`` is what the model actually held (tensor bytes, the number worth
    quoting as a model's footprint), ``reserved`` is what the caching allocator
    took from the driver and is what determines whether the job fits on a card.
    """
    if not cuda_available():
        return {}
    return {
        "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
        "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
    }


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
# Subset selection and statistics
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


def summarize_timings(latencies: list[float], durations: list[float]) -> dict:
    """Aggregate per-clip latencies + clip durations into the reported metrics.

    Clips whose duration could not be determined still count towards latency
    (a measured wall time is a measured wall time) but are excluded from the
    audio-second totals, so RTF and throughput are never computed against a
    duration of zero.
    """
    lat = np.asarray(latencies, dtype=float)
    dur = np.asarray(durations, dtype=float)
    usable = np.isfinite(dur) & (dur > 0)

    audio_total = float(dur[usable].sum())
    proc_total = float(lat.sum())
    proc_usable = float(lat[usable].sum())

    metrics = {
        "n_clips_timed": int(lat.size),
        "n_clips_with_duration": int(usable.sum()),
        "audio_seconds_total": round(audio_total, 2),
        "processing_seconds_total": round(proc_total, 3),
        "latency_mean_s": round(float(lat.mean()), 4) if lat.size else None,
        "latency_p50_s": round(float(np.percentile(lat, 50)), 4) if lat.size else None,
        "latency_p90_s": round(float(np.percentile(lat, 90)), 4) if lat.size else None,
        "latency_p95_s": round(float(np.percentile(lat, 95)), 4) if lat.size else None,
        "latency_min_s": round(float(lat.min()), 4) if lat.size else None,
        "latency_max_s": round(float(lat.max()), 4) if lat.size else None,
    }
    if audio_total > 0:
        # Aggregate RTF (total over total) is what a batch job experiences; the per-clip
        # median is robust to one pathological clip and is the better "typical clip" number.
        per_clip_rtf = lat[usable] / dur[usable]
        metrics["rtf"] = round(proc_usable / audio_total, 4)
        metrics["rtf_p50"] = round(float(np.percentile(per_clip_rtf, 50)), 4)
        metrics["throughput_audio_s_per_s"] = round(audio_total / proc_usable, 3) if proc_usable else None
    else:
        metrics["rtf"] = None
        metrics["rtf_p50"] = None
        metrics["throughput_audio_s_per_s"] = None
    return metrics


# ============================================================================
# The benchmark itself
# ============================================================================

def _clip_duration(sample: dict, spec) -> float | None:
    """Clip length in seconds, from the spec's duration column or the audio header.

    Never called inside a timed region: probing the audio would otherwise be
    charged to the model.
    """
    if spec.duration_col:
        value = positive_float(sample.get(spec.duration_col))
        if value is not None:
            return value
    return probe_audio_duration(sample.get(spec.audio_col))


def run_efficiency_benchmark(model_key: str, dataset_key: str, transcribe_one, *,
                             n_clips: int = DEFAULT_CLIPS, warmup: int = DEFAULT_WARMUP,
                             seed: int = DEFAULT_SEED, model_load_seconds: float | None = None,
                             param_count: int | None = None,
                             extra: dict | None = None) -> str:
    """Measure one model on a seeded subset and write its efficiency JSON + per-clip CSV.

    ``transcribe_one(sample) -> str`` is the driver's existing single-argument
    per-clip callable (the same one it passes to utils.inference_loop), so the
    thing measured here is exactly the thing the benchmark runs. Returns the path
    to the JSON summary.
    """
    mspec = MODEL_BY_KEY.get(model_key)
    ds, spec = load_eval(dataset_key)

    all_ids = extract_ids(ds, spec)
    subset, indices = select_subset(ds, n_clips, seed)
    ids = [all_ids[i] for i in indices]
    fingerprint = subset_fingerprint(ids)

    n_warmup = max(0, min(int(warmup), len(subset)))
    print(f"--- efficiency: {model_key} on {spec.display} [{spec.splits['eval']}] ---")
    print(f"  subset: {len(subset)}/{len(ds)} clips, seed={seed}, fingerprint={fingerprint}")
    print(f"  warmup: {n_warmup} untimed clips, batch size {BATCH_SIZE}, "
          f"device={'cuda' if cuda_available() else 'cpu'}")

    # Warmup runs the first few clips of the measured subset and throws the results away.
    # Drawing them from the subset (rather than from spare rows) keeps the measured set
    # exactly the seeded subset, independent of how many warmup clips were requested.
    for i in range(n_warmup):
        transcribe_one(subset[i])

    reset_peak_gpu_memory()

    latencies: list[float] = []
    durations: list[float] = []
    rows: list[dict] = []
    for i, sample in enumerate(tqdm(subset, desc=f"{dataset_key}:{model_key}:efficiency")):
        duration = _clip_duration(sample, spec)
        sink: list[float] = []
        with timed(sink):
            hyp = transcribe_one(sample)
        elapsed = sink[0]
        latencies.append(elapsed)
        durations.append(duration if duration else float("nan"))
        rows.append({
            "ID": ids[i] if i < len(ids) else sample_id(sample, spec),
            "duration_seconds": round(duration, 3) if duration else "",
            "latency_seconds": round(elapsed, 4),
            "rtf": round(elapsed / duration, 4) if duration else "",
            "hypothesis_chars": len(text_value(hyp)),
        })

    metrics = {**summarize_timings(latencies, durations), **peak_gpu_memory()}

    out_dir = efficiency_dir(dataset_key)
    clips_path = os.path.join(out_dir, f"efficiency_{model_key}_clips.csv")
    pd.DataFrame(rows).to_csv(clips_path, index=False)

    report = {
        "model_key": model_key,
        "model_id": mspec.model_id if mspec else "",
        "display": mspec.display if mspec else model_key,
        "engine": mspec.engine if mspec else "",
        "arch_class": mspec.arch_class if mspec else "",
        "params_registry": mspec.params if mspec else "",
        "dataset": dataset_key,
        "hf_id": spec.hf_id,
        "hf_revision": spec.hf_revision,
        "split": spec.splits["eval"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "batch_size": BATCH_SIZE,
            "n_clips_requested": int(n_clips),
            "n_warmup": n_warmup,
            "seed": int(seed),
            "subset_fingerprint": fingerprint,
            "rtf_convention": "processing_seconds / audio_seconds; lower is faster, <1 is faster than real time",
            "timed_region": "engine transcribe call including its audio decode and temp WAV write",
            "gpu_memory_convention": "torch.cuda.max_memory_allocated and max_memory_reserved, "
                                     "peak since the post-warmup reset",
        },
        "model_load_seconds": (round(model_load_seconds, 3)
                               if model_load_seconds is not None else None),
        "param_count": param_count,
        "metrics": metrics,
        "hardware": hardware_provenance(),
        "per_clip_csv": os.path.basename(clips_path),
        **(extra or {}),
    }
    json_path = os.path.join(out_dir, f"efficiency_{model_key}.json")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    rtf = metrics.get("rtf")
    print(f"\n  RTF {rtf if rtf is not None else 'n/a'} (lower is faster)"
          f" | latency p50 {metrics['latency_p50_s']}s p95 {metrics['latency_p95_s']}s"
          f" | throughput {metrics.get('throughput_audio_s_per_s')} audio-s per s")
    if "peak_gpu_allocated_mib" in metrics:
        print(f"  peak GPU: {metrics['peak_gpu_allocated_mib']} MiB allocated, "
              f"{metrics['peak_gpu_reserved_mib']} MiB reserved")
    else:
        print("  peak GPU: not measured (CPU run)")
    print(f"  Saved: {json_path}\n         {clips_path}")
    print(f"  Aggregate with: python analysis/compare_efficiency.py --dataset {dataset_key}")
    return json_path
