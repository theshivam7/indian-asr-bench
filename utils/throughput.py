"""Shared, engine-independent saturated-throughput benchmark.

The existing :mod:`utils.efficiency` protocol remains the batch-1, end-to-end
single-stream latency measurement.  This module measures the complementary
offline scenario: short-form audio is converted to identical 16 kHz mono WAVs
before the clock starts, clips are sorted by duration for every model, and the
native batch API is swept until the configured maximum or an OOM.

The timed region still includes WAV decoding, feature extraction, model forward
passes, and decoding.  It excludes one-time dataset download, model load, and
WAV conversion; those are recorded separately and must not be mixed into steady-
state serving throughput.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from utils.datasets import extract_ids, load_eval
from utils.efficiency import (
    count_parameters,
    hardware_provenance,
    select_subset,
    subset_fingerprint,
    synchronize_device,
)
from utils.io_helpers import audio_to_wav_16k, probe_audio_duration, text_value, throughput_dir
from utils.normalize import normalize_for_mode
from utils.registry import MODEL_BY_KEY
from utils.wer_compute import compute_corpus_cer, compute_corpus_wer

try:
    import torch
except ImportError:  # pragma: no cover - GPU environments always include torch
    torch = None

PROTOCOL_VERSION = "offline-throughput-v1"
DEFAULT_CLIPS = 512
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_WARMUP_BATCHES = 3
DEFAULT_REPEATS = 3
DEFAULT_SEED = 42
DEFAULT_TELEMETRY_INTERVAL_MS = 250
DEFAULT_MAX_WER_DELTA_PP = 0.10
DEFAULT_MAX_CLIP_SECONDS = 30.0
THROUGHPUT_TIE_TOLERANCE = 0.01


@dataclass(frozen=True)
class PreparedClip:
    clip_id: str
    path: str
    duration_s: float
    reference: str


def parse_batch_sizes(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse, validate, sort, and de-duplicate a batch-size specification."""
    raw = value.split(",") if isinstance(value, str) else value
    sizes = sorted({int(v) for v in raw})
    if not sizes or sizes[0] < 1:
        raise ValueError("batch sizes must be positive integers")
    if 1 not in sizes:
        raise ValueError("batch-size sweep must include 1 for the quality baseline")
    return tuple(sizes)


def package_versions(names: Sequence[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            pass
    return versions


def _git_commit() -> str:
    supplied = os.environ.get("GIT_COMMIT", "").strip()
    if supplied:
        return supplied
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _declared_duration(ds, spec, index: int) -> float | None:
    if not spec.duration_col:
        return probe_audio_duration(ds[index][spec.audio_col])
    try:
        value = float(ds.data.column(spec.duration_col)[index].as_py())
    except (TypeError, ValueError):
        value = float("nan")
    if math.isfinite(value) and value > 0:
        return value
    return probe_audio_duration(ds[index][spec.audio_col])


def prepare_subset(
    dataset_key: str,
    n_clips: int,
    seed: int,
    wav_dir: str,
    max_clip_seconds: float = DEFAULT_MAX_CLIP_SECONDS,
) -> tuple[list[PreparedClip], dict]:
    """Select and pre-stage the identical, model-independent workload.

    Selection occurs before duration sorting.  This preserves the existing
    seeded-subset contract while making all engines see the same efficient
    duration buckets.  Only non-empty gold references are eligible.
    """
    ds, spec = load_eval(dataset_key)
    # Read only the Arrow text column here; indexing every full row would decode
    # or copy the audio of clips that are never selected.
    gold_refs = ds.data.column(spec.gold_ref_col).to_pylist()
    if n_clips < 1:
        raise ValueError("n_clips must be positive")
    eligible = []
    for i, ref in enumerate(gold_refs):
        if not normalize_for_mode("transcript_clean", text_value(ref)):
            continue
        duration = _declared_duration(ds, spec, i)
        # Metadata can be rounded. Probe the real audio near the boundary so a
        # nominal 30.0 s clip cannot silently enter Whisper's long-form path.
        if (
            duration is not None
            and spec.duration_col
            and max_clip_seconds - 1.0 <= duration <= max_clip_seconds
        ):
            duration = probe_audio_duration(ds[i][spec.audio_col])
        if duration is not None and duration <= max_clip_seconds:
            eligible.append(i)
    if not eligible:
        raise ValueError(f"dataset '{dataset_key}' has no non-empty references")
    if len(eligible) < n_clips:
        raise ValueError(
            f"dataset '{dataset_key}' has only {len(eligible)} eligible clips "
            f"at <= {max_clip_seconds:.1f}s, fewer than requested {n_clips}"
        )
    eligible_ds = ds.select(eligible).flatten_indices()
    subset, selected_eligible_indices = select_subset(eligible_ds, n_clips, seed)
    selected_original_indices = [eligible[i] for i in selected_eligible_indices]
    ids = extract_ids(subset, spec)

    Path(wav_dir).mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedClip] = []
    prep_start = time.perf_counter()
    for i in range(len(subset)):
        sample = subset[i]
        wav_path = os.path.join(wav_dir, f"{i:05d}.wav")
        audio_to_wav_16k(sample[spec.audio_col], wav_path)
        # Use the staged artifact itself as the RTF denominator. Metadata can be
        # rounded; frame_count / sample_rate is the exact audio actually served.
        with wave.open(wav_path, "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        if duration > max_clip_seconds:
            raise ValueError(
                f"staged clip '{ids[i]}' is {duration:.3f}s, above the fixed "
                f"{max_clip_seconds:.1f}s short-form limit"
            )
        prepared.append(
            PreparedClip(
                clip_id=ids[i],
                path=wav_path,
                duration_s=float(duration),
                reference=text_value(sample.get(spec.gold_ref_col)),
            )
        )
    prep_seconds = time.perf_counter() - prep_start

    # One identical, public rule for every model. Stable ID tie-break makes the
    # order deterministic even when many short clips share a duration.
    prepared.sort(key=lambda c: (c.duration_s, c.clip_id))
    manifest = {
        "dataset": dataset_key,
        "dataset_id": spec.hf_id,
        "dataset_revision": spec.hf_revision,
        "split": spec.splits["eval"],
        "n_clips": len(prepared),
        "eligible_clips": len(eligible),
        "eligibility": "transcript_clean_reference_nonempty_and_duration_at_most_limit",
        "seed": int(seed),
        "max_clip_seconds": float(max_clip_seconds),
        "selection_indices": selected_original_indices,
        "ordered_clip_ids": [c.clip_id for c in prepared],
        "subset_fingerprint": subset_fingerprint(ids),
        "ordered_workload_fingerprint": subset_fingerprint(
            [c.clip_id for c in prepared]
        ),
        "audio_seconds_total": round(sum(c.duration_s for c in prepared), 3),
        "duration_min_s": round(min(c.duration_s for c in prepared), 3),
        "duration_p50_s": round(
            float(np.percentile([c.duration_s for c in prepared], 50)), 3
        ),
        "duration_p95_s": round(
            float(np.percentile([c.duration_s for c in prepared], 95)), 3
        ),
        "duration_max_s": round(max(c.duration_s for c in prepared), 3),
        "prestage_seconds": round(prep_seconds, 3),
        "ordering": "ascending_duration_then_clip_id",
        "audio_format": "mono_pcm_s16le_16000_hz_wav",
    }
    audio_hash = hashlib.sha256()
    for clip in prepared:
        audio_hash.update(clip.clip_id.encode("utf-8"))
        audio_hash.update(b"\0")
        with open(clip.path, "rb") as wav_file:
            for block in iter(lambda: wav_file.read(1024 * 1024), b""):
                audio_hash.update(block)
        audio_hash.update(b"\0")
    manifest["audio_content_sha256"] = audio_hash.hexdigest()
    return prepared, manifest


def _percentile(values: Sequence[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 4) if values else None


def summarize_trial(
    processing_seconds: float,
    batch_latencies: Sequence[float],
    batch_counts: Sequence[int],
    audio_seconds: float,
) -> dict:
    """Compute one repeat's latency/throughput metrics."""
    n = int(sum(batch_counts))
    completion_latencies = [
        lat for lat, count in zip(batch_latencies, batch_counts) for _ in range(count)
    ]
    return {
        "n_clips": n,
        "n_batches": len(batch_latencies),
        "audio_seconds": round(float(audio_seconds), 3),
        "processing_seconds": round(float(processing_seconds), 4),
        "rtf": round(processing_seconds / audio_seconds, 6),
        "rtfx_audio_s_per_s": round(audio_seconds / processing_seconds, 3),
        "utterances_per_s": round(n / processing_seconds, 3),
        "batch_latency_p50_s": _percentile(batch_latencies, 50),
        "batch_latency_p95_s": _percentile(batch_latencies, 95),
        "completion_latency_p50_s": _percentile(completion_latencies, 50),
        "completion_latency_p95_s": _percentile(completion_latencies, 95),
    }


def _gpu_target() -> str | None:
    """Resolve the physical GPU used by this process for nvidia-smi telemetry."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
        for line in out.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) >= 2 and fields[0] == str(os.getpid()):
                return fields[1]
    except Exception:
        pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return visible or "0"


class NvidiaSmiSampler:
    """Low-overhead process-level GPU telemetry sampled by nvidia-smi."""

    def __init__(self, interval_ms: int = DEFAULT_TELEMETRY_INTERVAL_MS):
        self.interval_ms = int(interval_ms)
        self.process = None
        self.handle = None
        self.path = None

    def start(self) -> None:
        try:
            fd, self.path = tempfile.mkstemp(prefix="asr_gpu_", suffix=".csv")
            self.handle = os.fdopen(fd, "w")
            query = (
                "utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm"
            )
            cmd = [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                "-lms",
                str(self.interval_ms),
            ]
            target = _gpu_target()
            if target:
                cmd.extend(["-i", target])
            self.process = subprocess.Popen(
                cmd,
                stdout=self.handle,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            self.stop()

    def stop(self) -> dict:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.handle is not None:
            self.handle.close()
        rows = []
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, newline="") as f:
                    for row in csv.reader(f):
                        if len(row) < 5:
                            continue
                        try:
                            rows.append([float(v.strip()) for v in row[:5]])
                        except ValueError:
                            continue
            finally:
                os.unlink(self.path)
        self.process = self.handle = self.path = None
        if not rows:
            return {"telemetry_samples": 0}
        a = np.asarray(rows, dtype=float)
        return {
            "telemetry_samples": len(rows),
            "gpu_util_mean_pct": round(float(a[:, 0].mean()), 2),
            "gpu_util_p50_pct": round(float(np.percentile(a[:, 0], 50)), 2),
            "gpu_util_p95_pct": round(float(np.percentile(a[:, 0], 95)), 2),
            "memory_util_mean_pct": round(float(a[:, 1].mean()), 2),
            "device_memory_peak_mib": round(float(a[:, 2].max()), 1),
            "power_mean_w": round(float(a[:, 3].mean()), 2),
            "power_p95_w": round(float(np.percentile(a[:, 3], 95)), 2),
            "sm_clock_mean_mhz": round(float(a[:, 4].mean()), 1),
        }


def _torch_peak_memory() -> dict:
    if torch is None or not torch.cuda.is_available():
        return {}
    return {
        "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }


def _is_oom(exc: BaseException) -> bool:
    if torch is not None and isinstance(
        exc, getattr(torch.cuda, "OutOfMemoryError", ())
    ):
        return True
    return "out of memory" in str(exc).lower()


def _quality(prepared: Sequence[PreparedClip], hypotheses: Sequence[str]) -> dict:
    refs = [normalize_for_mode("transcript_clean", c.reference) for c in prepared]
    hyps = [normalize_for_mode("transcript_clean", h) for h in hypotheses]
    wer = compute_corpus_wer(refs, hyps)
    return {
        "corpus_wer_pct": round(100 * wer["corpus_wer"], 4),
        "corpus_cer_pct": round(100 * compute_corpus_cer(refs, hyps), 4),
        "empty_hypotheses": int(wer["num_empty_hyps"]),
        "hypothesis_sha256": hashlib.sha256(
            json.dumps(hyps, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "normalization_mode": "transcript_clean",
    }


def _median_metric(trials: Sequence[dict], key: str) -> float | None:
    vals = [t[key] for t in trials if t.get(key) is not None]
    return round(float(np.median(vals)), 4) if vals else None


def _variability(trials: Sequence[dict], key: str) -> dict:
    vals = np.asarray([t[key] for t in trials if t.get(key) is not None], dtype=float)
    if vals.size == 0:
        return {"min": None, "max": None, "cv_pct": None}
    mean = float(vals.mean())
    cv = float(vals.std(ddof=1) / mean * 100) if vals.size > 1 and mean else 0.0
    return {
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
        "cv_pct": round(cv, 3),
    }


def select_best_entry(
    valid_entries: Sequence[dict], tie_tolerance: float = THROUGHPUT_TIE_TOLERANCE
) -> dict:
    """Choose the smallest batch within 1% of the measured maximum RTFx."""
    if not valid_entries:
        raise ValueError("cannot select from an empty list")
    max_rtfx = max(e["median"]["rtfx_audio_s_per_s"] for e in valid_entries)
    near_max = [
        e
        for e in valid_entries
        if e["median"]["rtfx_audio_s_per_s"] >= max_rtfx * (1.0 - tie_tolerance)
    ]
    return min(near_max, key=lambda e: e["batch_size"])


def warmup_starts(n_clips: int, batch_size: int, warmup_batches: int) -> list[int]:
    """Spread warmups over short, median, and longest duration buckets."""
    if warmup_batches <= 0:
        return []
    last = n_clips - batch_size
    if last < 0:
        raise ValueError("warmup batch exceeds workload")
    if warmup_batches == 1:
        return [last]
    return [int(round(v)) for v in np.linspace(0, last, warmup_batches)]


def _write_result(result: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    os.replace(tmp, output_path)


def run_throughput_benchmark(
    model_key: str,
    dataset_key: str,
    transcribe_batch: Callable[[list[str]], list[str]],
    *,
    model,
    model_load_seconds: float,
    runtime: str,
    runtime_config: dict,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    n_clips: int = DEFAULT_CLIPS,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    repeats: int = DEFAULT_REPEATS,
    seed: int = DEFAULT_SEED,
    telemetry_interval_ms: int = DEFAULT_TELEMETRY_INTERVAL_MS,
    max_wer_delta_pp: float = DEFAULT_MAX_WER_DELTA_PP,
) -> dict:
    """Run and persist a quality-gated batch-size sweep."""
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("offline throughput benchmark requires one CUDA GPU")
    sizes = parse_batch_sizes(batch_sizes)
    if repeats < 1 or warmup_batches < 0:
        raise ValueError("repeats must be >=1 and warmup_batches must be >=0")
    if telemetry_interval_ms < 1:
        raise ValueError("telemetry_interval_ms must be positive")
    if max_wer_delta_pp < 0:
        raise ValueError("max_wer_delta_pp must be non-negative")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    out_dir = throughput_dir(dataset_key)
    output_path = os.path.join(out_dir, f"throughput_{model_key}.json")
    with tempfile.TemporaryDirectory(
        prefix=f"asr_{dataset_key}_", dir=os.environ.get("TMPDIR")
    ) as wav_dir:
        prepared, workload = prepare_subset(dataset_key, n_clips, seed, wav_dir)
        if max(sizes) > len(prepared):
            raise ValueError(
                f"largest batch size {max(sizes)} exceeds prepared workload {len(prepared)}"
            )
        audio_seconds = sum(c.duration_s for c in prepared)
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_key": model_key,
            "model_display": MODEL_BY_KEY[model_key].display,
            # Engine-specific drivers record the checkpoint they actually load.
            # Whisper's throughput runtime uses HF IDs rather than the short
            # aliases used by the existing openai-whisper latency runtime.
            "model_id": runtime_config.get(
                "checkpoint", MODEL_BY_KEY[model_key].model_id
            ),
            "runtime": runtime,
            "runtime_config": runtime_config,
            "model_load_seconds": round(float(model_load_seconds), 3),
            "parameter_count": count_parameters(model),
            "git_commit": _git_commit(),
            "source_sha256": os.environ.get("SOURCE_SHA256", "").strip(),
            "workload": workload,
            "protocol": {
                "scenario": "offline_saturated_throughput",
                "timed_region": "pre_staged_wav_read_plus_feature_extraction_plus_model_plus_decode",
                "batch_sizes": list(sizes),
                "warmup_batches_per_size": int(warmup_batches),
                "timed_repeats": int(repeats),
                "telemetry_interval_ms": int(telemetry_interval_ms),
                "quality_gate_max_abs_wer_delta_pp_vs_batch1": float(max_wer_delta_pp),
                "quality_gate_empty_hypotheses": "no_increase_vs_batch1",
                "quality_gate_repeat_outputs": "identical_hypothesis_hashes",
                "throughput_tie_rule": "smallest_batch_within_1pct_of_maximum_median_rtfx",
                "cpu_thread_limits": {
                    name: os.environ.get(name, "")
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            "hardware": hardware_provenance(),
            "software": package_versions(
                (
                    "torch",
                    "transformers",
                    "accelerate",
                    "safetensors",
                    "openai-whisper",
                    "nemo_toolkit",
                    "qwen-asr",
                    "datasets",
                    "pandas",
                    "numpy",
                    "soundfile",
                    "librosa",
                    "jiwer",
                    "num2words",
                )
            ),
            "batch_results": [],
            "status": "running",
        }
        result["hardware"]["gpu_uuid"] = _gpu_target() or ""
        _write_result(result, output_path)

        baseline_quality = None
        for batch_size in sizes:
            print(f"\n--- batch_size={batch_size} ---", flush=True)
            entry = {"batch_size": batch_size, "status": "running", "trials": []}
            result["batch_results"].append(entry)
            sampler = None
            try:
                # Cover short, median, and longest shapes. Warming only the first
                # duration-sorted batch would leave long-input allocations inside
                # the timed region and bias models with dynamic padding.
                for start in warmup_starts(len(prepared), batch_size, warmup_batches):
                    warm = list(prepared[start : start + batch_size])
                    warm_hyps = transcribe_batch([c.path for c in warm])
                    if len(warm_hyps) != len(warm):
                        raise RuntimeError(
                            "batch callable returned wrong number of warmup hypotheses"
                        )
                synchronize_device()

                repeat_hypotheses = []
                for repeat in range(repeats):
                    torch.cuda.reset_peak_memory_stats()
                    sampler = NvidiaSmiSampler(telemetry_interval_ms)
                    sampler.start()
                    batch_latencies, batch_counts, hypotheses = [], [], []
                    synchronize_device()
                    trial_start = time.perf_counter()
                    for offset in range(0, len(prepared), batch_size):
                        chunk = prepared[offset : offset + batch_size]
                        synchronize_device()
                        t0 = time.perf_counter()
                        outputs = transcribe_batch([c.path for c in chunk])
                        synchronize_device()
                        batch_latencies.append(time.perf_counter() - t0)
                        if len(outputs) != len(chunk):
                            raise RuntimeError(
                                f"batch callable returned {len(outputs)} outputs for {len(chunk)} inputs"
                            )
                        hypotheses.extend(text_value(v) for v in outputs)
                        batch_counts.append(len(chunk))
                    synchronize_device()
                    processing_seconds = time.perf_counter() - trial_start
                    telemetry = sampler.stop()
                    sampler = None
                    if telemetry.get("telemetry_samples", 0) < 1:
                        raise RuntimeError(
                            "nvidia-smi produced no GPU telemetry samples"
                        )
                    trial = summarize_trial(
                        processing_seconds,
                        batch_latencies,
                        batch_counts,
                        audio_seconds,
                    )
                    trial.update(_torch_peak_memory())
                    trial.update(telemetry)
                    if (trial.get("power_mean_w") or 0) > 0:
                        trial["estimated_gpu_energy_wh"] = round(
                            trial["power_mean_w"] * processing_seconds / 3600, 4
                        )
                        trial["estimated_gpu_wh_per_audio_hour"] = round(
                            trial["power_mean_w"] * processing_seconds / audio_seconds,
                            4,
                        )
                        trial["audio_seconds_per_gpu_joule"] = round(
                            audio_seconds
                            / (trial["power_mean_w"] * processing_seconds),
                            6,
                        )
                    repeat_hypotheses.append(hypotheses)
                    trial["repeat"] = repeat + 1
                    entry["trials"].append(trial)
                    print(
                        f"repeat {repeat + 1}/{repeats}: "
                        f"RTFx={trial['rtfx_audio_s_per_s']:.2f}, "
                        f"GPU mean={trial.get('gpu_util_mean_pct', 'NA')}%, "
                        f"peak={trial.get('device_memory_peak_mib', 'NA')} MiB",
                        flush=True,
                    )

                # Quality work is intentionally outside and after all timing
                # repeats so CPU-side WER does not cool the GPU between trials.
                repeat_qualities = [_quality(prepared, h) for h in repeat_hypotheses]
                for trial, trial_quality in zip(entry["trials"], repeat_qualities):
                    trial["quality"] = trial_quality
                quality = dict(repeat_qualities[0])
                repeat_hashes = {q["hypothesis_sha256"] for q in repeat_qualities}
                quality["repeat_hypotheses_identical"] = len(repeat_hashes) == 1
                entry["quality"] = quality
                if baseline_quality is None:
                    baseline_quality = quality
                wer_delta = (
                    quality["corpus_wer_pct"] - baseline_quality["corpus_wer_pct"]
                )
                quality["wer_delta_pp_vs_batch1"] = round(wer_delta, 4)
                quality["empty_delta_vs_batch1"] = (
                    quality["empty_hypotheses"] - baseline_quality["empty_hypotheses"]
                )
                quality["hypotheses_identical_to_batch1"] = (
                    quality["hypothesis_sha256"]
                    == baseline_quality["hypothesis_sha256"]
                )
                entry["quality_valid"] = (
                    abs(wer_delta) <= max_wer_delta_pp
                    and quality["empty_hypotheses"]
                    <= baseline_quality["empty_hypotheses"]
                    and quality["repeat_hypotheses_identical"]
                )
                if batch_size == 1 and not entry["quality_valid"]:
                    raise RuntimeError(
                        "batch-1 baseline produced non-identical hypotheses across repeats"
                    )
                entry["median"] = {
                    key: _median_metric(entry["trials"], key)
                    for key in (
                        "processing_seconds",
                        "rtf",
                        "rtfx_audio_s_per_s",
                        "utterances_per_s",
                        "batch_latency_p50_s",
                        "batch_latency_p95_s",
                        "completion_latency_p50_s",
                        "completion_latency_p95_s",
                        "gpu_util_mean_pct",
                        "gpu_util_p95_pct",
                        "device_memory_peak_mib",
                        "torch_peak_allocated_mib",
                        "torch_peak_reserved_mib",
                        "power_mean_w",
                        "estimated_gpu_energy_wh",
                        "estimated_gpu_wh_per_audio_hour",
                        "audio_seconds_per_gpu_joule",
                    )
                }
                entry["variability"] = {
                    "rtfx_audio_s_per_s": _variability(
                        entry["trials"], "rtfx_audio_s_per_s"
                    ),
                    "processing_seconds": _variability(
                        entry["trials"], "processing_seconds"
                    ),
                }
                entry["status"] = "ok"
            except Exception as exc:
                if sampler is not None:
                    try:
                        sampler.stop()
                    except Exception:
                        pass
                entry["status"] = "oom" if _is_oom(exc) else "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{entry['status'].upper()}] {entry['error']}", flush=True)
                torch.cuda.empty_cache()
                if batch_size == 1:
                    result["status"] = "failed"
                    _write_result(result, output_path)
                    raise
                if entry["status"] == "failed":
                    result["status"] = "failed"
                    _write_result(result, output_path)
                    raise
            _write_result(result, output_path)
            if entry["status"] in {"oom", "failed"}:
                break

        valid = [
            e
            for e in result["batch_results"]
            if e.get("status") == "ok" and e.get("quality_valid")
        ]
        if not valid:
            result["status"] = "failed"
            result["selection"] = {"reason": "no quality-valid batch configuration"}
        else:
            best = select_best_entry(valid)
            result["status"] = "complete"
            result["selection"] = {
                "rule": "smallest batch within 1% of maximum median RTFx among quality-valid configurations",
                "best_batch_size": best["batch_size"],
                "best_median_rtfx_audio_s_per_s": best["median"]["rtfx_audio_s_per_s"],
                "best_median_utterances_per_s": best["median"]["utterances_per_s"],
                "best_median_gpu_util_mean_pct": best["median"].get(
                    "gpu_util_mean_pct"
                ),
                "best_median_device_memory_peak_mib": best["median"].get(
                    "device_memory_peak_mib"
                ),
                "best_corpus_wer_pct": best["quality"]["corpus_wer_pct"],
            }
        _write_result(result, output_path)
    print(f"\nSaved {output_path}")
    return result
