"""Dependency-light invariants for the offline throughput protocol."""

import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.throughput import (
    parse_batch_sizes,
    select_best_entry,
    summarize_trial,
    warmup_starts,
)
from analysis.compare_throughput import validate


def test_batch_sizes_require_batch_one_and_are_canonical():
    assert parse_batch_sizes("8,1,4,4") == (1, 4, 8)
    try:
        parse_batch_sizes("2,4")
    except ValueError as exc:
        assert "include 1" in str(exc)
    else:
        raise AssertionError("missing batch-1 baseline was accepted")


def test_trial_metrics_use_total_audio_and_wall_time():
    result = summarize_trial(
        processing_seconds=2.0,
        batch_latencies=[0.8, 1.2],
        batch_counts=[2, 2],
        audio_seconds=20.0,
    )
    assert result["n_clips"] == 4
    assert result["n_batches"] == 2
    assert result["rtf"] == 0.1
    assert result["rtfx_audio_s_per_s"] == 10.0
    assert result["utterances_per_s"] == 2.0
    assert result["completion_latency_p50_s"] == 1.0


def test_invalid_batch_sizes_rejected():
    for value in ("", "0,1", "-1,1"):
        try:
            parse_batch_sizes(value)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"invalid batch sizes accepted: {value!r}")


def test_best_batch_uses_smallest_size_within_one_percent_of_peak():
    entries = [
        {"batch_size": 8, "median": {"rtfx_audio_s_per_s": 100.0}},
        {"batch_size": 16, "median": {"rtfx_audio_s_per_s": 100.8}},
        {"batch_size": 32, "median": {"rtfx_audio_s_per_s": 101.0}},
    ]
    assert select_best_entry(entries)["batch_size"] == 8


def test_warmups_cover_longest_duration_bucket():
    assert warmup_starts(512, 128, 3) == [0, 192, 384]
    assert warmup_starts(512, 128, 1) == [384]
    assert warmup_starts(512, 128, 0) == []


def _fake_result(model: str, runtime: str, package: str, version: str) -> dict:
    checkpoints = {
        "tiny": "openai/whisper-tiny",
        "parakeet": "nvidia/parakeet-tdt-0.6b-v2",
    }
    checkpoint = checkpoints[model]
    trial = {
        "rtfx_audio_s_per_s": 10.0,
        "processing_seconds": 1.0,
        "telemetry_samples": 4,
    }
    entry = {
        "batch_size": 1,
        "status": "ok",
        "quality_valid": True,
        "trials": [deepcopy(trial), deepcopy(trial), deepcopy(trial)],
        "median": {"rtfx_audio_s_per_s": 10.0},
    }
    clip_ids = [f"clip-{i:03d}" for i in range(512)]
    software = {
        "torch": "2.5.1",
        "datasets": "4.8.5",
        "pandas": "2.2.3",
        "numpy": "1.26.4",
        "soundfile": "0.13.1",
        "librosa": "0.11.0",
        "jiwer": "4.0.0",
        "num2words": "0.5.14",
        package: version,
    }
    if runtime == "huggingface_transformers_whisper_pipeline":
        software.update({"accelerate": "1.12.0", "safetensors": "0.6.2"})
    return {
        "_path": f"throughput_{model}.json",
        "protocol_version": "offline-throughput-v1",
        "status": "complete",
        "model_key": model,
        "model_display": {
            "tiny": "Whisper Tiny",
            "parakeet": "Parakeet-TDT-0.6B-v2",
        }[model],
        "model_id": checkpoint,
        "runtime": runtime,
        "runtime_config": {"checkpoint": checkpoint},
        "git_commit": "a" * 40,
        "source_sha256": "f" * 64,
        "software": software,
        "hardware": {
            "gpu_name": "NVIDIA A100-SXM4-40GB",
            "gpu_count": 1,
            "gpu_total_mem_mib": 40536.0,
            "nvidia_driver": "570.124.06",
            "torch_cuda": "12.4",
            "cudnn": 90100,
        },
        "workload": {
            "dataset": "tie",
            "dataset_id": "raianand/TIE_shorts",
            "dataset_revision": "28c53e285feae86f4ba25d8aaeca4fd0c709784c",
            "split": "test",
            "n_clips": 512,
            "seed": 42,
            "eligibility": "transcript_clean_reference_nonempty_and_duration_at_most_limit",
            "max_clip_seconds": 30.0,
            "selection_indices": list(range(512)),
            "ordered_clip_ids": clip_ids,
            "subset_fingerprint": "selected",
            "ordered_workload_fingerprint": "fingerprint",
            "audio_content_sha256": "c" * 64,
            "audio_seconds_total": 20.0,
            "duration_min_s": 0.5,
            "duration_p50_s": 3.0,
            "duration_p95_s": 8.0,
            "duration_max_s": 12.0,
            "ordering": "ascending_duration_then_clip_id",
            "audio_format": "mono_pcm_s16le_16000_hz_wav",
        },
        "protocol": {
            "scenario": "offline_saturated_throughput",
            "timed_region": "pre_staged_wav_read_plus_feature_extraction_plus_model_plus_decode",
            "batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
            "warmup_batches_per_size": 3,
            "timed_repeats": 3,
            "telemetry_interval_ms": 250,
            "quality_gate_max_abs_wer_delta_pp_vs_batch1": 0.1,
            "quality_gate_empty_hypotheses": "no_increase_vs_batch1",
            "quality_gate_repeat_outputs": "identical_hypothesis_hashes",
            "throughput_tie_rule": "smallest_batch_within_1pct_of_maximum_median_rtfx",
            "cpu_thread_limits": {
                "OMP_NUM_THREADS": "16",
                "MKL_NUM_THREADS": "16",
                "NUMEXPR_NUM_THREADS": "16",
            },
        },
        "batch_results": [entry],
        "selection": {"best_batch_size": 1},
    }


def test_aggregator_rejects_mixed_workloads_and_commits():
    whisper = _fake_result(
        "tiny", "huggingface_transformers_whisper_pipeline", "transformers", "4.57.6"
    )
    parakeet = _fake_result(
        "parakeet", "nvidia_nemo_native_transcribe", "nemo_toolkit", "2.3.0"
    )
    validate([whisper, parakeet], "tie", require_complete=False)

    bad_audio = deepcopy(parakeet)
    bad_audio["workload"]["audio_content_sha256"] = "d" * 64
    try:
        validate([whisper, bad_audio], "tie", require_complete=False)
    except ValueError as exc:
        assert "audio_content_sha256 differs" in str(exc)
    else:
        raise AssertionError("mixed audio workloads were accepted")

    bad_commit = deepcopy(parakeet)
    bad_commit["git_commit"] = "e" * 40
    try:
        validate([whisper, bad_commit], "tie", require_complete=False)
    except ValueError as exc:
        assert "git commits" in str(exc)
    else:
        raise AssertionError("mixed commits were accepted")

    bad_source = deepcopy(parakeet)
    bad_source["source_sha256"] = "0" * 64
    try:
        validate([whisper, bad_source], "tie", require_complete=False)
    except ValueError as exc:
        assert "source checksums" in str(exc)
    else:
        raise AssertionError("mixed source checksums were accepted")

    bad_cuda = deepcopy(parakeet)
    bad_cuda["hardware"]["torch_cuda"] = "11.8"
    try:
        validate([whisper, bad_cuda], "tie", require_complete=False)
    except ValueError as exc:
        assert "PyTorch CUDA runtime differs" in str(exc)
    else:
        raise AssertionError("mixed CUDA runtimes were accepted")


def test_aggregator_accepts_cuda_local_torch_version_suffix():
    whisper = _fake_result(
        "tiny", "huggingface_transformers_whisper_pipeline", "transformers", "4.57.6"
    )
    parakeet = _fake_result(
        "parakeet", "nvidia_nemo_native_transcribe", "nemo_toolkit", "2.3.0"
    )
    whisper["software"]["torch"] = "2.5.1+cu124"
    validate([whisper, parakeet], "tie", require_complete=False)


def test_aggregator_rejects_wrong_model_runtime_or_checkpoint():
    whisper = _fake_result(
        "tiny", "huggingface_transformers_whisper_pipeline", "transformers", "4.57.6"
    )

    wrong_runtime = deepcopy(whisper)
    wrong_runtime["runtime"] = "nvidia_nemo_native_transcribe"
    wrong_runtime["software"]["nemo_toolkit"] = "2.3.0"
    try:
        validate([wrong_runtime], "tie", require_complete=False)
    except ValueError as exc:
        assert "model tiny used runtime" in str(exc)
    else:
        raise AssertionError("wrong model/runtime mapping was accepted")

    wrong_checkpoint = deepcopy(whisper)
    wrong_checkpoint["model_id"] = "openai/whisper-base"
    wrong_checkpoint["runtime_config"]["checkpoint"] = "openai/whisper-base"
    try:
        validate([wrong_checkpoint], "tie", require_complete=False)
    except ValueError as exc:
        assert "model tiny used checkpoint" in str(exc)
    else:
        raise AssertionError("wrong model/checkpoint mapping was accepted")
