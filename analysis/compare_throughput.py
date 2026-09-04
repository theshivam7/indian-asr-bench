"""Validate and aggregate quality-gated offline throughput results.

Usage:
    python analysis/compare_throughput.py --dataset tie --require-complete
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.io_helpers import analysis_dir, build_md_table, throughput_dir
from utils.registry import CHART_MODELS, MODEL_BY_KEY, get_dataset
from utils.throughput import PROTOCOL_VERSION, select_best_entry

EXPECTED_RUNTIME = {
    "huggingface_transformers_whisper_pipeline": {
        "transformers": "4.57.6",
        "accelerate": "1.12.0",
        "safetensors": "0.6.2",
    },
    "nvidia_nemo_native_transcribe": {"nemo_toolkit": "2.3.0"},
    "qwen_asr_transformers_backend": {
        "qwen-asr": "0.0.6",
        "transformers": "4.57.6",
    },
}

EXPECTED_MODEL = {
    "tiny": ("huggingface_transformers_whisper_pipeline", "openai/whisper-tiny"),
    "base": ("huggingface_transformers_whisper_pipeline", "openai/whisper-base"),
    "small": ("huggingface_transformers_whisper_pipeline", "openai/whisper-small"),
    "medium": ("huggingface_transformers_whisper_pipeline", "openai/whisper-medium"),
    "large": ("huggingface_transformers_whisper_pipeline", "openai/whisper-large-v3"),
    "large_v3_turbo": (
        "huggingface_transformers_whisper_pipeline",
        "openai/whisper-large-v3-turbo",
    ),
    "parakeet": ("nvidia_nemo_native_transcribe", "nvidia/parakeet-tdt-0.6b-v2"),
    "parakeet_ctc": ("nvidia_nemo_native_transcribe", "nvidia/parakeet-ctc-1.1b"),
    "qwen3": ("qwen_asr_transformers_backend", "Qwen/Qwen3-ASR-1.7B"),
}

EXPECTED_COMMON_SOFTWARE = {
    "datasets": "4.8.5",
    "pandas": "2.2.3",
    "numpy": "1.26.4",
    "soundfile": "0.13.1",
    "librosa": "0.11.0",
    "jiwer": "4.0.0",
    "num2words": "0.5.14",
}

EXPECTED_WORKLOAD = {
    "n_clips": 512,
    "seed": 42,
    "eligibility": "transcript_clean_reference_nonempty_and_duration_at_most_limit",
    "max_clip_seconds": 30.0,
    "ordering": "ascending_duration_then_clip_id",
    "audio_format": "mono_pcm_s16le_16000_hz_wav",
}

EXPECTED_PROTOCOL = {
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
}


def _runtime_package(r: dict) -> str:
    if r["runtime"] == "huggingface_transformers_whisper_pipeline":
        return f"transformers={r['software'].get('transformers')}"
    if r["runtime"] == "nvidia_nemo_native_transcribe":
        return f"nemo_toolkit={r['software'].get('nemo_toolkit')}"
    return (
        f"qwen-asr={r['software'].get('qwen-asr')};"
        f"transformers={r['software'].get('transformers')}"
    )


def load_results(dataset: str) -> list[dict]:
    results = []
    for path in sorted(
        glob.glob(os.path.join(throughput_dir(dataset), "throughput_*.json"))
    ):
        with open(path) as f:
            data = json.load(f)
        data["_path"] = path
        results.append(data)
    return results


def validate(results: list[dict], dataset: str, require_complete: bool) -> None:
    if not results:
        raise FileNotFoundError(f"no throughput JSONs found for {dataset}")
    dataset_spec = get_dataset(dataset)
    errors = []
    for r in results:
        if r.get("protocol_version") != PROTOCOL_VERSION:
            errors.append(f"{r['_path']}: wrong protocol {r.get('protocol_version')}")
        if r.get("status") != "complete":
            errors.append(f"{r['_path']}: status={r.get('status')}")
        if r.get("workload", {}).get("dataset") != dataset:
            errors.append(f"{r['_path']}: dataset mismatch")
        expected_dataset_identity = {
            "dataset_id": dataset_spec.hf_id,
            "dataset_revision": dataset_spec.hf_revision,
            "split": dataset_spec.splits["eval"],
        }
        for field, expected in expected_dataset_identity.items():
            actual = r.get("workload", {}).get(field)
            if actual != expected:
                errors.append(
                    f"{r['_path']}: workload {field}={actual!r}, expected {expected!r}"
                )
        runtime = r.get("runtime")
        if runtime not in EXPECTED_RUNTIME:
            errors.append(f"{r['_path']}: unexpected runtime {runtime!r}")
        else:
            for package, expected in EXPECTED_RUNTIME[runtime].items():
                actual = r.get("software", {}).get(package)
                if actual != expected:
                    errors.append(
                        f"{r['_path']}: {package}={actual!r}, expected {expected!r}"
                    )
        model_key = r.get("model_key")
        if model_key not in EXPECTED_MODEL:
            errors.append(f"{r['_path']}: unexpected model key {model_key!r}")
        else:
            expected_runtime, expected_checkpoint = EXPECTED_MODEL[model_key]
            if runtime != expected_runtime:
                errors.append(
                    f"{r['_path']}: model {model_key} used runtime {runtime!r}, "
                    f"expected {expected_runtime!r}"
                )
            if r.get("model_id") != expected_checkpoint:
                errors.append(
                    f"{r['_path']}: model {model_key} used checkpoint "
                    f"{r.get('model_id')!r}, expected {expected_checkpoint!r}"
                )
            if r.get("runtime_config", {}).get("checkpoint") != expected_checkpoint:
                errors.append(
                    f"{r['_path']}: runtime checkpoint does not match "
                    f"{expected_checkpoint!r}"
                )
        expected_spec = MODEL_BY_KEY.get(model_key)
        if expected_spec and r.get("model_display") != expected_spec.display:
            errors.append(f"{r['_path']}: model display name does not match registry")
        for package, expected in EXPECTED_COMMON_SOFTWARE.items():
            actual = r.get("software", {}).get(package)
            if actual != expected:
                errors.append(
                    f"{r['_path']}: {package}={actual!r}, expected {expected!r}"
                )
        torch_version = r.get("software", {}).get("torch")
        if not isinstance(torch_version, str) or torch_version.split("+")[0] != "2.5.1":
            errors.append(
                f"{r['_path']}: torch={torch_version!r}, expected base version '2.5.1'"
            )
        for field, expected in EXPECTED_WORKLOAD.items():
            actual = r.get("workload", {}).get(field)
            if actual != expected:
                errors.append(
                    f"{r['_path']}: workload {field}={actual!r}, expected {expected!r}"
                )
        for field, expected in EXPECTED_PROTOCOL.items():
            actual = r.get("protocol", {}).get(field)
            if actual != expected:
                errors.append(
                    f"{r['_path']}: protocol {field}={actual!r}, expected {expected!r}"
                )

    def require_same(label: str, values: list) -> None:
        rendered = {json.dumps(v, sort_keys=True) for v in values}
        if len(rendered) != 1:
            errors.append(f"{label} differs: {sorted(rendered)}")

    workload_fields = (
        "dataset_id",
        "dataset_revision",
        "split",
        "n_clips",
        "seed",
        "eligibility",
        "max_clip_seconds",
        "selection_indices",
        "ordered_clip_ids",
        "subset_fingerprint",
        "ordered_workload_fingerprint",
        "audio_content_sha256",
        "audio_seconds_total",
        "duration_min_s",
        "duration_p50_s",
        "duration_p95_s",
        "duration_max_s",
        "ordering",
        "audio_format",
    )
    for field in workload_fields:
        require_same(field, [r.get("workload", {}).get(field) for r in results])
    protocol_fields = (
        "scenario",
        "timed_region",
        "batch_sizes",
        "warmup_batches_per_size",
        "timed_repeats",
        "telemetry_interval_ms",
        "quality_gate_max_abs_wer_delta_pp_vs_batch1",
        "quality_gate_empty_hypotheses",
        "quality_gate_repeat_outputs",
        "throughput_tie_rule",
        "cpu_thread_limits",
    )
    for field in protocol_fields:
        require_same(field, [r.get("protocol", {}).get(field) for r in results])

    gpu_names = {r.get("hardware", {}).get("gpu_name") for r in results}
    gpu_mems = {r.get("hardware", {}).get("gpu_total_mem_mib") for r in results}
    drivers = {r.get("hardware", {}).get("nvidia_driver") for r in results}
    if len(gpu_names) != 1 or len(gpu_mems) != 1:
        errors.append(f"GPU hardware differs: names={gpu_names}, memory={gpu_mems}")
    if not all(gpu_names) or not all(gpu_mems):
        errors.append("GPU name or capacity is missing")
    if any("A100" not in str(name) for name in gpu_names):
        errors.append(f"protocol requires A100 GPUs, found {gpu_names}")
    if any(not 39_000 <= float(mem or 0) <= 41_500 for mem in gpu_mems):
        errors.append(f"protocol requires 40 GB GPUs, found capacities {gpu_mems}")
    if any(r.get("hardware", {}).get("gpu_count") != 1 for r in results):
        errors.append("protocol requires exactly one CUDA-visible GPU per process")
    if len(drivers) != 1:
        errors.append(f"NVIDIA driver versions differ: {drivers}")
    require_same(
        "PyTorch CUDA runtime",
        [r.get("hardware", {}).get("torch_cuda") for r in results],
    )
    require_same("cuDNN runtime", [r.get("hardware", {}).get("cudnn") for r in results])
    if any(not r.get("hardware", {}).get("torch_cuda") for r in results):
        errors.append("PyTorch CUDA runtime is missing")
    if any(not r.get("hardware", {}).get("cudnn") for r in results):
        errors.append("cuDNN runtime is missing")
    require_same(
        "torch base version",
        [str(r.get("software", {}).get("torch")).split("+")[0] for r in results],
    )
    commits = {r.get("git_commit") for r in results}
    if len(commits) != 1 or not next(iter(commits), ""):
        errors.append(f"git commits are missing or differ: {commits}")
    source_digests = {r.get("source_sha256") for r in results}
    if len(source_digests) != 1 or not next(iter(source_digests), ""):
        errors.append(f"source checksums are missing or differ: {source_digests}")
    if require_complete:
        present = {r.get("model_key") for r in results}
        missing = sorted(set(CHART_MODELS) - present)
        if missing:
            errors.append(f"missing headline models: {missing}")
    keys = [r.get("model_key") for r in results]
    if len(keys) != len(set(keys)):
        errors.append("duplicate model result files")
    for r in results:
        batches = {e.get("batch_size"): e for e in r.get("batch_results", [])}
        b1 = batches.get(1)
        if not b1 or b1.get("status") != "ok" or not b1.get("quality_valid"):
            errors.append(f"{r['_path']}: no valid batch-1 baseline")
            continue
        best_size = r.get("selection", {}).get("best_batch_size")
        best = batches.get(best_size)
        if not best or best.get("status") != "ok" or not best.get("quality_valid"):
            errors.append(f"{r['_path']}: selected batch is absent or quality-invalid")
        else:
            valid_entries = [
                e
                for e in batches.values()
                if e.get("status") == "ok" and e.get("quality_valid")
            ]
            expected_best = select_best_entry(valid_entries)["batch_size"]
            if best_size != expected_best:
                errors.append(
                    f"{r['_path']}: selected batch {best_size} does not follow tie rule "
                    f"(expected {expected_best})"
                )
        expected_repeats = r.get("protocol", {}).get("timed_repeats")
        for entry in batches.values():
            if entry.get("status") != "ok":
                continue
            if len(entry.get("trials", [])) != expected_repeats:
                errors.append(
                    f"{r['_path']}: batch {entry.get('batch_size')} has the wrong repeat count"
                )
            if any(t.get("telemetry_samples", 0) < 1 for t in entry.get("trials", [])):
                errors.append(
                    f"{r['_path']}: batch {entry.get('batch_size')} is missing GPU telemetry"
                )
    if errors:
        raise ValueError(
            "throughput comparison is not valid:\n  - " + "\n  - ".join(errors)
        )


def aggregate(results: list[dict], dataset: str) -> pd.DataFrame:
    rows = []
    for r in results:
        selection = r["selection"]
        best_size = selection["best_batch_size"]
        best = next(e for e in r["batch_results"] if e["batch_size"] == best_size)
        b1 = next(e for e in r["batch_results"] if e["batch_size"] == 1)
        rows.append(
            {
                "dataset": dataset,
                "model": r["model_key"],
                "model_display": r["model_display"],
                "runtime": r["runtime"],
                "best_batch_size": best_size,
                "best_rtfx_audio_s_per_s": best["median"]["rtfx_audio_s_per_s"],
                "best_rtfx_min": best["variability"]["rtfx_audio_s_per_s"]["min"],
                "best_rtfx_max": best["variability"]["rtfx_audio_s_per_s"]["max"],
                "best_rtfx_cv_pct": best["variability"]["rtfx_audio_s_per_s"]["cv_pct"],
                "batch1_rtfx_audio_s_per_s": b1["median"]["rtfx_audio_s_per_s"],
                "batching_speedup_x": round(
                    best["median"]["rtfx_audio_s_per_s"]
                    / b1["median"]["rtfx_audio_s_per_s"],
                    3,
                ),
                "utterances_per_s": best["median"]["utterances_per_s"],
                "completion_latency_p50_s": best["median"]["completion_latency_p50_s"],
                "completion_latency_p95_s": best["median"]["completion_latency_p95_s"],
                "gpu_util_mean_pct": best["median"].get("gpu_util_mean_pct"),
                "gpu_util_p95_pct": best["median"].get("gpu_util_p95_pct"),
                "device_memory_peak_mib": best["median"].get("device_memory_peak_mib"),
                "gpu_memory_capacity_mib": r["hardware"].get("gpu_total_mem_mib"),
                "power_mean_w": best["median"].get("power_mean_w"),
                "estimated_gpu_energy_wh": best["median"].get(
                    "estimated_gpu_energy_wh"
                ),
                "estimated_gpu_wh_per_audio_hour": best["median"].get(
                    "estimated_gpu_wh_per_audio_hour"
                ),
                "audio_seconds_per_gpu_joule": best["median"].get(
                    "audio_seconds_per_gpu_joule"
                ),
                "batch1_wer_pct": b1["quality"]["corpus_wer_pct"],
                "best_wer_pct": best["quality"]["corpus_wer_pct"],
                "wer_delta_pp_vs_batch1": best["quality"]["wer_delta_pp_vs_batch1"],
                "empty_hypotheses": best["quality"]["empty_hypotheses"],
                "quality_valid": best["quality_valid"],
                "model_load_seconds": r.get("model_load_seconds"),
                "gpu_name": r["hardware"].get("gpu_name"),
                "driver": r["hardware"].get("nvidia_driver"),
                "torch": r["software"].get("torch"),
                "runtime_package": _runtime_package(r),
                "workload_fingerprint": r["workload"]["ordered_workload_fingerprint"],
                "git_commit": r.get("git_commit"),
                "source_sha256": r.get("source_sha256"),
            }
        )
    order = {m: MODEL_BY_KEY[m].order for m in CHART_MODELS}
    return (
        pd.DataFrame(rows)
        .sort_values("model", key=lambda s: s.map(order), kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("tie", "svarah", "aesrc"))
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    results = load_results(args.dataset)
    validate(results, args.dataset, args.require_complete)
    df = aggregate(results, args.dataset)
    out_csv = os.path.join(analysis_dir(args.dataset), f"throughput_{args.dataset}.csv")
    out_md = os.path.join(analysis_dir(args.dataset), f"throughput_{args.dataset}.md")
    df.to_csv(out_csv, index=False)
    display = df[
        [
            "model_display",
            "best_batch_size",
            "best_rtfx_audio_s_per_s",
            "best_rtfx_min",
            "best_rtfx_max",
            "batching_speedup_x",
            "gpu_util_mean_pct",
            "device_memory_peak_mib",
            "estimated_gpu_wh_per_audio_hour",
            "completion_latency_p95_s",
            "best_wer_pct",
            "wer_delta_pp_vs_batch1",
        ]
    ]
    with open(out_md, "w") as f:
        f.write(f"# Offline throughput: {args.dataset}\n\n")
        f.write(
            "Best quality-valid batch size on the common duration-sorted workload. "
        )
        f.write(
            "RTFx is audio seconds processed per wall-clock second; higher is better.\n\n"
        )
        f.write(build_md_table(display))
        f.write("\n")
    print(display.to_string(index=False))
    print(f"\nSaved {out_csv}\nSaved {out_md}")


if __name__ == "__main__":
    main()
