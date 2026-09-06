"""Stage 3: merge per-model efficiency runs into one comparable table.

Each engine writes its own results/<dataset>/efficiency/efficiency_<model>.json
(see utils/efficiency.py), because the three engines cannot share a process: NeMo,
openai-whisper and the Qwen3 stack live in separate conda envs. This script is the
CPU-only join that turns those per-model files into the cross-model table the paper
reports, and it is the only place that compares them.

Comparability is checked, not assumed. Runs measured on different clip subsets, or
on different GPUs, are not comparable, so mismatches are reported loudly rather than
silently averaged away.

Usage:
    python analysis/compare_efficiency.py --dataset tie

Writes results/<dataset>/analysis/efficiency_<dataset>.csv and .md
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import analysis_dir, efficiency_dir
from utils.registry import MODEL_BY_KEY, MODEL_ORDER

# Column order for the paper table. RTF first: it is the headline efficiency number,
# and the one the accuracy/cost argument in the paper turns on.
REPORT_COLUMNS = [
    ("display", "Model"),
    ("params_registry", "Params"),
    ("arch_class", "Arch"),
    ("rtf", "RTF"),
    ("rtf_p50", "RTF p50"),
    ("throughput_audio_s_per_s", "Audio-s/s"),
    ("latency_p50_s", "Lat. p50 (s)"),
    ("latency_p95_s", "Lat. p95 (s)"),
    ("peak_gpu_allocated_mib", "Peak GPU (MiB)"),
    ("model_load_seconds", "Load (s)"),
]


def load_reports(dataset: str) -> list[dict]:
    """Read every efficiency_<model>.json for this dataset, in registry order."""
    pattern = os.path.join(efficiency_dir(dataset), "efficiency_*.json")
    reports = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            reports.append(json.load(fh))
    order = {k: i for i, k in enumerate(MODEL_ORDER)}
    reports.sort(key=lambda r: order.get(r.get("model_key", ""), len(order)))
    return reports


def check_comparability(reports: list[dict]) -> list[str]:
    """Flag anything that makes two runs not directly comparable.

    A differing subset fingerprint means the models were not measured on the same
    audio; a differing GPU means the numbers came off different hardware. Either
    invalidates a head-to-head reading, so both are surfaced in the report itself
    rather than left for a reader to discover.
    """
    warnings: list[str] = []

    def distinct(getter) -> set:
        return {getter(r) for r in reports if getter(r) is not None}

    fingerprints = distinct(lambda r: r.get("protocol", {}).get("subset_fingerprint"))
    missing_fingerprints = [r.get("model_key", "?") for r in reports
                            if not r.get("protocol", {}).get("subset_fingerprint")]
    if missing_fingerprints:
        warnings.append(
            f"Runs are missing subset fingerprints ({', '.join(missing_fingerprints)}); "
            "identical audio cannot be verified."
        )
    if len(fingerprints) > 1:
        warnings.append(
            f"Runs used {len(fingerprints)} different clip subsets ({', '.join(sorted(fingerprints))}). "
            "Re-run every model with the same --clips and --seed before comparing."
        )

    gpus = distinct(lambda r: r.get("hardware", {}).get("gpu_name"))
    if len(gpus) > 1:
        warnings.append(f"Runs span different GPUs ({', '.join(sorted(gpus))}); timings are not comparable.")

    batch = distinct(lambda r: r.get("protocol", {}).get("batch_size"))
    if len(batch) > 1:
        warnings.append(f"Runs used different batch sizes ({sorted(batch)}); latency is not comparable.")

    cpu_runs = [r["model_key"] for r in reports
                if r.get("hardware", {}).get("gpu_name") in (None, "", "cpu")]
    if cpu_runs:
        if len(cpu_runs) != len(reports):
            warnings.append(f"Some runs are CPU-only ({', '.join(cpu_runs)}) and some are not.")
        else:
            warnings.append("All runs are CPU-only; these are not GPU-efficiency measurements.")

    # The three engines cannot share a conda env, so their torch builds differ by
    # construction: the Whisper env is cu118 and the NeMo/Qwen envs are cu124. This is
    # a disclosed property of the setup, not something to "fix" by rebuilding an env,
    # so it is reported as a caveat to state alongside the table rather than as a
    # blocker. Same GPU and driver, different CUDA runtime.
    cuda_builds = distinct(lambda r: r.get("hardware", {}).get("torch_cuda") or None)
    if len(cuda_builds) > 1:
        by_build: dict[str, list[str]] = {}
        for r in reports:
            b = r.get("hardware", {}).get("torch_cuda") or "unknown"
            by_build.setdefault(b, []).append(r.get("model_key", "?"))
        detail = "; ".join(f"CUDA {b}: {', '.join(sorted(m))}" for b, m in sorted(by_build.items()))
        warnings.append(
            f"Runs span {len(cuda_builds)} CUDA runtime versions ({detail}). The engines cannot "
            "share an environment, so this is expected; disclose it with the table rather than "
            "treating small cross-engine timing gaps as architectural."
        )

    drivers = distinct(lambda r: r.get("hardware", {}).get("nvidia_driver") or None)
    missing_drivers = [r.get("model_key", "?") for r in reports
                       if not r.get("hardware", {}).get("nvidia_driver")]
    if missing_drivers and not cpu_runs:
        warnings.append(
            f"Runs are missing NVIDIA driver provenance ({', '.join(missing_drivers)})."
        )
    if len(drivers) > 1:
        warnings.append(f"Runs span different NVIDIA drivers ({', '.join(sorted(drivers))}).")

    # A matching fingerprint proves the same clip IDs were selected, not that the same
    # audio was measured: RTF divides by audio_seconds_total, which each driver derives
    # for itself. If one engine resolved durations differently, every RTF in the table
    # is against a different denominator and the ranking is an artefact. Compare the
    # totals directly, with a tolerance for float rounding.
    totals = {r.get("model_key", "?"): r.get("metrics", {}).get("audio_seconds_total")
              for r in reports}
    known = {k: v for k, v in totals.items() if v}
    if len(known) > 1:
        lo, hi = min(known.values()), max(known.values())
        if hi - lo > 0.5:
            detail = ", ".join(f"{k}={v:.1f}s" for k, v in sorted(known.items()))
            warnings.append(
                f"Runs report different total audio despite a shared subset ({detail}). "
                "RTF and throughput divide by this, so the models are not comparable "
                "until the discrepancy is explained."
            )

    # Precision and cuDNN state are not in the batch-1 artifacts at all, so they
    # cannot be diffed the way GPU/driver/CUDA are above. They are not constant
    # across engines either: each driver takes its reference implementation's
    # default, which means fp32-resident weights for Whisper and Parakeet and bf16
    # for Qwen3-ASR, so the peak-memory column is not like-for-like. Warn from the
    # absence of the field, so this clears itself once the harness records it.
    if any("precision" not in r.get("protocol", {}) for r in reports):
        warnings.append(
            "Precision is not recorded in these results. Each engine runs at its "
            "reference implementation's default, which is not the same across "
            "engines (fp32-resident weights for the Whisper and Parakeet drivers, "
            "bf16 for Qwen3-ASR). Peak GPU memory is therefore not a like-for-like "
            "comparison; treat it as each system's default footprint, not as an "
            "architecture-controlled measurement."
        )
    if any("cudnn_enabled_during_inference" not in r.get("protocol", {}) for r in reports):
        warnings.append(
            "cuDNN state during inference is not recorded. Results produced before "
            "the cuDNN fix in the Parakeet and Qwen3 drivers were timed with cuDNN "
            "disabled, while the Whisper runs had it enabled; those Parakeet and "
            "Qwen3 latencies are pessimistic. Re-run to remove this caveat."
        )

    return warnings


def build_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        flat = {
            "model_key": r.get("model_key", ""),
            "display": r.get("display") or MODEL_BY_KEY.get(r.get("model_key", ""), None)
                       and MODEL_BY_KEY[r["model_key"]].display or r.get("model_key", ""),
            "params_registry": r.get("params_registry", ""),
            "arch_class": r.get("arch_class", ""),
            "model_load_seconds": r.get("model_load_seconds"),
            "param_count": r.get("param_count"),
            **r.get("metrics", {}),
        }
        rows.append(flat)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, dataset: str, reports: list[dict], warnings: list[str]) -> str:
    present = [(c, h) for c, h in REPORT_COLUMNS if c in df.columns and df[c].notna().any()]
    lines = [f"# Inference efficiency: {dataset}", ""]

    if reports:
        proto = reports[0].get("protocol", {})

        # Collect across every run rather than reading reports[0]: the engines cannot
        # share an environment, so torch and CUDA genuinely differ between them and
        # quoting one engine's value as the table's would be wrong. Key names must match
        # utils.efficiency.hardware_provenance exactly ("torch", "torch_cuda"), or every
        # field silently renders as a placeholder.
        def _values(field: str) -> list[str]:
            seen = {str(r.get("hardware", {}).get(field)) for r in reports
                    if r.get("hardware", {}).get(field)}
            return sorted(seen)

        def _fmt(field: str, default: str) -> str:
            vals = _values(field)
            return ", ".join(vals) if vals else default

        lines += [
            f"Measured on {_fmt('gpu_name', 'CPU')}, driver {_fmt('nvidia_driver', 'n/a')}, "
            f"torch {_fmt('torch', 'n/a')}, CUDA {_fmt('torch_cuda', 'n/a')}.",
            "",
            f"Protocol: {proto.get('n_clips_requested', '?')} clips sampled with seed "
            f"{proto.get('seed', '?')} (fingerprint `{proto.get('subset_fingerprint', '?')}`), "
            f"{proto.get('n_warmup', '?')} untimed warmup clips, batch size "
            f"{proto.get('batch_size', '?')}.",
            "",
            f"RTF convention: {proto.get('rtf_convention', 'processing / audio, lower is faster')}.",
            "",
        ]

    if warnings:
        lines.append("> **Comparability warnings**")
        lines += [f"> - {w}" for w in warnings]
        lines.append("")

    lines.append("| " + " | ".join(h for _, h in present) + " |")
    lines.append("|" + "|".join(["---"] * len(present)) + "|")
    for _, row in df.iterrows():
        cells = []
        for col, _ in present:
            v = row.get(col)
            cells.append("" if pd.isna(v) else (f"{v:g}" if isinstance(v, (int, float)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    batched = [r.get("display") or r.get("model_key", "") for r in reports
               if r.get("batched_throughput_available")]
    if batched:
        subject = batched[0] if len(batched) == 1 else ", ".join(batched)
        verb = "supports" if len(batched) == 1 else "support"
        lines += [
            f"Note: {subject} {verb} batched inference but {'was' if len(batched) == 1 else 'were'} "
            "measured one clip at a time, so every engine sees the same single-stream batch size. "
            "Batch size is the only thing equalized here; see the comparability warnings above for "
            "what is not. Throughput under batching is higher than reported here.",
            "",
        ]
    return "\n".join(lines)


def main(dataset: str) -> None:
    reports = load_reports(dataset)
    if not reports:
        print(f"[efficiency] no efficiency_*.json under {efficiency_dir(dataset)}")
        print("  Run a driver with --efficiency first, e.g.")
        print(f"    python whisper_asr/run_whisper.py --model medium --dataset {dataset} --efficiency")
        return

    df = build_table(reports)
    warnings = check_comparability(reports)

    out_dir = analysis_dir(dataset)
    csv_path = os.path.join(out_dir, f"efficiency_{dataset}.csv")
    md_path = os.path.join(out_dir, f"efficiency_{dataset}.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as fh:
        fh.write(to_markdown(df, dataset, reports, warnings))

    for w in warnings:
        print(f"  [WARN] {w}")
    print(f"[efficiency] {dataset}: {len(reports)} models -> {csv_path}")
    print(f"                                    {md_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge per-model efficiency runs into one table.")
    ap.add_argument("--dataset", default="tie", help="dataset key (tie, svarah, aesrc)")
    main(ap.parse_args().dataset)
