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
    if cpu_runs and len(cpu_runs) != len(reports):
        warnings.append(f"Some runs are CPU-only ({', '.join(cpu_runs)}) and some are not.")

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
        hw = reports[0].get("hardware", {})
        proto = reports[0].get("protocol", {})
        lines += [
            f"Measured on {hw.get('gpu_name') or 'CPU'}, torch {hw.get('torch_version', '?')}, "
            f"CUDA {hw.get('cuda_version', 'n/a')}.",
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
            "measured one clip at a time, so every engine is timed under identical single-stream "
            "conditions. Throughput under batching is higher than reported here.",
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
