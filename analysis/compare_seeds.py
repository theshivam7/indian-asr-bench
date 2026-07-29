"""Stage 3: aggregate the multi-seed fine-tuning study into mean +/- SD per size.

The single-seed study (analysis/compare_finetune.py) reports one delta per size with
a within-run bootstrap CI. That CI describes sampling error over test clips; it says
nothing about how much the result would move if training were repeated. This script
covers the second question by reading every per-seed scored table and reporting the
spread of the delta across seeds.

The two uncertainties are reported side by side and never combined. Pooling them would
misrepresent both: the bootstrap CI is a statement about this test set, the across-seed
SD is a statement about the training procedure.

Inputs (written by finetune/run_seeds.sh):
    results/<dataset>/stage2_processed/<mode>/wer_<size>_<dataset>_ft_seed<N>_<mode>.csv
Baseline for each size is the registry HF key (tiny_hf / small_hf / medium_hf), the same
engine-controlled baseline the single-seed study uses.

Usage:
    python analysis/compare_seeds.py --dataset aesrc
    python analysis/compare_seeds.py --dataset aesrc --mode whisper_norm

Writes results/<dataset>/analysis/finetune_seeds_<mode>.csv and .md
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import analysis_dir, stage2_dir
from utils.registry import MODEL_BY_KEY, PRIMARY_MODE

SIZES = ("tiny", "small", "medium")
DISPLAY = {"tiny": "Whisper Tiny", "small": "Whisper Small", "medium": "Whisper Medium"}


def corpus_wer(path: str) -> float | None:
    """Corpus WER (%) from a scored per-clip table: sum(errors) / sum(ref words).

    Recomputed from the clip rows rather than read from a summary, so this matches
    the corpus-WER definition used everywhere else (a mean of per-clip WERs would be
    a different and wrong quantity).
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or "wer" not in df.columns:
        return None
    ref_words = df["reference"].fillna("").astype(str).str.split().str.len()
    errors = df["wer"].astype(float) * ref_words
    total = float(ref_words.sum())
    return float(errors.sum()) / total * 100 if total else None


def find_seed_tables(dataset: str, mode: str, size: str) -> dict[int, str]:
    """Map seed -> scored-table path for one size, discovered from disk."""
    pattern = os.path.join(stage2_dir(dataset), mode,
                           f"wer_{size}_{dataset}_ft_seed*_{mode}.csv")
    found: dict[int, str] = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(rf"_{size}_{dataset}_ft_seed(\d+)_", os.path.basename(path))
        if m:
            found[int(m.group(1))] = path
    return found


def baseline_wer(dataset: str, mode: str, size: str) -> float | None:
    key = f"{size}_hf"
    if key not in MODEL_BY_KEY:
        return None
    return corpus_wer(os.path.join(stage2_dir(dataset), mode, f"wer_{key}_{mode}.csv"))


def build_rows(dataset: str, mode: str) -> list[dict]:
    rows = []
    for size in SIZES:
        tables = find_seed_tables(dataset, mode, size)
        if not tables:
            continue
        base = baseline_wer(dataset, mode, size)
        seeds = sorted(tables)
        ft_wers, deltas = [], []
        for s in seeds:
            w = corpus_wer(tables[s])
            if w is None:
                continue
            ft_wers.append(w)
            if base is not None:
                deltas.append(w - base)

        if not ft_wers:
            continue
        arr = np.asarray(deltas if deltas else ft_wers, dtype=float)
        # ddof=1: these seeds are a sample of the procedure's run-to-run behaviour, not
        # the whole population of possible runs. With n=1 the SD is undefined, not zero.
        sd = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
        rows.append({
            "size": size,
            "display_name": DISPLAY.get(size, size),
            "params": MODEL_BY_KEY[f"{size}_hf"].params if f"{size}_hf" in MODEL_BY_KEY else "",
            "n_seeds": len(ft_wers),
            "seeds": ",".join(str(s) for s in seeds),
            "hf_baseline_wer": round(base, 3) if base is not None else None,
            "ft_wer_mean": round(float(np.mean(ft_wers)), 3),
            "ft_wer_sd": round(float(np.std(ft_wers, ddof=1)), 3) if len(ft_wers) > 1 else None,
            "delta_pp_mean": round(float(arr.mean()), 3) if deltas else None,
            "delta_pp_sd": round(sd, 3) if np.isfinite(sd) else None,
            "delta_pp_min": round(float(arr.min()), 3) if deltas else None,
            "delta_pp_max": round(float(arr.max()), 3) if deltas else None,
        })
    return rows


def to_markdown(rows: list[dict], dataset: str, mode: str) -> str:
    lines = [
        f"# Multi-seed fine-tuning: {dataset} ({mode})",
        "",
        "Each size trained repeatedly with only the seed changed, then scored through the "
        "identical HF pipeline as its own pretrained baseline.",
        "",
        "The spread below is **across-seed** variation of the delta: how much the result moves "
        "when training is repeated. It is a different quantity from the within-run bootstrap CI "
        "in `finetune_capacity_summary.csv`, which describes sampling error over test clips. "
        "Report both, and do not pool them.",
        "",
        "| Size | Params | Seeds | Baseline WER | FT WER (mean) | Δ mean (pp) | Δ SD (pp) | Δ min | Δ max |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in rows:
        fmt = lambda v: "" if v is None else f"{v:g}"
        lines.append(
            f"| {r['display_name']} | {r['params']} | {r['n_seeds']} | {fmt(r['hf_baseline_wer'])}% | "
            f"{fmt(r['ft_wer_mean'])}% | {fmt(r['delta_pp_mean'])} | {fmt(r['delta_pp_sd'])} | "
            f"{fmt(r['delta_pp_min'])} | {fmt(r['delta_pp_max'])} |"
        )
    lines.append("")
    single = [r for r in rows if r["n_seeds"] < 2]
    if single:
        names = ", ".join(r["display_name"] for r in single)
        lines += [
            f"Note: {names} has only one seed, so no standard deviation is reported. "
            "Run more seeds with `bash finetune/run_seeds.sh --size <size>` before drawing "
            "any conclusion about run-to-run stability.",
            "",
        ]
    return "\n".join(lines)


def main(dataset: str, mode: str) -> None:
    rows = build_rows(dataset, mode)
    if not rows:
        print(f"[compare_seeds] no per-seed tables under {stage2_dir(dataset)}/{mode}")
        print("  Expected files named wer_<size>_%s_ft_seed<N>_%s.csv" % (dataset, mode))
        print("  Produce them with: bash finetune/run_seeds.sh --size tiny --dataset %s" % dataset)
        return

    out_dir = analysis_dir(dataset)
    csv_path = os.path.join(out_dir, f"finetune_seeds_{mode}.csv")
    md_path = os.path.join(out_dir, f"finetune_seeds_{mode}.md")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(md_path, "w") as fh:
        fh.write(to_markdown(rows, dataset, mode))

    for r in rows:
        sd = r["delta_pp_sd"]
        print(f"  {r['display_name']:16} n={r['n_seeds']}  delta {r['delta_pp_mean']} pp"
              f"  SD {sd if sd is not None else 'n/a (single seed)'}")
    print(f"[compare_seeds] {dataset}/{mode}: {len(rows)} sizes -> {csv_path}")
    print(f"                                       {md_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aggregate multi-seed fine-tuning runs.")
    ap.add_argument("--dataset", default="aesrc", help="dataset key (default: aesrc)")
    ap.add_argument("--mode", default=PRIMARY_MODE, help=f"scoring mode (default: {PRIMARY_MODE})")
    args = ap.parse_args()
    main(args.dataset, args.mode)
