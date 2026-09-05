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
    python analysis/compare_seeds.py --dataset aesrc --mode all   # both modes (recommended)
    python analysis/compare_seeds.py --dataset aesrc              # primary mode only
    python analysis/compare_seeds.py --dataset aesrc --mode whisper_norm

Writes, per mode, results/<dataset>/analysis/:
    finetune_seeds_<mode>.csv / .md        mean, SD, min, max per size
    finetune_seeds_<mode>_per_seed.csv     one row per run, the evidence behind the mean
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
from utils.wer_compute import compute_corpus_wer

SIZES = ("tiny", "small", "medium")
DISPLAY = {"tiny": "Whisper Tiny", "small": "Whisper Small", "medium": "Whisper Medium"}

# The two modes the seed study reports: the pre-registered primary one, and the
# community-standard normalizer it is cross-checked against. `--mode all` writes both,
# which is the recommended way to run this; see main().
SEED_MODES = (PRIMARY_MODE, "whisper_norm")


def corpus_wer(path: str) -> float | None:
    """Corpus WER (%) from a scored per-clip table: sum(errors) / sum(ref words).

    Recomputed from the clip rows rather than read from a summary, so this matches
    the corpus-WER definition used everywhere else (a mean of per-clip WERs would be
    a different and wrong quantity).
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or not {"reference", "hypothesis"}.issubset(df.columns):
        return None
    refs = df["reference"].fillna("").astype(str).tolist()
    hyps = df["hypothesis"].fillna("").astype(str).tolist()
    result = compute_corpus_wer(refs, hyps)
    return result["corpus_wer"] * 100 if result["total_ref_words"] else None


def _identity(path: str) -> pd.Series:
    """ID-indexed normalized references used to enforce a fixed evaluation panel."""
    df = pd.read_csv(path, usecols=["ID", "reference"])
    ids = df["ID"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    if (ids == "").any() or ids.duplicated().any():
        raise ValueError(f"{path}: empty or duplicate clip IDs invalidate seed comparison")
    refs = df["reference"].map(lambda value: "" if pd.isna(value) else str(value))
    return pd.Series(refs.to_numpy(), index=ids, name="reference").sort_index()


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


def build_rows(dataset: str, mode: str) -> tuple[list[dict], list[dict]]:
    """Returns (aggregate rows, per-seed rows).

    The per-seed rows are the evidence behind the aggregate: a claim like "every one of
    the 18 runs improved on its own baseline" is not checkable against mean/SD/min/max
    alone, and a reader who wants to recompute the SD needs the individual deltas. They
    cost nothing to emit, so they are written alongside rather than discarded.
    """
    rows: list[dict] = []
    per_seed: list[dict] = []
    for size in SIZES:
        tables = find_seed_tables(dataset, mode, size)
        if not tables:
            continue
        base = baseline_wer(dataset, mode, size)
        baseline_path = os.path.join(
            stage2_dir(dataset), mode, f"wer_{size}_hf_{mode}.csv"
        )
        baseline_identity = _identity(baseline_path) if base is not None else None
        seeds = sorted(tables)
        ft_wers, deltas, used_seeds = [], [], []
        for s in seeds:
            w = corpus_wer(tables[s])
            if w is None:
                continue
            identity = _identity(tables[s])
            if baseline_identity is not None and not identity.equals(baseline_identity):
                missing = len(baseline_identity.index.difference(identity.index))
                extra = len(identity.index.difference(baseline_identity.index))
                shared = identity.index.intersection(baseline_identity.index)
                ref_mismatch = int(identity.loc[shared].ne(baseline_identity.loc[shared]).sum())
                raise ValueError(
                    f"{tables[s]} does not match the {size}_hf evaluation panel "
                    f"(missing IDs={missing}, extra IDs={extra}, reference mismatches={ref_mismatch})"
                )
            used_seeds.append(s)
            ft_wers.append(w)
            d = (w - base) if base is not None else None
            if d is not None:
                deltas.append(d)
            per_seed.append({
                "size": size,
                "display_name": DISPLAY.get(size, size),
                "seed": s,
                "hf_baseline_wer": round(base, 3) if base is not None else None,
                "ft_wer": round(w, 3),
                "delta_pp": round(d, 3) if d is not None else None,
            })

        # Report the seeds that actually contributed, not every file on disk. A seed
        # whose table is empty or malformed is dropped from ft_wers above, and listing
        # it anyway would make n_seeds and the seed list disagree in the published
        # table with nothing to explain the difference.
        skipped = [s for s in seeds if s not in used_seeds]
        if skipped:
            print(f"  [WARN] {size}: seed(s) {skipped} found on disk but unreadable or "
                  f"empty; excluded from the aggregate.")

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
            "seeds": ",".join(str(s) for s in used_seeds),
            "hf_baseline_wer": round(base, 3) if base is not None else None,
            "ft_wer_mean": round(float(np.mean(ft_wers)), 3),
            "ft_wer_sd": round(float(np.std(ft_wers, ddof=1)), 3) if len(ft_wers) > 1 else None,
            "delta_pp_mean": round(float(arr.mean()), 3) if deltas else None,
            # Guarded on `deltas`, like mean/min/max above: without a baseline `arr`
            # holds absolute WERs, and an unguarded SD would publish the spread of
            # those under a column that claims to describe deltas.
            "delta_pp_sd": round(sd, 3) if deltas and np.isfinite(sd) else None,
            "delta_pp_min": round(float(arr.min()), 3) if deltas else None,
            "delta_pp_max": round(float(arr.max()), 3) if deltas else None,
        })
    return rows, per_seed


def to_markdown(rows: list[dict], per_seed: list[dict], dataset: str, mode: str) -> str:
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
    def fmt(value):
        return "" if value is None else f"{value:g}"

    for r in rows:
        lines.append(
            f"| {r['display_name']} | {r['params']} | {r['n_seeds']} | {fmt(r['hf_baseline_wer'])}% | "
            f"{fmt(r['ft_wer_mean'])}% | {fmt(r['delta_pp_mean'])} | {fmt(r['delta_pp_sd'])} | "
            f"{fmt(r['delta_pp_min'])} | {fmt(r['delta_pp_max'])} |"
        )
    if per_seed:
        lines += [
            "",
            "## Per-seed runs",
            "",
            "The individual runs behind the means above. Listed so the aggregate is "
            "checkable: whether every run improved on its baseline, and how the SD was "
            "computed, are both questions the summary table cannot answer on its own.",
            "",
            "| Size | Seed | Baseline WER | FT WER | Δ (pp) |",
            "|---|:---:|:---:|:---:|:---:|",
        ]
        for r in per_seed:
            lines.append(
                f"| {r['display_name']} | {r['seed']} | {fmt(r['hf_baseline_wer'])}% | "
                f"{fmt(r['ft_wer'])}% | {fmt(r['delta_pp'])} |"
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


def run_one(dataset: str, mode: str) -> bool:
    rows, per_seed = build_rows(dataset, mode)
    if not rows:
        print(f"[compare_seeds] no per-seed tables under {stage2_dir(dataset)}/{mode}")
        print("  Expected files named wer_<size>_%s_ft_seed<N>_%s.csv" % (dataset, mode))
        print("  Produce them with: bash finetune/run_seeds.sh --size tiny --dataset %s" % dataset)
        return False

    out_dir = analysis_dir(dataset)
    csv_path = os.path.join(out_dir, f"finetune_seeds_{mode}.csv")
    md_path = os.path.join(out_dir, f"finetune_seeds_{mode}.md")
    seed_path = os.path.join(out_dir, f"finetune_seeds_{mode}_per_seed.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(per_seed).to_csv(seed_path, index=False)
    with open(md_path, "w") as fh:
        fh.write(to_markdown(rows, per_seed, dataset, mode))

    for r in rows:
        sd = r["delta_pp_sd"]
        print(f"  {r['display_name']:16} n={r['n_seeds']}  delta {r['delta_pp_mean']} pp"
              f"  SD {sd if sd is not None else 'n/a (single seed)'}")
    print(f"[compare_seeds] {dataset}/{mode}: {len(rows)} sizes, {len(per_seed)} runs -> {csv_path}")
    print(f"                                       {md_path}")
    print(f"                                       {seed_path}")
    return True


def main(dataset: str, mode: str) -> None:
    # "all" writes every mode in one pass. Running one mode at a time is how the two
    # published tables drifted apart before: re-running the default command silently
    # left the whisper_norm table stale, with nothing in either file to show it.
    modes = list(SEED_MODES) if mode == "all" else [mode]
    for i, m in enumerate(modes):
        if i:
            print()
        run_one(dataset, m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aggregate multi-seed fine-tuning runs.")
    ap.add_argument("--dataset", default="aesrc", help="dataset key (default: aesrc)")
    ap.add_argument("--mode", default=PRIMARY_MODE,
                    help=f"scoring mode, or 'all' for {' + '.join(SEED_MODES)} "
                         f"(default: {PRIMARY_MODE})")
    args = ap.parse_args()
    main(args.dataset, args.mode)
