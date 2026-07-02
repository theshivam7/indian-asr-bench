"""
Stage 2: Normalization + WER/CER scoring (dataset-aware, registry-driven).

Reads raw transcripts from  results/<dataset>/stage1_raw_transcripts/
Writes scored results to     results/<dataset>/stage2_processed/

No GPU needed. Re-run any time to change normalization/metrics — this recomputes
everything from the committed raw transcripts (the immutable source of truth), so
inference is never repeated.

Modes and the model list are defined ONCE in utils/registry.py and gated per
dataset (e.g. Svarah has no pre-normalized reference, so its hf_* modes are absent).

Usage:
    python normalize_and_score.py                 # dataset = tie (default)
    python normalize_and_score.py --dataset svarah
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils.registry import (
    MODE_BY_KEY,
    get_dataset,
    get_reference_role,
    models_for_dataset,
    modes_for_dataset,
)
from utils.normalize import normalize_for_mode
from utils.wer_compute import (
    compute_sample_wer,
    compute_sample_cer,
    compute_corpus_wer,
    compute_corpus_cer,
    reference_word_recall,
    length_ratio,
)
from utils.io_helpers import build_md_table, stage1_raw_dir, stage2_dir

# Raw-CSV text columns are replaced by scored columns; everything else is carried
# through generically so the script is dataset-agnostic (no hard-coded metadata).
_TEXT_COLS = {"transcript_raw", "normalised_transcript_raw", "hypothesis_raw"}


def load_raw(dataset: str, model: str) -> pd.DataFrame | None:
    path = os.path.join(stage1_raw_dir(dataset), f"wer_{model}_raw.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {os.path.relpath(path)} not found")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} samples for {model}")
    return df


def _reference_column(role: str) -> str:
    """Map a mode's canonical reference role onto the raw-CSV column."""
    return "transcript_raw" if role == "gold" else "normalised_transcript_raw"


def process(df: pd.DataFrame, model: str, mode: str) -> tuple[list[dict], dict]:
    """Score one model x mode. Carries through all metadata columns generically."""
    ref_col = _reference_column(get_reference_role(mode))
    carry_cols = [c for c in df.columns if c not in _TEXT_COLS]

    rows = []
    for _, row in df.iterrows():
        ref_raw = str(row.get(ref_col) or "").strip()
        hyp_raw = str(row.get("hypothesis_raw") or "").strip()
        if not ref_raw:
            continue

        ref = normalize_for_mode(mode, ref_raw)
        hyp = normalize_for_mode(mode, hyp_raw)
        if not ref:
            continue

        wer = compute_sample_wer(ref, hyp)
        out = {c: row.get(c, "") for c in carry_cols}
        out.update({
            "model": model,
            "mode": mode,
            "reference_source": get_reference_role(mode),
            "reference_raw": ref_raw,
            "reference": ref,
            "hypothesis_raw": hyp_raw,
            "hypothesis": hyp,
            "wer": round(wer, 4),
            "cer": round(compute_sample_cer(ref, hyp), 4),
            "ref_recall": round(reference_word_recall(ref, hyp), 4),
            "length_ratio": round(length_ratio(ref, hyp), 4),
        })
        rows.append(out)

    refs = [r["reference"] for r in rows]
    hyps = [r["hypothesis"] for r in rows]
    wers = [r["wer"] for r in rows]
    stats = compute_corpus_wer(refs, hyps, per_sample_wers=wers)
    stats["corpus_cer"] = compute_corpus_cer(refs, hyps)
    return rows, stats


def save_csv(rows: list[dict], dataset: str, model: str, mode: str) -> None:
    out_dir = os.path.join(stage2_dir(dataset), mode)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"wer_{model}_{mode}.csv"), index=False)


def save_top20(rows: list[dict], dataset: str, model: str, mode: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("wer", ascending=False).head(20)
    df.to_csv(os.path.join(stage2_dir(dataset), f"top_20_high_wer_{model}_{mode}.csv"), index=False)


def main(dataset: str) -> None:
    spec = get_dataset(dataset)
    models = models_for_dataset(dataset)
    modes = modes_for_dataset(dataset)
    s2 = stage2_dir(dataset)

    print("=" * 70)
    print(f"STAGE 2: Normalization + WER/CER  —  dataset: {spec.display} ({dataset})")
    print("=" * 70)
    print(f"Reading from: {stage1_raw_dir(dataset)}")
    print(f"Writing to:   {s2}")
    print(f"Modes:        {modes}\n")

    all_summary = []
    for model in models:
        print(f"\n--- Model: {model} ---")
        df_raw = load_raw(dataset, model)
        if df_raw is None:
            continue
        for mode in modes:
            print(f"  [{mode}] ...", end=" ", flush=True)
            rows, stats = process(df_raw, model, mode)
            save_csv(rows, dataset, model, mode)
            save_top20(rows, dataset, model, mode)
            print(f"WER={stats['corpus_wer']*100:.2f}%  CER={stats['corpus_cer']*100:.2f}%  "
                  f"median={stats.get('median_wer', 0.0)*100:.2f}%  ins_rate={stats['insertion_rate']*100:.2f}%")
            all_summary.append({
                "dataset": dataset,
                "model": model,
                "mode": mode,
                "reference_source": get_reference_role(mode),
                "normalizer": MODE_BY_KEY[mode].normalizer,
                "corpus_wer_pct": round(stats["corpus_wer"] * 100, 2),
                "corpus_cer_pct": round(stats["corpus_cer"] * 100, 2),
                "mean_wer_pct": round(stats.get("mean_wer", 0.0) * 100, 2),
                "median_wer_pct": round(stats.get("median_wer", 0.0) * 100, 2),
                "std_wer_pct": round(stats.get("std_wer", 0.0) * 100, 2),
                "p90_wer_pct": round(stats.get("p90_wer", 0.0) * 100, 2),
                "p95_wer_pct": round(stats.get("p95_wer", 0.0) * 100, 2),
                "insertion_rate_pct": round(stats["insertion_rate"] * 100, 2),
                "num_samples": stats["num_samples"],
                "total_ref_words": stats["total_ref_words"],
                "total_errors": stats["total_errors"],
            })

    print("\n" + "=" * 70 + "\nSUMMARY (corpus WER %)\n" + "=" * 70)
    df_summary = pd.DataFrame(all_summary)
    present_modes = [m for m in modes if m in set(df_summary["mode"])]
    pivot = df_summary.pivot_table(index="model", columns="mode", values="corpus_wer_pct", aggfunc="first")[present_modes]
    print(pivot.to_string())

    df_summary.to_csv(os.path.join(s2, "wer_summary_all_models.csv"), index=False)
    with open(os.path.join(s2, "wer_summary_all_models.md"), "w") as f:
        f.write(f"# WER Summary — {spec.display} — All Models x Modes\n\n")
        f.write("## Corpus WER (%) Matrix\n\n")
        f.write(build_md_table(pivot.reset_index()) + "\n\n")
        f.write("## Modes\n\n| Mode | Reference | Normalizer |\n|---|---|---|\n")
        for m in modes:
            ms = MODE_BY_KEY[m]
            f.write(f"| `{m}` | {ms.reference} | {ms.normalizer} |\n")
    print(f"\nSaved summary to {s2}\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 2 normalization + scoring.")
    ap.add_argument("--dataset", default="tie", help="dataset key (tie, svarah, ...)")
    main(ap.parse_args().dataset)
