"""
Stage 3 (error analysis): codified artifact-vs-error taxonomy + cross-architecture agreement.

This replaces the previously HAND-WRITTEN results/.../error_analysis.md with a
reproducible computation — the paper's central methodological contribution.

For the top-K highest-WER clips per model (primary/gold mode) it measures, per clip:
  * reference-word recall  (fraction of reference word types present in the hypothesis)
  * hypothesis/reference length ratio
both already emitted per-clip by Stage 2. Each distinct worst-clip is classified:

  clip_over_run    recall >= 0.80 and ratio >= 1.50   -> model transcribed the reference
                                                          PLUS real speech the clip cut off
  content_mismatch recall <  0.40                      -> audio does not match the reference
  genuine_error    otherwise                           -> real substitution/omission

artifact_share = clip_over_run + content_mismatch  (the "~70% are dataset artifacts" figure).

Cross-architecture agreement: clips that land in MANY models' worst-K are examined
for how tightly recall/ratio agree ACROSS architecturally-disjoint models (enc_dec /
transducer / ctc / llm). Tight agreement among disjoint architectures is the proof
that the fault is in the audio/reference, not any one model.

Run on BOTH datasets to get the found-vs-curated contrast (TIE artifact share should
greatly exceed Svarah's).

Usage:
    python analysis/error_analysis.py                 # tie, primary mode
    python analysis/error_analysis.py --dataset svarah
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.registry import PRIMARY_MODE, MODEL_BY_KEY, MODEL_DISPLAY, models_for_dataset, get_dataset
from utils.io_helpers import stage2_dir, analysis_dir, build_md_table

TOP_K = 20
RECALL_OVERRUN = 0.80
RATIO_OVERRUN = 1.50
RECALL_MISMATCH = 0.40


def classify(recall: float, ratio: float) -> str:
    if recall >= RECALL_OVERRUN and ratio >= RATIO_OVERRUN:
        return "clip_over_run"
    if recall < RECALL_MISMATCH:
        return "content_mismatch"
    return "genuine_error"


def _load_topk(dataset: str, model: str, mode: str) -> pd.DataFrame | None:
    path = os.path.join(stage2_dir(dataset), mode, f"wer_{model}_{mode}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    need = {"ID", "wer", "ref_recall", "length_ratio"}
    if not need.issubset(df.columns):
        return None
    df = df.sort_values("wer", ascending=False).head(TOP_K).copy()
    df["model"] = model
    df["arch"] = MODEL_BY_KEY[model].arch_class
    return df[["ID", "model", "arch", "wer", "ref_recall", "length_ratio"]]


def analyze(dataset: str, mode: str):
    models = [m for m in models_for_dataset(dataset) if MODEL_BY_KEY[m].chart]
    frames = [t for m in models if (t := _load_topk(dataset, m, mode)) is not None]
    if not frames:
        return None
    pool = pd.concat(frames, ignore_index=True)          # up to n_models * TOP_K rows
    n_models = pool["model"].nunique()

    # Distinct worst-clips: average recall/ratio across the models that flag them.
    per_clip = pool.groupby("ID").agg(
        n_models=("model", "nunique"),
        n_arch=("arch", "nunique"),
        recall_mean=("ref_recall", "mean"),
        recall_std=("ref_recall", "std"),
        ratio_mean=("length_ratio", "mean"),
        ratio_std=("length_ratio", "std"),
        wer_mean=("wer", "mean"),
    ).reset_index()
    per_clip["category"] = per_clip.apply(lambda r: classify(r["recall_mean"], r["ratio_mean"]), axis=1)
    per_clip = per_clip.sort_values(["n_models", "wer_mean"], ascending=False)

    # Taxonomy shares over DISTINCT clips
    n_distinct = len(per_clip)
    tax = (per_clip.groupby("category")
           .agg(n_clips=("ID", "count"), mean_recall=("recall_mean", "mean"),
                mean_ratio=("ratio_mean", "mean"), mean_wer=("wer_mean", "mean"))
           .reset_index())
    tax["share_pct"] = (tax["n_clips"] / n_distinct * 100).round(1)
    tax = tax.round({"mean_recall": 2, "mean_ratio": 2, "mean_wer": 3})
    artifact_share = tax.loc[tax["category"].isin(["clip_over_run", "content_mismatch"]), "share_pct"].sum()

    # Row-weighted share (per model x clip; clips flagged by many models counted
    # once per model) — this is what the original hand analysis reported.
    pool = pool.copy()
    pool["category"] = pool.apply(lambda r: classify(r["ref_recall"], r["length_ratio"]), axis=1)
    row_share = (pool["category"].isin(["clip_over_run", "content_mismatch"]).mean() * 100)

    # Cross-architecture agreement: clips flagged by >=3 distinct architectures.
    shared = per_clip[per_clip["n_arch"] >= 3].copy()
    return {
        "n_models": n_models, "n_rows": len(pool), "n_distinct": n_distinct,
        "taxonomy": tax, "artifact_share": round(float(artifact_share), 1),
        "artifact_share_rows": round(float(row_share), 1),
        "per_clip": per_clip, "shared": shared,
    }


def main(dataset: str, mode: str) -> None:
    spec = get_dataset(dataset)
    res = analyze(dataset, mode)
    out = analysis_dir(dataset)
    if res is None:
        print(f"[error_analysis] {spec.display}: no scored top-K files with recall/ratio columns found.")
        return

    res["per_clip"].to_csv(os.path.join(out, f"error_analysis_{mode}.csv"), index=False)
    res["taxonomy"].to_csv(os.path.join(out, f"error_taxonomy_{mode}.csv"), index=False)

    tax_md = res["taxonomy"][["category", "n_clips", "share_pct", "mean_recall", "mean_ratio", "mean_wer"]]
    n_shared = len(res["shared"])
    shared_recall_std = round(float(res["shared"]["recall_std"].mean()), 3) if n_shared else float("nan")
    shared_ratio_std = round(float(res["shared"]["ratio_std"].mean()), 3) if n_shared else float("nan")

    with open(os.path.join(out, f"error_analysis_{mode}.md"), "w") as f:
        f.write(f"# Codified error analysis — {spec.display} — mode `{mode}`\n\n")
        f.write(f"Top-{TOP_K} highest-WER clips per model, {res['n_models']} models "
                f"({res['n_rows']} rows -> {res['n_distinct']} distinct clips).\n\n")
        f.write(f"**Artifact share (clip_over_run + content_mismatch): {res['artifact_share']}%** "
                f"of the worst distinct clips are dataset artifacts, not model errors "
                f"(row-weighted, counting each model-clip separately: {res['artifact_share_rows']}%).\n\n")
        f.write("Classifier thresholds: clip_over_run = recall>=%.2f & ratio>=%.2f; "
                "content_mismatch = recall<%.2f; else genuine_error.\n\n"
                % (RECALL_OVERRUN, RATIO_OVERRUN, RECALL_MISMATCH))
        f.write("## Taxonomy (distinct worst-clips)\n\n" + build_md_table(tax_md) + "\n\n")
        f.write("## Cross-architecture agreement\n\n")
        f.write(f"{n_shared} clips appear in the worst-{TOP_K} of >=3 distinct architectures "
                f"(of {res['n_models']} models spanning enc_dec / transducer / ctc / llm). "
                f"Across those disjoint architectures the mean per-clip spread is "
                f"recall std={shared_recall_std}, length-ratio std={shared_ratio_std} — "
                f"near-identical failure on models that share no architecture is only possible "
                f"if the fault is in the audio/reference.\n")
    print(f"[error_analysis] {spec.display}/{mode}: artifact_share={res['artifact_share']}% "
          f"({res['n_distinct']} distinct clips, {n_shared} cross-arch-shared)")
    print(tax_md.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tie")
    ap.add_argument("--mode", default=PRIMARY_MODE)
    a = ap.parse_args()
    main(a.dataset, a.mode)
