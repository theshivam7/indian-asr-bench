"""
Stage 3 (error analysis): codified artifact-vs-error taxonomy, full-corpus edition.

The paper's central diagnostic. Per clip (primary/gold mode), Stage 2 already
emits reference-word recall and hypothesis/reference length ratio. Averaging
those across architecturally disjoint models gives a per-clip CONSENSUS view,
classified as:

  clip_over_run    recall >= 0.80 and ratio >= 1.50 -> models transcribed the
                                                       reference PLUS real speech
                                                       the reference doesn't cover
  content_mismatch recall <  0.40                   -> audio does not match the
                                                       reference
  unflagged        otherwise                        -> no artifact signature

Four analyses, all reproducible from committed Stage-2 CSVs (no GPU):

1. FULL-CORPUS taxonomy — shares over ALL clips (with Wilson 95% CIs), not just
   the worst-K tail. This is what supports corpus-level claims.
2. ARTIFACT-ADJUSTED WER — each model's corpus WER on all common clips vs. on
   the unflagged subset. Quantifies how many WER points the benchmark's own
   artifacts add to every model's score (the found-vs-curated headline).
3. INTER-HYPOTHESIS AGREEMENT — normalized word edit distance BETWEEN model
   hypotheses vs. hypothesis-to-reference. Models that agree with each other but
   not with the reference (especially when CTC/transducer "cannot-hallucinate"
   witnesses agree with encoder-decoder models) prove the reference is at fault,
   independent of any one architecture. Also reports per-arch-class agreement
   with the reference on flagged clips (a train-set contamination probe: an
   enc_dec model agreeing with a *flawed* reference more than the CTC witness
   does would suggest caption memorization).
4. THRESHOLD SENSITIVITY — artifact share across a grid of classifier
   thresholds, showing conclusions do not hinge on the default cut-offs.

The legacy worst-K tail analysis is kept (backwards-comparable with the original
hand analysis).

Usage:
    python analysis/error_analysis.py                 # tie, primary mode
    python analysis/error_analysis.py --dataset svarah
"""

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
import jiwer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.registry import PRIMARY_MODE, MODEL_BY_KEY, models_for_dataset, get_dataset
from utils.io_helpers import stage2_dir, analysis_dir, build_md_table

TOP_K = 20
RECALL_OVERRUN = 0.80
RATIO_OVERRUN = 1.50
RECALL_MISMATCH = 0.40

SENS_RECALL_OVERRUN = (0.70, 0.75, 0.80, 0.85, 0.90)
SENS_RATIO_OVERRUN = (1.30, 1.40, 1.50, 1.60, 1.70)
SENS_RECALL_MISMATCH = (0.30, 0.35, 0.40, 0.45, 0.50)


def classify(recall: float, ratio: float,
             r_over: float = RECALL_OVERRUN, ratio_over: float = RATIO_OVERRUN,
             r_mis: float = RECALL_MISMATCH) -> str:
    if recall >= r_over and ratio >= ratio_over:
        return "clip_over_run"
    if recall < r_mis:
        return "content_mismatch"
    return "unflagged"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion (as percentages)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round((centre - half) * 100, 1), round((centre + half) * 100, 1)


def _word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(word_errors, ref_words) for one clip, matching corpus-WER accounting."""
    ref = ref if isinstance(ref, str) else ""
    hyp = hyp if isinstance(hyp, str) else ""
    n_ref = len(ref.split())
    if n_ref == 0:
        return 0, 0
    if not hyp.strip():
        return n_ref, n_ref
    out = jiwer.process_words([ref], [hyp])
    return out.substitutions + out.deletions + out.insertions, n_ref


def _edit_distance_words(a: str, b: str) -> int:
    """Symmetric word-level Levenshtein distance (S+D+I count)."""
    a = a if isinstance(a, str) else ""
    b = b if isinstance(b, str) else ""
    if not a.strip() and not b.strip():
        return 0
    if not a.strip():
        return len(b.split())
    if not b.strip():
        return len(a.split())
    out = jiwer.process_words([a], [b])
    return out.substitutions + out.deletions + out.insertions


def _load_full(dataset: str, model: str, mode: str) -> pd.DataFrame | None:
    path = os.path.join(stage2_dir(dataset), mode, f"wer_{model}_{mode}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    need = {"ID", "wer", "ref_recall", "length_ratio", "reference", "hypothesis"}
    if not need.issubset(df.columns):
        return None
    df = df.copy()
    df["ID"] = df["ID"].astype(str)
    df["model"] = model
    df["arch"] = MODEL_BY_KEY[model].arch_class
    return df[["ID", "model", "arch", "wer", "ref_recall", "length_ratio",
               "reference", "hypothesis"]]


# ----------------------------------------------------------------------------
# 1. Legacy worst-K tail analysis (kept for continuity with the hand analysis)
# ----------------------------------------------------------------------------

def analyze_tail(pool: pd.DataFrame, n_models: int):
    tail = (pool.sort_values("wer", ascending=False)
                .groupby("model", sort=False).head(TOP_K).copy())
    per_clip = tail.groupby("ID").agg(
        n_models=("model", "nunique"), n_arch=("arch", "nunique"),
        recall_mean=("ref_recall", "mean"), recall_std=("ref_recall", "std"),
        ratio_mean=("length_ratio", "mean"), ratio_std=("length_ratio", "std"),
        wer_mean=("wer", "mean"),
    ).reset_index()
    per_clip["category"] = per_clip.apply(
        lambda r: classify(r["recall_mean"], r["ratio_mean"]), axis=1)
    per_clip.loc[per_clip["category"] == "unflagged", "category"] = "genuine_error"
    per_clip = per_clip.sort_values(["n_models", "wer_mean"], ascending=False)

    n_distinct = len(per_clip)
    tax = (per_clip.groupby("category")
           .agg(n_clips=("ID", "count"), mean_recall=("recall_mean", "mean"),
                mean_ratio=("ratio_mean", "mean"), mean_wer=("wer_mean", "mean"))
           .reset_index())
    tax["share_pct"] = (tax["n_clips"] / n_distinct * 100).round(1)
    tax = tax.round({"mean_recall": 2, "mean_ratio": 2, "mean_wer": 3})
    n_art = int(tax.loc[tax["category"].isin(["clip_over_run", "content_mismatch"]), "n_clips"].sum())
    lo, hi = wilson_ci(n_art, n_distinct)
    shared = per_clip[per_clip["n_arch"] >= 3].copy()
    return {"per_clip": per_clip, "taxonomy": tax, "n_distinct": n_distinct,
            "artifact_share": round(n_art / n_distinct * 100, 1) if n_distinct else 0.0,
            "artifact_ci": (lo, hi), "shared": shared, "n_rows": len(tail)}


# ----------------------------------------------------------------------------
# 2. Full-corpus consensus classification + artifact-adjusted WER
# ----------------------------------------------------------------------------

def consensus_table(pool: pd.DataFrame) -> pd.DataFrame:
    cons = pool.groupby("ID").agg(
        n_models=("model", "nunique"), n_arch=("arch", "nunique"),
        recall_mean=("ref_recall", "mean"), ratio_mean=("length_ratio", "mean"),
        wer_mean=("wer", "mean"), wer_min=("wer", "min"),
    ).reset_index()
    cons["category"] = cons.apply(
        lambda r: classify(r["recall_mean"], r["ratio_mean"]), axis=1)
    return cons


def full_corpus_taxonomy(cons: pd.DataFrame) -> pd.DataFrame:
    n = len(cons)
    rows = []
    for cat in ("clip_over_run", "content_mismatch", "unflagged"):
        sub = cons[cons["category"] == cat]
        lo, hi = wilson_ci(len(sub), n)
        rows.append({"category": cat, "n_clips": len(sub),
                     "share_pct": round(len(sub) / n * 100, 1),
                     "share_ci_lo": lo, "share_ci_hi": hi,
                     "mean_recall": round(sub["recall_mean"].mean(), 2) if len(sub) else float("nan"),
                     "mean_ratio": round(sub["ratio_mean"].mean(), 2) if len(sub) else float("nan"),
                     "mean_wer": round(sub["wer_mean"].mean(), 3) if len(sub) else float("nan")})
    return pd.DataFrame(rows)


def artifact_adjusted_wer(pool: pd.DataFrame, cons: pd.DataFrame) -> pd.DataFrame:
    flagged = set(cons.loc[cons["category"] != "unflagged", "ID"])
    rows = []
    for model, grp in pool.groupby("model", sort=False):
        errs_words = [(e, w) for e, w in
                      (_word_errors(r, h) for r, h in zip(grp["reference"], grp["hypothesis"]))]
        ids = grp["ID"].tolist()
        E = sum(e for e, _ in errs_words)
        W = sum(w for _, w in errs_words)
        E_adj = sum(e for (e, _), cid in zip(errs_words, ids) if cid not in flagged)
        W_adj = sum(w for (_, w), cid in zip(errs_words, ids) if cid not in flagged)
        wer_all = E / W * 100 if W else float("nan")
        wer_adj = E_adj / W_adj * 100 if W_adj else float("nan")
        rows.append({"model": model, "display": MODEL_BY_KEY[model].display,
                     "order": MODEL_BY_KEY[model].order,
                     "wer_all_pct": round(wer_all, 2), "wer_adjusted_pct": round(wer_adj, 2),
                     "artifact_inflation_pp": round(wer_all - wer_adj, 2),
                     "n_clips_all": len(ids), "n_clips_adjusted": len(ids) - sum(1 for c in ids if c in flagged)})
    return pd.DataFrame(rows).sort_values("order").drop(columns="order")


# ----------------------------------------------------------------------------
# 3. Inter-hypothesis agreement (reference-error proof + contamination probe)
# ----------------------------------------------------------------------------

def agreement_analysis(pool: pd.DataFrame, cons: pd.DataFrame) -> pd.DataFrame:
    """Per consensus-category: mean normalized inter-hypothesis distance vs.
    mean hypothesis-to-reference WER, plus per-arch-class agreement with the
    reference. Inter-hyp distance is Levenshtein(S+D+I)/mean(word count) —
    symmetric, comparable to WER in scale."""
    hyp = pool.pivot_table(index="ID", columns="model", values="hypothesis", aggfunc="first")
    wer = pool.pivot_table(index="ID", columns="model", values="wer", aggfunc="first")
    arch = {m: MODEL_BY_KEY[m].arch_class for m in hyp.columns}
    grounded = [m for m in hyp.columns if arch[m] in ("ctc", "transducer")]
    free = [m for m in hyp.columns if arch[m] in ("enc_dec", "llm")]

    inter, cross_arch = {}, {}
    for cid, row in hyp.iterrows():
        hyps = {m: (row[m] if isinstance(row[m], str) else "") for m in hyp.columns}
        dists = []
        for a, b in itertools.combinations(hyp.columns, 2):
            la, lb = len(hyps[a].split()), len(hyps[b].split())
            denom = (la + lb) / 2
            if denom == 0:
                continue
            dists.append(_edit_distance_words(hyps[a], hyps[b]) / denom)
        inter[cid] = float(np.mean(dists)) if dists else float("nan")
        xa = []
        for a in free:
            for b in grounded:
                la, lb = len(hyps[a].split()), len(hyps[b].split())
                denom = (la + lb) / 2
                if denom == 0:
                    continue
                xa.append(_edit_distance_words(hyps[a], hyps[b]) / denom)
        cross_arch[cid] = float(np.mean(xa)) if xa else float("nan")

    cons = cons.set_index("ID")
    rows = []
    for cat in ("clip_over_run", "content_mismatch", "unflagged"):
        ids = cons.index[cons["category"] == cat]
        if len(ids) == 0:
            continue
        row = {"category": cat, "n_clips": len(ids),
               "inter_hyp_dist": round(float(np.nanmean([inter[i] for i in ids])), 3),
               "cross_arch_dist": round(float(np.nanmean([cross_arch[i] for i in ids])), 3),
               "hyp_to_ref_wer": round(float(np.nanmean(wer.loc[ids].to_numpy())), 3)}
        for cls, members in (("grounded", grounded), ("free", free)):
            if members:
                row[f"ref_wer_{cls}"] = round(float(np.nanmean(wer.loc[ids, members].to_numpy())), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. Threshold sensitivity
# ----------------------------------------------------------------------------

def threshold_sensitivity(cons: pd.DataFrame) -> pd.DataFrame:
    n = len(cons)
    rows = []
    for r_over, ratio_over in itertools.product(SENS_RECALL_OVERRUN, SENS_RATIO_OVERRUN):
        cat = cons.apply(lambda r: classify(r["recall_mean"], r["ratio_mean"],
                                            r_over, ratio_over, RECALL_MISMATCH), axis=1)
        rows.append({"vary": "overrun", "recall_overrun": r_over, "ratio_overrun": ratio_over,
                     "recall_mismatch": RECALL_MISMATCH,
                     "artifact_share_pct": round((cat != "unflagged").mean() * 100, 1)})
    for r_mis in SENS_RECALL_MISMATCH:
        cat = cons.apply(lambda r: classify(r["recall_mean"], r["ratio_mean"],
                                            RECALL_OVERRUN, RATIO_OVERRUN, r_mis), axis=1)
        rows.append({"vary": "mismatch", "recall_overrun": RECALL_OVERRUN,
                     "ratio_overrun": RATIO_OVERRUN, "recall_mismatch": r_mis,
                     "artifact_share_pct": round((cat != "unflagged").mean() * 100, 1)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------

def main(dataset: str, mode: str) -> None:
    spec = get_dataset(dataset)
    out = analysis_dir(dataset)

    models = [m for m in models_for_dataset(dataset) if MODEL_BY_KEY[m].chart]
    frames = [t for m in models if (t := _load_full(dataset, m, mode)) is not None]
    if not frames:
        print(f"[error_analysis] {spec.display}: no scored files with recall/ratio columns found.")
        return
    pool = pd.concat(frames, ignore_index=True)
    n_models = pool["model"].nunique()

    # Restrict to the cross-model common clip set so every statistic below is
    # computed on identical clips for all models.
    counts = pool.groupby("ID")["model"].nunique()
    common = set(counts.index[counts == n_models])
    dropped = len(counts) - len(common)
    if dropped:
        print(f"  [note] {dropped} clips absent from >=1 model's table were excluded "
              f"(common set: {len(common)} clips x {n_models} models).")
    pool = pool[pool["ID"].isin(common)].copy()

    cons = consensus_table(pool)
    tail = analyze_tail(pool, n_models)
    tax_full = full_corpus_taxonomy(cons)
    adjusted = artifact_adjusted_wer(pool, cons)
    print("  computing inter-hypothesis agreement (pairwise edit distances) ...", flush=True)
    agree = agreement_analysis(pool, cons)
    sens = threshold_sensitivity(cons)

    cons.to_csv(os.path.join(out, f"error_analysis_full_{mode}.csv"), index=False)
    tail["per_clip"].to_csv(os.path.join(out, f"error_analysis_{mode}.csv"), index=False)
    tail["taxonomy"].to_csv(os.path.join(out, f"error_taxonomy_{mode}.csv"), index=False)
    tax_full.to_csv(os.path.join(out, f"error_taxonomy_full_{mode}.csv"), index=False)
    adjusted.to_csv(os.path.join(out, f"artifact_adjusted_{mode}.csv"), index=False)
    agree.to_csv(os.path.join(out, f"agreement_{mode}.csv"), index=False)
    sens.to_csv(os.path.join(out, f"threshold_sensitivity_{mode}.csv"), index=False)

    n_art_full = int(tax_full.loc[tax_full["category"] != "unflagged", "n_clips"].sum())
    full_share = round(n_art_full / len(cons) * 100, 1)
    lo_f, hi_f = wilson_ci(n_art_full, len(cons))
    n_shared = len(tail["shared"])

    with open(os.path.join(out, f"error_analysis_{mode}.md"), "w") as f:
        f.write(f"# Codified error analysis — {spec.display} — mode `{mode}`\n\n")
        f.write(f"{n_models} models, {len(cons)} common clips. Classifier thresholds: "
                f"clip_over_run = recall>={RECALL_OVERRUN:.2f} & ratio>={RATIO_OVERRUN:.2f}; "
                f"content_mismatch = recall<{RECALL_MISMATCH:.2f}; else unflagged. "
                f"Consensus = per-clip mean of recall/ratio across all models.\n\n")

        f.write("## Full-corpus taxonomy (all clips)\n\n")
        f.write(f"**Artifact share over the full corpus: {full_share}% "
                f"(95% Wilson CI {lo_f}–{hi_f}%)** of clips carry an artifact signature.\n\n")
        f.write(build_md_table(tax_full) + "\n\n")

        f.write("## Artifact-adjusted corpus WER\n\n")
        f.write("Corpus WER on all common clips vs. on the unflagged subset. "
                "`artifact_inflation_pp` is how many WER points the benchmark's own "
                "artifacts add to each model's score.\n\n")
        f.write(build_md_table(adjusted) + "\n\n")

        f.write("## Inter-hypothesis agreement\n\n")
        f.write("`inter_hyp_dist` = mean normalized word edit distance BETWEEN model "
                "hypotheses (all pairs); `cross_arch_dist` = same, restricted to "
                "(enc_dec|llm) x (ctc|transducer) pairs; `hyp_to_ref_wer` = mean WER "
                "against the reference. Hypotheses that agree with each other but not "
                "with the reference localize the fault in the reference — across "
                "architectures that share no decoder or training objective. "
                "`ref_wer_grounded` vs `ref_wer_free` on flagged clips is a "
                "contamination probe (free-decoding models matching a flawed reference "
                "better than acoustically-grounded ones would suggest caption "
                "memorization).\n\n")
        f.write(build_md_table(agree) + "\n\n")

        f.write(f"## Worst-{TOP_K} tail (continuity with the original hand analysis)\n\n")
        f.write(f"Top-{TOP_K} highest-WER clips per model ({tail['n_rows']} rows -> "
                f"{tail['n_distinct']} distinct). **Tail artifact share: "
                f"{tail['artifact_share']}%** (95% Wilson CI "
                f"{tail['artifact_ci'][0]}–{tail['artifact_ci'][1]}%).\n\n")
        f.write(build_md_table(tail["taxonomy"][["category", "n_clips", "share_pct",
                                                 "mean_recall", "mean_ratio", "mean_wer"]]) + "\n\n")
        shared_recall_std = round(float(tail["shared"]["recall_std"].mean()), 3) if n_shared else float("nan")
        shared_ratio_std = round(float(tail["shared"]["ratio_std"].mean()), 3) if n_shared else float("nan")
        f.write(f"{n_shared} tail clips appear in the worst-{TOP_K} of >=3 distinct "
                f"architectures; across those the mean per-clip spread is recall "
                f"std={shared_recall_std}, length-ratio std={shared_ratio_std}.\n\n")

        f.write("## Threshold sensitivity\n\n")
        f.write("Full-corpus artifact share under alternative classifier thresholds "
                "(see threshold_sensitivity CSV for the full grid):\n\n")
        f.write(build_md_table(sens[sens["vary"] == "mismatch"]) + "\n")

    print(f"[error_analysis] {spec.display}/{mode}: full-corpus artifact share = {full_share}% "
          f"(CI {lo_f}-{hi_f}), tail share = {tail['artifact_share']}% "
          f"({tail['n_distinct']} tail clips)")
    print(tax_full.to_string(index=False))
    print("\nArtifact-adjusted WER:")
    print(adjusted.to_string(index=False))
    print("\nAgreement:")
    print(agree.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tie")
    ap.add_argument("--mode", default=PRIMARY_MODE)
    a = ap.parse_args()
    main(a.dataset, a.mode)
