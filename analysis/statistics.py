"""
Stage 3 (statistics): bootstrap confidence intervals + paired significance tests.

Computes, per dataset and evaluation mode:

  * a 95% bootstrap CI on each model's CORPUS WER (resampling with replacement
    and recomputing Sum(errors)/Sum(ref_words) — NOT the mean of per-clip WER,
    which is a different, tail-inflated quantity);
  * paired bootstrap significance for every model pair (identical resample
    indices for both models), with Holm–Bonferroni-adjusted p-values.

Resampling unit: **speakers**, when the dataset exposes a speaker id. Clips from
one speaker share accent/microphone/room, so their errors are correlated;
resampling clips i.i.d. understates variance and overstates significance
(TIE: 986 clips from 280 speakers, median 3 clips/speaker). Clip-level CIs are
reported alongside for transparency; datasets without a speaker id (Svarah's HF
config) fall back to clip-level with an explicit note.

Consistency guards: duplicate clip IDs raise; a model whose scored table covers
fewer clips than the common intersection triggers a loud warning (prevents two
different "corpus WER" numbers for the same model reaching the paper).

Reads   results/<dataset>/stage2_processed/<mode>/wer_<model>_<mode>.csv
Writes  results/<dataset>/analysis/statistics_<mode>.{csv,md}
        results/<dataset>/analysis/statistics_pairwise_<mode>.csv

Usage:
    python analysis/statistics.py                        # tie, primary mode
    python analysis/statistics.py --dataset svarah --mode transcript_clean
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import jiwer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.registry import PRIMARY_MODE, MODEL_BY_KEY, MODEL_DISPLAY, models_for_dataset, get_dataset
from utils.io_helpers import stage2_dir, analysis_dir, build_md_table

B_DEFAULT = 2000
SEED = 42


def _clip_errors(ref: str, hyp: str) -> tuple[int, int]:
    """Return (word_errors, ref_words) for one clip, matching corpus-WER accounting."""
    ref = "" if not isinstance(ref, str) else ref
    hyp = "" if not isinstance(hyp, str) else hyp
    n_ref = len(ref.split())
    if n_ref == 0:
        return 0, 0
    if not hyp.strip():
        return n_ref, n_ref  # empty hypothesis => all reference words are deletions
    out = jiwer.process_words([ref], [hyp])
    return out.substitutions + out.deletions + out.insertions, n_ref


def _load_clip_table(dataset: str, model: str, mode: str) -> pd.DataFrame | None:
    path = os.path.join(stage2_dir(dataset), mode, f"wer_{model}_{mode}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ids = df["ID"].astype(str)
    if ids.duplicated().any():
        dupes = ids[ids.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"[statistics] {path}: {ids.duplicated().sum()} duplicate clip IDs (e.g. {dupes}) — "
            f"per-clip joins would silently misalign. Fix the dataset id column / Stage 1 output."
        )
    errs, words = zip(*(_clip_errors(r, h) for r, h in zip(df["reference"], df["hypothesis"])))
    # NaN-safe: astype(str) would turn missing speakers into the literal string
    # "nan", silently merging unrelated clips into one giant pseudo-speaker cluster.
    speaker = (df["Speaker_ID"].map(lambda v: "" if pd.isna(v) else str(v).strip())
               if "Speaker_ID" in df.columns else pd.Series([""] * len(df)))
    out = pd.DataFrame({"ID": ids, "errors": errs, "ref_words": words,
                        "speaker": speaker.values})
    return out.set_index("ID")


def _bootstrap_paired(E: dict, W: np.ndarray, B: int, rng) -> tuple[dict, np.ndarray]:
    """Shared-index bootstrap over rows of E[m] / W. Returns per-model (B,) corpus WERs."""
    n = len(W)
    idx = rng.integers(0, n, size=(B, n))
    sw = W[idx].sum(axis=1)
    return {m: e[idx].sum(axis=1) / sw for m, e in E.items()}, sw


def _holm(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def analyze(dataset: str, mode: str, B: int = B_DEFAULT):
    spec = get_dataset(dataset)
    tables = {}
    # One hypothesis family per analysis: only the headline (chart) models enter
    # this pairwise table + Holm correction. The FT-study variants are a separate
    # controlled comparison with its own paired test (analysis/compare_finetune.py);
    # mixing the families would penalize the headline comparisons for tests that
    # belong to a different question.
    for m in models_for_dataset(dataset):
        if not MODEL_BY_KEY[m].chart:
            continue
        t = _load_clip_table(dataset, m, mode)
        if t is not None:
            tables[m] = t
    models = list(tables)
    if not models:
        return None

    # Common clips (intersection) so all models are compared on identical resamples.
    common = None
    for t in tables.values():
        common = set(t.index) if common is None else (common & set(t.index))
    common = sorted(common)
    N = len(common)
    for m, t in tables.items():
        if len(t) != N:
            print(f"  [WARNING] {m}: scored table has {len(t)} clips but the cross-model "
                  f"intersection is {N}. Its standalone Stage-2 corpus WER covers a DIFFERENT "
                  f"clip set than the numbers below — do not mix them in the paper.")

    ref_words = tables[models[0]].loc[common, "ref_words"].to_numpy()
    E_clip = {m: tables[m].loc[common, "errors"].to_numpy() for m in models}
    point = {m: E_clip[m].sum() / ref_words.sum() for m in models}

    # --- Cluster structure: speakers if available, else clips ---
    speakers = tables[models[0]].loc[common, "speaker"].to_numpy()
    have_speakers = pd.Series(speakers).replace("", np.nan).notna().sum() > 0 and \
        len(set(s for s in speakers if s)) > 1
    if have_speakers:
        # clips with a missing speaker id become their own singleton cluster
        labels = np.array([s if s else f"clip:{cid}" for s, cid in zip(speakers, common)])
        cluster_unit = "speaker"
    else:
        labels = np.array([f"clip:{cid}" for cid in common])
        cluster_unit = "clip"
        print(f"  [note] '{dataset}' exposes no speaker id — falling back to clip-level "
              f"resampling; CIs may understate within-speaker correlation.")

    uniq = sorted(set(labels))
    G = len(uniq)
    gpos = {g: i for i, g in enumerate(uniq)}
    gidx = np.array([gpos[l] for l in labels])
    W_grp = np.bincount(gidx, weights=ref_words, minlength=G)
    E_grp = {m: np.bincount(gidx, weights=E_clip[m], minlength=G) for m in models}

    rng = np.random.default_rng(SEED)
    boot_cl, _ = _bootstrap_paired(E_grp, W_grp, B, rng)       # cluster-level (primary)
    rng2 = np.random.default_rng(SEED)
    boot_clip, _ = _bootstrap_paired(E_clip, ref_words, B, rng2)  # clip-level (secondary)

    per_model = []
    for m in models:
        lo, hi = np.percentile(boot_cl[m], [2.5, 97.5])
        lo_c, hi_c = np.percentile(boot_clip[m], [2.5, 97.5])
        per_model.append({
            "model": m, "display": MODEL_DISPLAY.get(m, m),
            "n_clips": N, "n_clusters": G, "cluster_unit": cluster_unit,
            "corpus_wer_pct": round(point[m] * 100, 2),
            "ci_lo_pct": round(lo * 100, 2), "ci_hi_pct": round(hi * 100, 2),
            "ci_halfwidth_pp": round((hi - lo) / 2 * 100, 2),
            "ci_lo_clip_pct": round(lo_c * 100, 2), "ci_hi_clip_pct": round(hi_c * 100, 2),
        })
    per_model.sort(key=lambda r: r["corpus_wer_pct"])

    # Pairwise paired significance on the CLUSTER bootstrap (the defensible unit).
    pairwise = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            d = boot_cl[a] - boot_cl[b]
            obs = point[a] - point[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            # add-one smoothing avoids p=0 artifacts at finite B
            p = 2.0 * min(((d <= 0).sum() + 1) / (B + 1), ((d >= 0).sum() + 1) / (B + 1))
            pairwise.append({
                "model_a": a, "model_b": b,
                "diff_pp": round(obs * 100, 2),
                "ci_lo_pp": round(lo * 100, 2), "ci_hi_pp": round(hi * 100, 2),
                "p_value": min(round(p, 4), 1.0),
            })
    adj = _holm([r["p_value"] for r in pairwise])
    for r, pa in zip(pairwise, adj):
        r["p_holm"] = round(pa, 4)
        r["significant_holm_0.05"] = "yes" if pa < 0.05 else "no"

    return {"models": models, "per_model": per_model, "pairwise": pairwise,
            "N": N, "G": G, "cluster_unit": cluster_unit}


def main(dataset: str, mode: str, B: int) -> None:
    spec = get_dataset(dataset)
    res = analyze(dataset, mode, B)
    out = analysis_dir(dataset)

    if res is None:
        print(f"[statistics] {spec.display} / {mode}: no scored clip tables found — nothing to analyze.")
        return

    per_model, pairwise = res["per_model"], res["pairwise"]
    N, G, unit = res["N"], res["G"], res["cluster_unit"]

    df_pm = pd.DataFrame(per_model)
    df_pw = pd.DataFrame(pairwise)
    df_pm.to_csv(os.path.join(out, f"statistics_{mode}.csv"), index=False)
    df_pw.to_csv(os.path.join(out, f"statistics_pairwise_{mode}.csv"), index=False)

    md_pm = df_pm[["display", "corpus_wer_pct", "ci_lo_pct", "ci_hi_pct", "ci_halfwidth_pp"]].copy()
    md_pm.columns = ["Model", "Corpus WER %", "CI low", "CI high", "±pp"]
    with open(os.path.join(out, f"statistics_{mode}.md"), "w") as f:
        f.write(f"# Statistical significance — {spec.display} — mode `{mode}`\n\n")
        f.write(f"Corpus WER with 95% bootstrap CI: {B} resamples, seed {SEED}, N={N} clips, "
                f"resampled by **{unit}** ({G} clusters). Headline (chart) models only — "
                f"the fine-tuning study is a separate hypothesis family with its own paired "
                f"test in `finetune_comparison.md`. ")
        if unit == "speaker":
            f.write("Speaker-level resampling accounts for within-speaker correlation "
                    "(clips from one speaker share accent/channel); clip-level CIs are in the "
                    "CSV for comparison and are narrower, i.e. anti-conservative.\n\n")
        else:
            f.write("No speaker id is exposed for this dataset, so resampling is clip-level; "
                    "CIs may understate within-speaker correlation (limitation).\n\n")
        f.write(build_md_table(md_pm) + "\n\n")
        f.write("## Pairwise paired significance\n\n")
        f.write("Difference = WER(A) − WER(B) in pp; paired bootstrap on identical "
                f"{unit}-level resamples; two-sided p-values with Holm–Bonferroni correction "
                f"across all {len(pairwise)} pairs.\n\n")
        f.write(build_md_table(df_pw) + "\n")
    print(f"[statistics] {spec.display} / {mode}: wrote statistics_{mode}.{{csv,md}} + pairwise "
          f"({N} clips, {G} {unit}s, B={B})")
    print(df_pm[["display", "corpus_wer_pct", "ci_lo_pct", "ci_hi_pct",
                 "ci_lo_clip_pct", "ci_hi_clip_pct"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tie")
    ap.add_argument("--mode", default=PRIMARY_MODE)
    ap.add_argument("--bootstrap", type=int, default=B_DEFAULT)
    a = ap.parse_args()
    main(a.dataset, a.mode, a.bootstrap)
