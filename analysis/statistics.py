"""
Stage 3 (statistics): bootstrap confidence intervals + paired significance tests.

Fixes the paper's biggest rigour gap: headline WER deltas ("0.33pp", "no
significant gain") were reported without uncertainty. This computes, per dataset
and evaluation mode:

  * a 95% bootstrap CI on each model's CORPUS WER (resampling clips with
    replacement and recomputing Sum(errors)/Sum(ref_words) — NOT the mean of
    per-clip WER, which is a different, tail-inflated quantity), and
  * paired bootstrap significance for every model pair (same resampled clip
    indices for both models, so the comparison is properly paired), reporting the
    WER difference, its 95% CI, and a two-sided p-value.

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

from utils.registry import PRIMARY_MODE, MODEL_DISPLAY, models_for_dataset, get_dataset
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
    errs, words = zip(*(_clip_errors(r, h) for r, h in zip(df["reference"], df["hypothesis"])))
    return pd.DataFrame({"ID": df["ID"].astype(str), "errors": errs, "ref_words": words}).set_index("ID")


def analyze(dataset: str, mode: str, B: int = B_DEFAULT):
    models = [m for m in models_for_dataset(dataset)
              if _load_clip_table(dataset, m, mode) is not None]
    if not models:
        return [], [], 0
    tables = {m: _load_clip_table(dataset, m, mode) for m in models}

    # Common clips (intersection) so all models are compared on identical resamples.
    common = None
    for t in tables.values():
        common = set(t.index) if common is None else (common & set(t.index))
    common = sorted(common)
    N = len(common)

    ref_words = tables[models[0]].loc[common, "ref_words"].to_numpy()
    E = {m: tables[m].loc[common, "errors"].to_numpy() for m in models}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, N, size=(B, N))            # shared resample indices (paired)
    sw = ref_words[idx].sum(axis=1)                   # (B,) total ref words per resample
    boot = {m: E[m][idx].sum(axis=1) / sw for m in models}   # (B,) corpus WER per resample
    point = {m: E[m].sum() / ref_words.sum() for m in models}

    # Per-model corpus WER + 95% CI
    per_model = []
    for m in models:
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        per_model.append({
            "model": m, "display": MODEL_DISPLAY.get(m, m), "n_clips": N,
            "corpus_wer_pct": round(point[m] * 100, 2),
            "ci_lo_pct": round(lo * 100, 2), "ci_hi_pct": round(hi * 100, 2),
            "ci_halfwidth_pp": round((hi - lo) / 2 * 100, 2),
        })
    per_model.sort(key=lambda r: r["corpus_wer_pct"])

    # Pairwise paired significance
    pairwise = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            d = boot[a] - boot[b]
            obs = point[a] - point[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
            p = min(p, 1.0)
            pairwise.append({
                "model_a": a, "model_b": b,
                "diff_pp": round(obs * 100, 2),
                "ci_lo_pp": round(lo * 100, 2), "ci_hi_pp": round(hi * 100, 2),
                "p_value": round(p, 4),
                "significant_0.05": "yes" if (lo > 0 or hi < 0) else "no",
            })
    return per_model, pairwise, N


def main(dataset: str, mode: str, B: int) -> None:
    spec = get_dataset(dataset)
    per_model, pairwise, N = analyze(dataset, mode, B)
    out = analysis_dir(dataset)

    if not per_model:
        print(f"[statistics] {spec.display} / {mode}: no scored clip tables found — nothing to analyze.")
        return

    df_pm = pd.DataFrame(per_model)
    df_pw = pd.DataFrame(pairwise)
    df_pm.to_csv(os.path.join(out, f"statistics_{mode}.csv"), index=False)
    df_pw.to_csv(os.path.join(out, f"statistics_pairwise_{mode}.csv"), index=False)

    md_pm = df_pm[["display", "corpus_wer_pct", "ci_lo_pct", "ci_hi_pct", "ci_halfwidth_pp"]].copy()
    md_pm.columns = ["Model", "Corpus WER %", "CI low", "CI high", "±pp"]
    with open(os.path.join(out, f"statistics_{mode}.md"), "w") as f:
        f.write(f"# Statistical significance — {spec.display} — mode `{mode}`\n\n")
        f.write(f"Corpus WER with 95% bootstrap CI ({B} resamples, seed {SEED}, N={N} clips).\n\n")
        f.write(build_md_table(md_pm) + "\n\n")
        f.write("## Pairwise paired significance\n\n")
        f.write("Difference = WER(A) − WER(B) in pp; CI and two-sided bootstrap p-value; "
                "paired on identical resampled clips.\n\n")
        f.write(build_md_table(df_pw) + "\n")
    print(f"[statistics] {spec.display} / {mode}: wrote statistics_{mode}.{{csv,md}} + pairwise ({N} clips, B={B})")
    print(df_pm[["display", "corpus_wer_pct", "ci_lo_pct", "ci_hi_pct"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tie")
    ap.add_argument("--mode", default=PRIMARY_MODE)
    ap.add_argument("--bootstrap", type=int, default=B_DEFAULT)
    a = ap.parse_args()
    main(a.dataset, a.mode, a.bootstrap)
