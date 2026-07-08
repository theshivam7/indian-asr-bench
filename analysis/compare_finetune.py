"""
Fine-tuning capacity study: pretrained vs fine-tuned, per model size.

One report per size (Tiny / Small / Medium), each comparing:
    Headline   : <size>_ft  vs  <size>_hf   (pretrained through the SAME HF chunked
                 pipeline). Same engine, same decoding => the delta is the true FT gain.
    Secondary  : <size>_ft  vs  <size>       (the original openai-whisper number, for continuity).

Medium additionally carries the multi-seed speaker-disjoint / size-matched-control study
(Tiny/Small use the minimal protocol only — one official-split fine-tune each; see
results/tie/analysis/findings_tiny_small_ft.md for why).

Reads results/tie/stage2_processed/{mode}/wer_{model}_{mode}.csv
(produced by normalize_and_score.py --dataset tie). Writes, per size:
    results/tie/analysis/finetune_comparison[_<size>].md
    results/tie/analysis/finetune_comparison[_<size>].png
    results/tie/analysis/finetune_wer_distribution[_<size>].png
    results/tie/analysis/finetune_disjoint_forest[_<size>].png   (medium only, this phase)
and once across all sizes:
    results/tie/analysis/finetune_capacity_summary.md / .csv

Run after normalize_and_score.py --dataset tie.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.wer_compute import compute_corpus_wer
from utils.registry import ALL_MODES as MODES, PRIMARY_MODE, MODEL_ORDER, MODEL_BY_KEY
from utils.io_helpers import stage2_dir, analysis_dir
from analysis.statistics import _clip_errors, _holm, analyze

# The fine-tuning study is TIE-only (Svarah is eval-only, not fine-tunable).
DATASET = "tie"

# Speaker-disjoint re-split FT, one entry per training seed (medium only this phase). A null
# result from a single seed is not credible (FT seed variance ~ the effect size being denied),
# so the study runs 3 seeds and reports the spread + a paired bootstrap CI.
DISJOINT_SEEDS = {
    "medium_ft_disjoint": 42,
    "medium_ft_disjoint_s43": 43,
    "medium_ft_disjoint_s44": 44,
}
# Size-matched (speaker-OVERLAPPING) control runs (medium only this phase): same clip count
# as the disjoint train set, sampled at random from the full train split. Separates the
# effect of the ~13x training-set shrinkage from the effect of speaker-disjointness itself.
SIZEMATCH_SEEDS = {
    "medium_ft_sizematch_s42": 42,
    "medium_ft_sizematch_s43": 43,
    "medium_ft_sizematch_s44": 44,
}
# Disjoint train-set size, printed by finetune.py's "[speaker-disjoint]" log
# line and independently recomputed from the dataset's Metadata.csv
# (2026-07-03), applying the same filters finetune.py applies (non-empty
# transcript, <=30 s) BEFORE the speaker filter: removing the 280 test speakers
# keeps 51/331 speakers, 567/7200 clips, 3.8/46.9 hours.
DISJOINT_TRAIN = {"clips": 567, "clips_total": 7200, "speakers": 51,
                  "speakers_total": 331, "hours": 3.8, "hours_total": 46.9}

# One entry per model size. Tiny/Small run the minimal protocol (official-split FT vs its
# own HF-pipeline pretrained baseline only); Medium keeps the full disjoint/size-matched
# seed study. out_stem="finetune_comparison" for medium preserves the exact historical
# filenames (finetune_comparison.md/.png, finetune_wer_distribution.png,
# finetune_disjoint_forest.png) so this refactor reproduces medium's committed outputs
# byte-for-byte (verify with `git diff` after regenerating).
FT_PAIRS = (
    dict(key="tiny", display_name="Whisper Tiny", params="39M",
         secondary="tiny", baseline="tiny_hf", finetuned="tiny_ft",
         out_stem="finetune_comparison_tiny", disjoint={}, sizematch={}),
    dict(key="small", display_name="Whisper Small", params="244M",
         secondary="small", baseline="small_hf", finetuned="small_ft",
         out_stem="finetune_comparison_small", disjoint={}, sizematch={}),
    dict(key="medium", display_name="Whisper Medium", params="769M",
         secondary="medium", baseline="medium_hf", finetuned="medium_ft",
         out_stem="finetune_comparison", disjoint=DISJOINT_SEEDS, sizematch=SIZEMATCH_SEEDS),
)

STAGE2_DIR = stage2_dir(DATASET)
ANALYSIS_DIR = analysis_dir(DATASET)


def load(model: str, mode: str) -> pd.DataFrame | None:
    path = os.path.join(STAGE2_DIR, mode, f"wer_{model}_{mode}.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    return pd.read_csv(path)


def corpus_wer(df: pd.DataFrame) -> float:
    refs = df["reference"].fillna("").tolist()
    hyps = df["hypothesis"].fillna("").tolist()
    return compute_corpus_wer(refs, hyps)["corpus_wer"] * 100


def paired_speaker_bootstrap(df_base: pd.DataFrame, df_ft: pd.DataFrame,
                             B: int = 2000, seed: int = 42):
    """Paired bootstrap CI + p for corpus-WER(ft) − corpus-WER(base), resampling
    SPEAKERS (accounts for within-speaker correlation; see analysis/statistics.py).

    Returns (diff_pp, ci_lo_pp, ci_hi_pp, p_value, n_clips, n_speakers)."""
    cols = ["ID", "reference", "hypothesis"]
    a = df_base[cols + ["Speaker_ID"]].merge(df_ft[cols], on="ID", suffixes=("_a", "_b"))
    ea, wa = zip(*(_clip_errors(r, h) for r, h in zip(a["reference_a"], a["hypothesis_a"])))
    eb, _ = zip(*(_clip_errors(r, h) for r, h in zip(a["reference_b"], a["hypothesis_b"])))
    ea, eb, wa = np.array(ea, float), np.array(eb, float), np.array(wa, float)

    # NaN-safe (astype(str) would merge missing speakers into one "nan" cluster)
    spk = a["Speaker_ID"].map(lambda v: "" if pd.isna(v) else str(v).strip())
    labels = np.where(spk != "", spk, "clip:" + a["ID"].astype(str))
    uniq = sorted(set(labels))
    gpos = {g: i for i, g in enumerate(uniq)}
    gidx = np.array([gpos[l] for l in labels])
    G = len(uniq)
    Ea = np.bincount(gidx, weights=ea, minlength=G)
    Eb = np.bincount(gidx, weights=eb, minlength=G)
    W = np.bincount(gidx, weights=wa, minlength=G)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(B, G))
    sw = W[idx].sum(axis=1)
    d = (Eb[idx].sum(axis=1) - Ea[idx].sum(axis=1)) / sw   # ft − base, per resample
    obs = (eb.sum() - ea.sum()) / wa.sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min(((d <= 0).sum() + 1) / (B + 1), ((d >= 0).sum() + 1) / (B + 1))
    return obs * 100, lo * 100, hi * 100, min(p, 1.0), len(a), G


def fmt_delta(base: float, ft: float) -> tuple[str, str]:
    """Absolute (pp) and relative (%) change of ft vs base.

    WER going down is an improvement, shown with a leading '−'; a regression
    (WER up) is shown with a leading '+'. So '−2.50 pp' = 2.5pp better,
    '+1.20 pp' = 1.2pp worse.
    """
    abs_pp = base - ft
    rel = (abs_pp / base * 100) if base else 0.0
    sign = "−" if abs_pp > 0 else "+"  # WER down (abs_pp>0) = improvement
    return f"{sign}{abs(abs_pp):.2f} pp", f"{sign}{abs(rel):.1f}%"


def run_pair(pair: dict) -> dict:
    """Generate the full pretrained-vs-fine-tuned report + charts for one model size.

    Always returns a dict with key/display_name/params plus a "headline" entry — that
    entry is a small dict of headline-comparison stats (for the cross-size capacity
    summary) if both the HF baseline and fine-tuned results were available, else None.
    The returned stats are NOT written into this pair's own .md — they exist only to
    feed finetune_capacity_summary.{md,csv} below, so medium's committed report text
    is unaffected by this addition.
    """
    key = pair["key"]
    display_name = pair["display_name"]
    params = pair["params"]
    secondary = pair["secondary"]
    baseline = pair["baseline"]
    finetuned = pair["finetuned"]
    disjoint_seeds = pair["disjoint"]
    sizematch_seeds = pair["sizematch"]
    out_stem = pair["out_stem"]
    suffix = out_stem[len("finetune_comparison"):]   # "" for medium, "_tiny"/"_small" otherwise

    print("=" * 70)
    print(f"PRETRAINED vs FINE-TUNED — {display_name}")
    print("=" * 70)

    lines = [
        f"# {display_name} — Pretrained vs Fine-tuned",
        "",
        "Fine-tuned on the `raianand/TIE_shorts` **train** split (7,884 raw clips; ~7,200 remain",
        "after dropping empty transcripts, clips >30s, and clips with no embedded audio — see the",
        "run log for the exact realized count), best checkpoint selected on the **validation**",
        "split, evaluated on the **test** split (986 clips) — the same test set used for every",
        "pretrained model in this benchmark.",
        "",
        f"**Headline comparison** is against `{baseline}` — the *pretrained* {display_name} run",
        "through the **same** HuggingFace chunked pipeline as the fine-tuned model. This isolates",
        "the fine-tuning gain from any decoding/engine differences. The original `openai-whisper`",
        "number is shown as a secondary reference.",
        "",
    ]

    # --------------- 1. Corpus WER across all modes ---------------
    print("\n--- Corpus WER by mode ---")
    all_ft_models = (secondary, baseline, finetuned, *disjoint_seeds, *sizematch_seeds)
    have = {m: {} for m in all_ft_models}
    for mode in MODES:
        for model in all_ft_models:
            df = load(model, mode)
            if df is not None:
                have[model][mode] = corpus_wer(df)

    lines += [
        "## Corpus WER (%) by evaluation mode",
        "",
        "| Mode | Pretrained (HF) | Fine-tuned | Δ abs | Δ rel | _openai-whisper ref_ |",
        "|------|:---------------:|:----------:|:-----:|:-----:|:--------------------:|",
    ]
    for mode in MODES:
        base = have[baseline].get(mode)
        ft = have[finetuned].get(mode)
        sec = have[secondary].get(mode)
        if base is None or ft is None:
            continue
        d_abs, d_rel = fmt_delta(base, ft)
        star = " **(gold)**" if mode == PRIMARY_MODE else ""
        sec_str = f"{sec:.2f}%" if sec is not None else "N/A"
        lines.append(
            f"| `{mode}`{star} | {base:.2f}% | {ft:.2f}% | {d_abs} | {d_rel} | {sec_str} |"
        )
        print(f"  {mode:18s} base(HF)={base:.2f}%  ft={ft:.2f}%  delta={d_abs}")
    lines.append("")

    # Headline callout on the gold mode.
    headline_stats = None
    if PRIMARY_MODE in have[baseline] and PRIMARY_MODE in have[finetuned]:
        b = have[baseline][PRIMARY_MODE]
        f = have[finetuned][PRIMARY_MODE]
        d_abs, d_rel = fmt_delta(b, f)
        verdict = "improves" if f < b else "does NOT improve"
        lines += [
            f"> **Headline ({PRIMARY_MODE})**: fine-tuning {verdict} WER "
            f"{b:.2f}% → {f:.2f}%  ({d_abs}, {d_rel} relative).",
            "",
        ]

        df_b = load(baseline, PRIMARY_MODE)
        df_f = load(finetuned, PRIMARY_MODE)
        if df_b is not None and df_f is not None:
            d, lo, hi, p, n, g = paired_speaker_bootstrap(df_b, df_f)
            headline_stats = dict(
                pretrained_openai_wer=have[secondary].get(PRIMARY_MODE),
                hf_baseline_wer=b, ft_wer=f, delta_pp=d, ci_lo_pp=lo, ci_hi_pp=hi,
                p=p, n_clips=n, n_speakers=g,
            )

    # --------------- 1b. Speaker-disjoint re-split fine-tune (multi-seed; medium only) ---------------
    disjoint_present = [m for m in disjoint_seeds if have[m].get(PRIMARY_MODE) is not None]
    sm_present = [m for m in sizematch_seeds if have[m].get(PRIMARY_MODE) is not None]
    if disjoint_present:
        dt = DISJOINT_TRAIN
        lines += [
            "## Speaker-disjoint re-split fine-tune (multi-seed)",
            "",
            "Same recipe as the headline fine-tune, but every train clip whose speaker also "
            "appears in `test` is removed first (see `speaker_overlap.md`). Evaluated on the "
            f"SAME test set as `{baseline}`, so any gain here cannot come from speaker "
            "adaptation. Run with multiple training seeds: a null claim from one seed would "
            "be indistinguishable from seed variance.",
            "",
            f"> **Training-set confound (disclosed)**: TIE_shorts' official test speakers are so "
            f"entangled with train that removing them keeps only **{dt['clips']}/{dt['clips_total']} "
            f"train clips ({dt['hours']}/{dt['hours_total']} h, {dt['speakers']}/{dt['speakers_total']} "
            f"speakers)**. The disjoint runs therefore differ from the headline fine-tune in BOTH "
            f"speaker overlap and training-set size (~13x smaller) — this dataset cannot support a "
            f"size-matched speaker-disjoint split at all, which is itself an evaluation-validity "
            f"finding. Any WER regression below must not be attributed to speaker-disjointness "
            f"alone; see the size-matched control section "
            + ("below for the separation." if sm_present else "(when run) for the separation."),
            "",
            f"| Seed | WER (`{PRIMARY_MODE}`) | Δ vs pretrained (paired, speaker-resampled) | 95% CI | p | p (Holm) |",
            "|------|:----:|:----:|:----:|:----:|:----:|",
        ]
        df_base_pm = load(baseline, PRIMARY_MODE)
        diffs, halfwidths, pvals = [], [], []
        seed_rows = []
        for m in disjoint_present:
            seed = disjoint_seeds[m]
            wer_m = have[m][PRIMARY_MODE]
            df_m = load(m, PRIMARY_MODE)
            d, lo, hi, p, n, g = paired_speaker_bootstrap(df_base_pm, df_m)
            diffs.append(d)
            halfwidths.append((hi - lo) / 2)
            pvals.append(p)
            seed_rows.append((seed, wer_m, d, lo, hi, p, n, g))

        p_holm = _holm(pvals) if len(pvals) > 1 else pvals
        n_sig_holm = 0
        for (seed, wer_m, d, lo, hi, p, n, g), ph in zip(seed_rows, p_holm):
            sig = "" if lo <= 0 <= hi else " *"
            if ph < 0.05:
                n_sig_holm += 1
            lines.append(f"| {seed} | {wer_m:.2f}% | {d:+.2f} pp | [{lo:+.2f}, {hi:+.2f}]{sig} | {p:.3f} | {ph:.3f} |")
            print(f"  [disjoint s{seed}] wer={wer_m:.2f}%  diff={d:+.2f}pp  CI=[{lo:+.2f},{hi:+.2f}]  p={p:.3f}  p_holm={ph:.3f}  ({n} clips, {g} speakers)")
        if len(disjoint_present) > 1:
            lines.append("\n_\\* = uncorrected 95% CI excludes 0. Use the Holm-adjusted p (multiplicity-corrected "
                          f"across these {len(disjoint_present)} seeds) for significance calls._")
        lines.append("")

        # Forest plot: per-seed delta vs pretrained with 95% speaker-clustered CIs,
        # Holm-marked, with the minimum-detectable-effect band. The figure version of
        # the seed table above (committed so the README can embed it).
        mde_fig = float(np.mean(halfwidths))
        fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(seed_rows) + 1.6))
        ys = np.arange(len(seed_rows))[::-1]
        ax.axvspan(-mde_fig, mde_fig, color="#EEEEEE", zorder=0)
        ax.axvline(0, color="#555555", linewidth=1.0, zorder=1)
        for yi, (seed, wer_m, d, lo, hi, p, n, g), ph in zip(ys, seed_rows, p_holm):
            sig = ph < 0.05
            c = "#D55E00" if sig else "#0072B2"
            ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", ms=7, color=c,
                        ecolor=c, elinewidth=1.6, capsize=3.5, zorder=2)
            ax.text(hi + 0.12, yi, f"p={ph:.3f}" + ("*" if sig else ""),
                    fontsize=9, va="center", color=c)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"seed {s}" for s, *_ in seed_rows])
        ax.set_xlabel("Δ corpus WER vs. pretrained (pp) — 95% speaker-clustered bootstrap CI")
        ax.text(-mde_fig + 0.06, ys[0] - 0.42, f"shaded: MDE ≈ {mde_fig:.1f} pp",
                fontsize=8.5, ha="left", va="top", color="#666666")
        ax.set_title(f"{display_name}: speaker-disjoint fine-tune per-seed effect (Holm-corrected)")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(left=min(-mde_fig - 0.4, min(r[3] for r in seed_rows) - 0.3),
                    right=max(r[4] for r in seed_rows) + 1.2)
        fig.tight_layout()
        forest_path = os.path.join(ANALYSIS_DIR, f"finetune_disjoint_forest{suffix}.png")
        fig.savefig(forest_path, dpi=150)
        plt.close(fig)
        print(f"  Saved chart: {forest_path}")

        wers = [have[m][PRIMARY_MODE] for m in disjoint_present]
        b = have[baseline][PRIMARY_MODE]
        if len(disjoint_present) > 1:
            lines += [
                f"Across {len(disjoint_present)} seeds: WER {np.mean(wers):.2f}% "
                f"(range {min(wers):.2f}–{max(wers):.2f}%), mean Δ vs pretrained "
                f"{np.mean(diffs):+.2f} pp; seed-to-seed spread "
                f"{max(wers) - min(wers):.2f} pp.",
                "",
            ]
        mde = float(np.mean(halfwidths))
        if n_sig_holm == 0:
            lines += [
                f"> **Minimum detectable effect**: the paired 95% CI half-width is ≈{mde:.2f} pp, "
                f"so a true fine-tuning gain of ≥{mde:.2f} pp would have been detected. The "
                f"observed differences ({', '.join(f'{d:+.2f}' for d in diffs)} pp) are within "
                f"that resolution — the correct claim is *any residual gain is below "
                f"≈{mde:.1f} pp*, not merely 'not significant'.",
                "",
            ]
        else:
            lines += [
                f"> **Mixed result, not a clean null**: {n_sig_holm}/{len(disjoint_present)} seed(s) show a "
                f"Holm-corrected significant WORSENING relative to pretrained (fine-tuning increases WER), "
                f"while the remaining seed(s) fall within the ≈{mde:.2f} pp minimum detectable effect. "
                f"The seed-to-seed spread ({max(wers) - min(wers):.2f} pp) is itself larger than the per-seed "
                f"effect being estimated, so a single-seed run — including the checkpoint published as the "
                f"'primary' disjoint model — is not representative of the study as a whole. The safe claim is: "
                f"fine-tuning on the speaker-disjoint training subset ({DISJOINT_TRAIN['clips']} clips) shows "
                f"no evidence of improving WER over pretrained, and at least one seed shows evidence of making "
                f"it worse. Whether the worsening is caused by the disjointness or by the 13x-smaller training "
                f"set is separated by the size-matched control below"
                + ("." if sm_present else " (if run)."),
                "",
            ]

        # ---- Size-matched speaker-overlapping control (separates size from disjointness) ----
        if sm_present:
            lines += [
                "## Size-matched control (speaker-overlapping, multi-seed)",
                "",
                f"Same recipe and clip count as the disjoint runs ({DISJOINT_TRAIN['clips']} train clips), "
                "but sampled at random from the FULL train split — speaker overlap with test is preserved. "
                "If these runs regress like the disjoint runs, the disjoint regression is a small-training-set "
                "effect; if they hold up, the disjointness itself is implicated.",
                "",
                f"| Seed | WER (`{PRIMARY_MODE}`) | Δ vs pretrained (paired, speaker-resampled) | 95% CI | p | p (Holm) |",
                "|------|:----:|:----:|:----:|:----:|:----:|",
            ]
            sm_pvals, sm_rows = [], []
            for m in sm_present:
                d, lo, hi, p, n, g = paired_speaker_bootstrap(df_base_pm, load(m, PRIMARY_MODE))
                sm_pvals.append(p)
                sm_rows.append((sizematch_seeds[m], have[m][PRIMARY_MODE], d, lo, hi, p))
            sm_holm = _holm(sm_pvals) if len(sm_pvals) > 1 else sm_pvals
            for (seed, wer_m, d, lo, hi, p), ph in zip(sm_rows, sm_holm):
                sig = "" if lo <= 0 <= hi else " *"
                lines.append(f"| {seed} | {wer_m:.2f}% | {d:+.2f} pp | [{lo:+.2f}, {hi:+.2f}]{sig} | {p:.3f} | {ph:.3f} |")
                print(f"  [sizematch s{seed}] wer={wer_m:.2f}%  diff={d:+.2f}pp  CI=[{lo:+.2f},{hi:+.2f}]  p={p:.3f}  p_holm={ph:.3f}")
            lines.append("")

            n_sig_sm = sum(1 for ph in sm_holm if ph < 0.05)
            if n_sig_sm == 0 and n_sig_holm > 0:
                lines += [
                    f"> **Confound resolved**: all {len(sm_present)} size-matched seeds are statistically "
                    f"indistinguishable from pretrained (0/{len(sm_present)} Holm-significant), while "
                    f"{n_sig_holm}/{len(disjoint_present)} disjoint seed(s) regressed significantly. Since "
                    f"both conditions train on the identical {DISJOINT_TRAIN['clips']}-clip count, training-set "
                    f"size alone cannot explain the disjoint regression — **speaker-disjointness is the cause**, "
                    f"not the smaller training set.",
                    "",
                ]
            elif n_sig_sm > 0 and n_sig_holm > 0:
                lines += [
                    f"> **Size effect implicated**: {n_sig_sm}/{len(sm_present)} size-matched seed(s) also "
                    f"regressed significantly despite preserving speaker overlap, so the small training-set "
                    f"size (not disjointness alone) contributes to the disjoint-run regression.",
                    "",
                ]
            elif n_sig_sm == 0 and n_sig_holm == 0:
                lines += [
                    "> **Inconclusive**: neither the disjoint nor the size-matched seeds show a "
                    "Holm-significant effect vs pretrained — the study is underpowered to separate the "
                    "two confounds at this seed count.",
                    "",
                ]
            else:
                lines += [
                    f"> **Unexpected pattern**: {n_sig_sm}/{len(sm_present)} size-matched seed(s) regressed "
                    f"significantly but no disjoint seed did — re-check the runs before drawing a causal "
                    f"conclusion.",
                    "",
                ]
        else:
            lines += [
                "_Size-matched control runs (`medium_ft_sizematch_s*`) not yet available — "
                "submit `hpc/job_finetune_sizematch.pbs` (SEED=42/43/44) to separate the "
                "training-set-size effect from the speaker-disjointness effect._",
                "",
            ]
    elif disjoint_seeds:
        print(f"  [SKIP] no {finetuned.rsplit('_ft', 1)[0]}_ft_disjoint results yet")

    # --------------- 2. Breakdowns (primary mode) ---------------
    df_base = load(baseline, PRIMARY_MODE)
    df_ft = load(finetuned, PRIMARY_MODE)

    if df_base is not None and df_ft is not None:
        breakdowns = [
            ("Native_Region", "Region"),
            ("Speech_Class", "Speech rate"),
            ("Gender", "Gender"),
            ("Discipline_Group", "Discipline"),
        ]
        for col, title in breakdowns:
            if col not in df_base.columns or col not in df_ft.columns:
                continue
            lines += [f"## By {title} (`{PRIMARY_MODE}`)", "",
                      "| Group | Pretrained (HF) | Fine-tuned | Δ abs | Samples |",
                      "|-------|:---------------:|:----------:|:-----:|:-------:|"]
            for g in sorted(set(df_base[col].dropna()) | set(df_ft[col].dropna())):
                gb = df_base[df_base[col] == g]
                gf = df_ft[df_ft[col] == g]
                if gb.empty or gf.empty:
                    continue
                b = corpus_wer(gb)
                f = corpus_wer(gf)
                d_abs, _ = fmt_delta(b, f)
                lines.append(f"| {g} | {b:.2f}% | {f:.2f}% | {d_abs} | {len(gf)} |")
            lines.append("")

        # Duration buckets (same bins as compare_all.py).
        lines += [f"## By Audio Duration (`{PRIMARY_MODE}`)", "",
                  "| Duration | Pretrained (HF) | Fine-tuned | Δ abs |",
                  "|----------|:---------------:|:----------:|:-----:|"]
        bins = [0, 5, 15, 30, 60, float("inf")]
        labels = ["0-5s", "5-15s", "15-30s", "30-60s", "60s+"]
        for d in (df_base, df_ft):
            d["Speech_Duration_seconds"] = pd.to_numeric(d["Speech_Duration_seconds"], errors="coerce")
            d["_bucket"] = pd.cut(d["Speech_Duration_seconds"], bins=bins, labels=labels)
        for bucket in labels:
            gb = df_base[df_base["_bucket"] == bucket]
            gf = df_ft[df_ft["_bucket"] == bucket]
            if gb.empty or gf.empty:
                continue
            b = corpus_wer(gb)
            f = corpus_wer(gf)
            d_abs, _ = fmt_delta(b, f)
            lines.append(f"| {bucket} | {b:.2f}% | {f:.2f}% | {d_abs} |")
        lines.append("")

        # --------------- 3. Per-sample paired analysis ---------------
        merged = df_base[["ID", "wer"]].merge(
            df_ft[["ID", "wer"]], on="ID", suffixes=("_base", "_ft")
        )
        improved = (merged["wer_ft"] < merged["wer_base"] - 1e-9).sum()
        regressed = (merged["wer_ft"] > merged["wer_base"] + 1e-9).sum()
        unchanged = len(merged) - improved - regressed
        merged["delta"] = merged["wer_base"] - merged["wer_ft"]  # positive = FT better

        lines += [
            "## Per-sample paired analysis (`transcript_clean`)",
            "",
            f"- Samples compared: **{len(merged)}**",
            f"- Improved by fine-tuning: **{improved}** ({improved/len(merged)*100:.1f}%)",
            f"- Regressed: **{regressed}** ({regressed/len(merged)*100:.1f}%)",
            f"- Unchanged: **{unchanged}** ({unchanged/len(merged)*100:.1f}%)",
            "",
            "### Biggest improvements (top 10)",
            "",
            "| ID | Pretrained WER | Fine-tuned WER | Δ |",
            "|----|:--------------:|:--------------:|:-:|",
        ]
        gains = merged[merged["delta"] > 1e-9].sort_values("delta", ascending=False).head(10)
        for _, r in gains.iterrows():
            lines.append(f"| {r['ID']} | {r['wer_base']*100:.1f}% | {r['wer_ft']*100:.1f}% | "
                         f"−{r['delta']*100:.1f} pp |")
        lines += ["", "### Biggest regressions (top 10)", "",
                  "| ID | Pretrained WER | Fine-tuned WER | Δ |",
                  "|----|:--------------:|:--------------:|:-:|"]
        losses = merged[merged["delta"] < -1e-9].sort_values("delta", ascending=True).head(10)
        if losses.empty:
            lines.append("| _none_ | — | — | — |")
        for _, r in losses.iterrows():
            lines.append(f"| {r['ID']} | {r['wer_base']*100:.1f}% | {r['wer_ft']*100:.1f}% | "
                         f"+{-r['delta']*100:.1f} pp |")
        lines.append("")

        # --------------- 4. Chart ---------------
        fig, ax = plt.subplots(figsize=(9, 5.5))
        modes_present = [m for m in MODES if m in have[baseline] and m in have[finetuned]]
        x = range(len(modes_present))
        w = 0.38
        base_vals = [have[baseline][m] for m in modes_present]
        ft_vals = [have[finetuned][m] for m in modes_present]
        b1 = ax.bar([i - w/2 for i in x], base_vals, w, label="Pretrained (HF)", color="#888")
        b2 = ax.bar([i + w/2 for i in x], ft_vals, w, label="Fine-tuned", color="#2a7")
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(modes_present, rotation=15)
        ax.set_ylabel("Corpus WER (%)")
        ax.set_title(f"{display_name}: Pretrained vs Fine-tuned (TIE_shorts test)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        chart_path = os.path.join(ANALYSIS_DIR, f"{out_stem}.png")
        fig.savefig(chart_path)
        plt.close(fig)
        print(f"\n  Saved chart: {chart_path}")

        # --------------- 4b. WER distribution overlay (pretrained vs fine-tuned) ---------------
        # Same idea as the professor's slide: % of utterances vs per-sample WER in 5% bins.
        # A leftward shift (more mass near 0%) = fine-tuning helps. WER>100% clipped to last bin.
        dist_bins = [i * 5 for i in range(21)]  # 0, 5, ..., 100
        fig, ax = plt.subplots(figsize=(9, 5))
        for d, lab, col in [(df_base, "Pretrained (HF)", "#888888"), (df_ft, "Fine-tuned", "#2a9d4a")]:
            wv = (d["wer"].dropna().clip(upper=1.0) * 100).values
            if not len(wv):
                continue
            ax.hist(wv, bins=dist_bins, weights=[100.0 / len(wv)] * len(wv),
                    alpha=0.6, label=lab, color=col, edgecolor="white")
            ax.axvline(float(pd.Series(wv).median()), color=col, linestyle="--", linewidth=1.2)
        ax.set_xlabel("WER (%) — bin width 5%")
        ax.set_ylabel("% of utterances")
        ax.set_title(f"{display_name}: WER distribution, pretrained vs fine-tuned ({PRIMARY_MODE})")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        dist_path = os.path.join(ANALYSIS_DIR, f"finetune_wer_distribution{suffix}.png")
        fig.savefig(dist_path)
        plt.close(fig)
        print(f"  Saved chart: {dist_path}")

    # --------------- 5. Caveats ---------------
    caveats = [
        "## Caveats",
        "",
        f"- **Engine**: the headline compares fine-tuned vs *pretrained-via-HF* (`{baseline}`), both decoded",
        "  with the same chunked `transformers` pipeline, so the engine is held constant. The original",
        "  `openai-whisper` number is shown only as a continuity reference.",
        "- **Speaker overlap**: see `speaker_overlap.md`. If test speakers also appear in train, part of the",
        "  gain reflects speaker adaptation (disclosed, per the dataset's official splits).",
    ]
    if disjoint_seeds:
        caveats += [
            "- **Disjoint train-set size**: the speaker-disjoint runs train on 567 clips (3.8 h) vs the official",
            "  split's 7200 (46.9 h, after the same duration/text filters) — speaker-disjointness and training-set",
            "  size are confounded on this dataset by construction. The size-matched control isolates the size effect.",
        ]
    caveats.append("")
    lines += caveats

    report_path = os.path.join(ANALYSIS_DIR, f"{out_stem}.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved report: {report_path}")

    return dict(key=key, display_name=display_name, params=params, headline=headline_stats)


results = [run_pair(p) for p in FT_PAIRS]

# --------------- Cross-size capacity summary ---------------
capacity_rows = [r for r in results if r["headline"]]
if capacity_rows:
    pvals = [r["headline"]["p"] for r in capacity_rows]
    p_holm = _holm(pvals) if len(pvals) > 1 else pvals
    cap_lines = [
        "# Fine-tuning capacity summary — Tiny / Small / Medium (official split)",
        "",
        "One official-split fine-tune per model size (no disjoint/size-matched seeds — out of",
        "scope for this minimal protocol; see the Medium-only sections in `finetune_comparison.md`",
        "for that fuller study).",
        "",
        f"Holm-Bonferroni family = exactly these **{len(capacity_rows)} official-split FT-vs-HF",
        "tests** (one per size) — deliberately kept separate from the headline cross-model pairwise",
        "family in `statistics_pairwise_transcript_clean.csv` (that file is NOT regenerated by this",
        "study; tiny/small joining the chart-model set would shift every existing pair's Holm p-value,",
        "so regeneration is deferred to the README/paper-update phase — see the findings report) and",
        "separate from Medium's own disjoint/size-matched seed families above.",
        "",
        "| Size | Params | Pretrained (openai) | HF baseline | Fine-tuned | Δ (paired, speaker-clustered) | 95% CI | p | p (Holm) | n clips | n speakers |",
        "|------|:------:|:--------------------:|:-----------:|:----------:|:-----------------------------:|:------:|:-:|:--------:|:-------:|:----------:|",
    ]
    csv_rows = []
    for r, ph in zip(capacity_rows, p_holm):
        h = r["headline"]
        pretrained_str = f"{h['pretrained_openai_wer']:.2f}%" if h["pretrained_openai_wer"] is not None else "N/A"
        cap_lines.append(
            f"| {r['display_name']} | {r['params']} | {pretrained_str} | {h['hf_baseline_wer']:.2f}% | "
            f"{h['ft_wer']:.2f}% | {h['delta_pp']:+.2f} pp | [{h['ci_lo_pp']:+.2f}, {h['ci_hi_pp']:+.2f}] | "
            f"{h['p']:.3f} | {ph:.3f} | {h['n_clips']} | {h['n_speakers']} |"
        )
        csv_rows.append(dict(
            size=r["key"], display_name=r["display_name"], params=r["params"],
            pretrained_openai_wer=h["pretrained_openai_wer"], hf_baseline_wer=h["hf_baseline_wer"],
            ft_wer=h["ft_wer"], delta_pp=h["delta_pp"], ci_lo_pp=h["ci_lo_pp"], ci_hi_pp=h["ci_hi_pp"],
            p=h["p"], p_holm=ph, n_clips=h["n_clips"], n_speakers=h["n_speakers"],
        ))
    cap_lines.append("")

    # --------------- Pretrained capacity curve (tiny -> large), for context ---------------
    # analyze() is write-free (only analysis/statistics.py's main() writes files) and its
    # `per_model` entries are UNCONDITIONAL per-model bootstrap CIs — not the pairwise/Holm
    # family — so pulling them here does not touch or imply anything about the deferred
    # statistics_pairwise_*.csv regeneration flagged above.
    res = analyze(DATASET, PRIMARY_MODE)
    if res is not None:
        curve_keys = [m for m in ("tiny", "base", "small", "medium", "large") if m in res["models"]]
        if curve_keys:
            by_model = {r["model"]: r for r in res["per_model"]}
            cap_lines += [
                "## Pretrained capacity curve (for context; not a fine-tuning statistic)",
                "",
                f"Speaker-clustered bootstrap CIs from `analysis/statistics.py:analyze()` "
                f"(N={res['N']} clips, G={res['G']} {res['cluster_unit']}s, B=2000). Point estimates only "
                "— no Holm correction applied or needed here (these are per-model CIs, not pairwise tests).",
                "",
                "| Model | Params | Corpus WER | 95% CI |",
                "|-------|:------:|:----------:|:------:|",
            ]
            for m in sorted(curve_keys, key=lambda k: MODEL_ORDER.index(k)):
                row = by_model[m]
                cap_lines.append(
                    f"| {row['display']} | {MODEL_BY_KEY[m].params} | {row['corpus_wer_pct']:.2f}% | "
                    f"[{row['ci_lo_pct']:.2f}, {row['ci_hi_pct']:.2f}] |"
                )
            cap_lines.append("")

    cap_md_path = os.path.join(ANALYSIS_DIR, "finetune_capacity_summary.md")
    with open(cap_md_path, "w") as fh:
        fh.write("\n".join(cap_lines))
    pd.DataFrame(csv_rows).to_csv(os.path.join(ANALYSIS_DIR, "finetune_capacity_summary.csv"), index=False)
    print(f"\n  Saved capacity summary: {cap_md_path}")
else:
    print("\n  [SKIP] capacity summary — no size has both a HF baseline and a fine-tuned result yet")

print("\nDone.")
