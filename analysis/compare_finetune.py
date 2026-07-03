"""
Dedicated comparison: Whisper Medium — pretrained vs fine-tuned.

Headline   : medium_ft  vs  medium_hf   (pretrained Whisper Medium through the SAME HF chunked
             pipeline). Same engine, same decoding => the delta is the true fine-tuning gain.
Secondary  : medium_ft  vs  medium      (the original openai-whisper number, for continuity).

Reads results/tie/stage2_processed/{mode}/wer_{medium,medium_hf,medium_ft}_{mode}.csv
(produced by normalize_and_score.py --dataset tie). Writes:
    results/tie/analysis/finetune_comparison.md
    results/tie/analysis/finetune_comparison.png

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
from utils.registry import ALL_MODES as MODES, PRIMARY_MODE
from utils.io_helpers import stage2_dir, analysis_dir
from analysis.statistics import _clip_errors, _holm

# The fine-tuning study is TIE-only (Svarah is eval-only, not fine-tunable).
DATASET = "tie"
BASELINE = "medium_hf"       # same-engine pretrained baseline (headline)
SECONDARY = "medium"         # original openai-whisper number (continuity)
FINETUNED = "medium_ft"
# Speaker-disjoint re-split FT, one entry per training seed. A null result from a
# single seed is not credible (FT seed variance ~ the effect size being denied),
# so the study runs 3 seeds and reports the spread + a paired bootstrap CI.
DISJOINT_SEEDS = {
    "medium_ft_disjoint": 42,
    "medium_ft_disjoint_s43": 43,
    "medium_ft_disjoint_s44": 44,
}
# Size-matched (speaker-OVERLAPPING) control runs: same clip count as the disjoint
# train set, sampled at random from the full train split. Separates the effect of
# the 12x training-set shrinkage from the effect of speaker-disjointness itself.
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

DISPLAY = {
    "medium": "Whisper Medium (openai-whisper)",
    "medium_hf": "Whisper Medium (pretrained, HF)",
    "medium_ft": "Whisper Medium (fine-tuned)",
    "medium_ft_disjoint": "Whisper Medium (fine-tuned, speaker-disjoint)",
}

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


print("=" * 70)
print("PRETRAINED vs FINE-TUNED — Whisper Medium")
print("=" * 70)

lines = [
    "# Whisper Medium — Pretrained vs Fine-tuned",
    "",
    "Fine-tuned on the `raianand/TIE_shorts` **train** split (7884 clips, clips >30s filtered),",
    "best checkpoint selected on the **validation** split, evaluated on the **test** split (986 clips) —",
    "the same test set used for every pretrained model in this benchmark.",
    "",
    "**Headline comparison** is against `medium_hf` — the *pretrained* Whisper Medium run through the",
    "**same** HuggingFace chunked pipeline as the fine-tuned model. This isolates the fine-tuning gain",
    "from any decoding/engine differences. The original `openai-whisper` number is shown as a secondary",
    "reference.",
    "",
]

# --------------- 1. Corpus WER across all 4 modes ---------------
print("\n--- Corpus WER by mode ---")
ALL_FT_MODELS = (SECONDARY, BASELINE, FINETUNED, *DISJOINT_SEEDS, *SIZEMATCH_SEEDS)
have = {m: {} for m in ALL_FT_MODELS}
for mode in MODES:
    for model in ALL_FT_MODELS:
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
    base = have[BASELINE].get(mode)
    ft = have[FINETUNED].get(mode)
    sec = have[SECONDARY].get(mode)
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
if PRIMARY_MODE in have[BASELINE] and PRIMARY_MODE in have[FINETUNED]:
    b = have[BASELINE][PRIMARY_MODE]
    f = have[FINETUNED][PRIMARY_MODE]
    d_abs, d_rel = fmt_delta(b, f)
    verdict = "improves" if f < b else "does NOT improve"
    lines += [
        f"> **Headline ({PRIMARY_MODE})**: fine-tuning {verdict} WER "
        f"{b:.2f}% → {f:.2f}%  ({d_abs}, {d_rel} relative).",
        "",
    ]

# --------------- 1b. Speaker-disjoint re-split fine-tune (multi-seed; hardens the null) ---------------
disjoint_present = [m for m in DISJOINT_SEEDS if have[m].get(PRIMARY_MODE) is not None]
if disjoint_present:
    dt = DISJOINT_TRAIN
    lines += [
        "## Speaker-disjoint re-split fine-tune (multi-seed)",
        "",
        "Same recipe as the headline fine-tune, but every train clip whose speaker also "
        "appears in `test` is removed first (see `speaker_overlap.md`). Evaluated on the "
        f"SAME test set as `{BASELINE}`, so any gain here cannot come from speaker "
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
        f"alone; see the size-matched control section (when run) for the separation.",
        "",
        f"| Seed | WER (`{PRIMARY_MODE}`) | Δ vs pretrained (paired, speaker-resampled) | 95% CI | p | p (Holm) |",
        "|------|:----:|:----:|:----:|:----:|:----:|",
    ]
    df_base_pm = load(BASELINE, PRIMARY_MODE)
    diffs, halfwidths, pvals = [], [], []
    seed_rows = []
    for m in disjoint_present:
        seed = DISJOINT_SEEDS[m]
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

    wers = [have[m][PRIMARY_MODE] for m in disjoint_present]
    b = have[BASELINE][PRIMARY_MODE]
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
            f"set is separated by the size-matched control below (if run).",
            "",
        ]

    # ---- Size-matched speaker-overlapping control (separates size from disjointness) ----
    sm_present = [m for m in SIZEMATCH_SEEDS if have[m].get(PRIMARY_MODE) is not None]
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
            sm_rows.append((SIZEMATCH_SEEDS[m], have[m][PRIMARY_MODE], d, lo, hi, p))
        sm_holm = _holm(sm_pvals) if len(sm_pvals) > 1 else sm_pvals
        for (seed, wer_m, d, lo, hi, p), ph in zip(sm_rows, sm_holm):
            sig = "" if lo <= 0 <= hi else " *"
            lines.append(f"| {seed} | {wer_m:.2f}% | {d:+.2f} pp | [{lo:+.2f}, {hi:+.2f}]{sig} | {p:.3f} | {ph:.3f} |")
            print(f"  [sizematch s{seed}] wer={wer_m:.2f}%  diff={d:+.2f}pp  CI=[{lo:+.2f},{hi:+.2f}]  p={p:.3f}  p_holm={ph:.3f}")
        lines.append("")
    else:
        lines += [
            "_Size-matched control runs (`medium_ft_sizematch_s*`) not yet available — "
            "submit `hpc/job_finetune_sizematch.pbs` (SEED=42/43/44) to separate the "
            "training-set-size effect from the speaker-disjointness effect._",
            "",
        ]
else:
    print("  [SKIP] no medium_ft_disjoint results yet (run job_finetune_disjoint.pbs first)")

# --------------- 2. Breakdowns (primary mode) ---------------
df_base = load(BASELINE, PRIMARY_MODE)
df_ft = load(FINETUNED, PRIMARY_MODE)

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
    modes_present = [m for m in MODES if m in have[BASELINE] and m in have[FINETUNED]]
    x = range(len(modes_present))
    w = 0.38
    base_vals = [have[BASELINE][m] for m in modes_present]
    ft_vals = [have[FINETUNED][m] for m in modes_present]
    b1 = ax.bar([i - w/2 for i in x], base_vals, w, label="Pretrained (HF)", color="#888")
    b2 = ax.bar([i + w/2 for i in x], ft_vals, w, label="Fine-tuned", color="#2a7")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes_present, rotation=15)
    ax.set_ylabel("Corpus WER (%)")
    ax.set_title("Whisper Medium: Pretrained vs Fine-tuned (TIE_shorts test)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    chart_path = os.path.join(ANALYSIS_DIR, "finetune_comparison.png")
    fig.savefig(chart_path)
    plt.close(fig)
    print(f"\n  Saved chart: {chart_path}")

    # --------------- 4b. WER distribution overlay (pretrained vs fine-tuned) ---------------
    # Same idea as the professor's slide: % of utterances vs per-sample WER in 5% bins.
    # A leftward shift (more mass near 0%) = fine-tuning helps. WER>100% clipped to last bin.
    dist_bins = [i * 5 for i in range(21)]  # 0, 5, ..., 100
    fig, ax = plt.subplots(figsize=(9, 5))
    for d, lab, col in [(df_base, "Pretrained (HF)", "#888888"), (df_ft, "Fine-tuned", "#2a9d4a")]:
        w = (d["wer"].dropna().clip(upper=1.0) * 100).values
        if not len(w):
            continue
        ax.hist(w, bins=dist_bins, weights=[100.0 / len(w)] * len(w),
                alpha=0.6, label=lab, color=col, edgecolor="white")
        ax.axvline(float(pd.Series(w).median()), color=col, linestyle="--", linewidth=1.2)
    ax.set_xlabel("WER (%) — bin width 5%")
    ax.set_ylabel("% of utterances")
    ax.set_title(f"WER distribution: pretrained vs fine-tuned ({PRIMARY_MODE})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    dist_path = os.path.join(ANALYSIS_DIR, "finetune_wer_distribution.png")
    fig.savefig(dist_path)
    plt.close(fig)
    print(f"  Saved chart: {dist_path}")

# --------------- 5. Caveats ---------------
lines += [
    "## Caveats",
    "",
    "- **Engine**: the headline compares fine-tuned vs *pretrained-via-HF* (`medium_hf`), both decoded",
    "  with the same chunked `transformers` pipeline, so the engine is held constant. The original",
    "  `openai-whisper` number is shown only as a continuity reference.",
    "- **Speaker overlap**: see `speaker_overlap.md`. If test speakers also appear in train, part of the",
    "  gain reflects speaker adaptation (disclosed, per the dataset's official splits).",
    "- **Disjoint train-set size**: the speaker-disjoint runs train on 567 clips (3.8 h) vs the official",
    "  split's 7200 (46.9 h, after the same duration/text filters) — speaker-disjointness and training-set",
    "  size are confounded on this dataset by construction. The size-matched control isolates the size effect.",
    "",
]

report_path = os.path.join(ANALYSIS_DIR, "finetune_comparison.md")
with open(report_path, "w") as f:
    f.write("\n".join(lines))
print(f"  Saved report: {report_path}")
print("\nDone.")
