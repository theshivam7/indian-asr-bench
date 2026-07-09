"""
Fine-tuning capacity study: pretrained vs fine-tuned, per model size.

One report per size (Tiny / Small / Medium), each comparing:
    Headline   : <size>_ft  vs  <size>_hf   (pretrained through the SAME HF chunked
                 pipeline). Same engine, same decoding => the delta is the true FT gain.
    Secondary  : <size>_ft  vs  <size>       (the original openai-whisper number, for continuity).

Each size runs the same minimal protocol: one official-split fine-tune vs its own
HF-pipeline pretrained baseline (see results/tie/analysis/findings_tiny_small_ft.md).

Reads results/tie/stage2_processed/{mode}/wer_{model}_{mode}.csv
(produced by normalize_and_score.py --dataset tie). Writes, per size:
    results/tie/analysis/finetune_comparison[_<size>].md
    results/tie/analysis/finetune_comparison[_<size>].png
    results/tie/analysis/finetune_wer_distribution[_<size>].png
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

plt.rcParams.update({"savefig.dpi": 300, "savefig.facecolor": "white"})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.wer_compute import compute_corpus_wer
from utils.registry import ALL_MODES as MODES, PRIMARY_MODE, MODEL_ORDER, MODEL_BY_KEY
from utils.io_helpers import stage2_dir, analysis_dir
from analysis.statistics import _clip_errors, _holm, analyze

# The fine-tuning study is TIE-only (Svarah is eval-only, not fine-tunable).
DATASET = "tie"

# One entry per model size, each running the same minimal protocol: one official-split
# fine-tune vs its own HF-pipeline pretrained baseline. out_stem="finetune_comparison"
# for medium preserves the historical filename (finetune_comparison.md/.png,
# finetune_wer_distribution.png).
FT_PAIRS = (
    dict(key="tiny", display_name="Whisper Tiny", params="39M",
         secondary="tiny", baseline="tiny_hf", finetuned="tiny_ft",
         out_stem="finetune_comparison_tiny"),
    dict(key="small", display_name="Whisper Small", params="244M",
         secondary="small", baseline="small_hf", finetuned="small_ft",
         out_stem="finetune_comparison_small"),
    dict(key="medium", display_name="Whisper Medium", params="769M",
         secondary="medium", baseline="medium_hf", finetuned="medium_ft",
         out_stem="finetune_comparison"),
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
    all_ft_models = (secondary, baseline, finetuned)
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
        # % of utterances vs per-sample WER in 5% bins.
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
        "",
    ]
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
        "One official-split fine-tune per model size, each compared against its own",
        "HF-pipeline pretrained baseline.",
        "",
        f"Holm-Bonferroni family = exactly these **{len(capacity_rows)} official-split FT-vs-HF",
        "tests** (one per size) — kept separate from the headline cross-model pairwise family in",
        "`statistics_pairwise_transcript_clean.csv` (that family covers PRETRAINED models only;",
        "the fine-tuned variants run through a different decoding engine, so mixing them in would",
        "confound fine-tuning with an engine change — see `analysis/statistics.py`).",
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
