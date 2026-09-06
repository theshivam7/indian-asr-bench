"""
Stage 3: cross-model comparison tables + quick-look analysis charts (dataset-aware).

Model list, chart subset, display names, colours, modes and per-dataset subgroup
breakdowns all come from utils.registry, nothing is hard-coded here. Publication
figures live in paper/figures/make_paper_figures.py; these PNGs are for quick
inspection. Grouped-bar charts colour each model by its fixed registry colour, so a
model keeps the same colour in every figure (colour follows the entity, not rank).

Reads  results/<dataset>/stage2_processed/<mode>/wer_<model>_<mode>.csv
Writes results/<dataset>/analysis/ (tables, PNGs, summary_report.md)

Usage:
    python analysis/compare_all.py                 # tie
    python analysis/compare_all.py --dataset svarah
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.wer_compute import compute_corpus_wer, compute_corpus_cer
from utils.registry import (
    PRIMARY_MODE, MODEL_BY_KEY, MODEL_DISPLAY, MODEL_COLOR, MODEL_ORDER,
    models_for_dataset, modes_for_dataset, get_dataset,
)
from utils.io_helpers import stage2_dir, analysis_dir, build_md_table, text_value

DURATION_BINS = [0, 5, 15, 30, 60, float("inf")]
DURATION_LABELS = ["0-5s", "5-15s", "15-30s", "30-60s", "60s+"]


def _corpus_wer(df: pd.DataFrame) -> float:
    return compute_corpus_wer(df["reference"].fillna("").tolist(),
                              df["hypothesis"].fillna("").tolist())["corpus_wer"] * 100


def _corpus_cer(df: pd.DataFrame) -> float:
    return compute_corpus_cer(df["reference"].fillna("").tolist(),
                              df["hypothesis"].fillna("").tolist()) * 100


def load_result_csv(dataset, model, mode):
    path = os.path.join(stage2_dir(dataset), mode, f"wer_{model}_{mode}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def _ordered_chart_models(present):
    """Chart models (chart=True) that have data, in registry order."""
    return [m for m in MODEL_ORDER if m in present and MODEL_BY_KEY[m].chart]


def _validate_headline_panels(all_data: dict, mode: str) -> None:
    """Require every headline model to use the same clip/reference panel."""
    models = _ordered_chart_models({model for model, table_mode in all_data
                                    if table_mode == mode})
    if len(models) < 2:
        return

    identities = {}
    for model in models:
        df = all_data[(model, mode)]
        required = {"ID", "reference", "hypothesis"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{model}/{mode}: missing required columns {missing}")
        ids = df["ID"].map(text_value)
        if (ids == "").any() or ids.duplicated().any():
            raise ValueError(f"{model}/{mode}: empty or duplicate clip IDs")
        refs = df["reference"].map(text_value)
        identities[model] = pd.Series(refs.to_numpy(), index=ids).sort_index()

    baseline = models[0]
    expected = identities[baseline]
    for model in models[1:]:
        actual = identities[model]
        if not actual.index.equals(expected.index):
            missing = len(expected.index.difference(actual.index))
            extra = len(actual.index.difference(expected.index))
            raise ValueError(
                f"{model}/{mode}: headline evaluation panel differs from {baseline} "
                f"(missing IDs={missing}, extra IDs={extra})"
            )
        mismatch = actual.ne(expected)
        if mismatch.any():
            examples = mismatch[mismatch].index[:5].tolist()
            raise ValueError(
                f"{model}/{mode}: normalized references differ from {baseline} for "
                f"{int(mismatch.sum())} clips (e.g. {examples})"
            )


def _grouped_bar(ax, groups, models, value_of, bar_width=None):
    """Grouped bars coloured by the registry entity colour."""
    bw = bar_width or 0.8 / max(len(models), 1)
    x = range(len(groups))
    for i, m in enumerate(models):
        vals = [value_of(m, g) for g in groups]
        offset = (i - (len(models) - 1) / 2) * bw
        ax.bar([xi + offset for xi in x], vals, bw,
               label=MODEL_DISPLAY.get(m, m), color=MODEL_COLOR.get(m))
    ax.set_xticks(list(x))
    ax.grid(axis="y", alpha=0.3)


def main(dataset: str) -> None:
    spec = get_dataset(dataset)
    modes = list(modes_for_dataset(dataset))
    models = [m for m in models_for_dataset(dataset)]
    out_dir = analysis_dir(dataset)

    print("=" * 70)
    print(f"STAGE 3: comparison: {spec.display} ({dataset})")
    print("=" * 70)

    # ---- 1. corpus WER + CER matrix -------------------------------------------
    all_data, summary_rows = {}, []
    for model in models:
        row = {"model": model, "display": MODEL_DISPLAY.get(model, model)}
        for mode in modes:
            df = load_result_csv(dataset, model, mode)
            if df is None:
                row[mode] = None
                continue
            all_data[(model, mode)] = df
            row[mode] = round(_corpus_wer(df), 2)
        pkey = (model, PRIMARY_MODE)
        row["CER_primary"] = round(_corpus_cer(all_data[pkey]), 2) if pkey in all_data else None
        summary_rows.append(row)
    for mode in modes:
        _validate_headline_panels(all_data, mode)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "wer_summary.csv"), index=False)
    print(df_summary.to_string(index=False))

    # Headline charts require the headline mode. A model with only some secondary
    # mode must not appear as a zero-valued (apparently perfect) primary result.
    chart_models = _ordered_chart_models({m for (m, mode) in all_data
                                           if mode == PRIMARY_MODE})
    # A headline model with no primary-mode table drops out of every chart and every
    # downstream table without warning, and the leaderboard still renders as if it
    # were the full panel.
    absent_chart = [m for m in models
                    if MODEL_BY_KEY[m].chart and m not in chart_models]
    if absent_chart:
        raise FileNotFoundError(
            f"[compare_all] {dataset}: no '{PRIMARY_MODE}' table for headline "
            f"model(s) {absent_chart}. They would silently vanish from the "
            f"leaderboard and every chart. Run normalize_and_score.py first."
        )

    # ---- 2. subgroup breakdowns (registry-driven) + duration ------------------
    for col, label in spec.subgroup_dims:
        rows = []
        for model in models:
            df = all_data.get((model, PRIMARY_MODE))
            if df is None or col not in df.columns:
                continue
            for gval, gdf in df.groupby(col):
                rows.append({"model": model, col: gval,
                             "corpus_wer_pct": round(_corpus_wer(gdf), 2), "num_samples": len(gdf)})
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"comparison_by_{col}.csv"), index=False)

    # duration breakdown. Duration is a clip property, not a model property, so pool an
    # ID -> duration map across every model's table first: engines that predate the
    # derived-duration path (the NeMo runners on AESRC) left the column empty, and
    # without the fill their models silently vanish from this table.
    dur_by_id = {}
    for model in models:
        df = all_data.get((model, PRIMARY_MODE))
        if df is None:
            continue
        d = pd.to_numeric(df["Speech_Duration_seconds"], errors="coerce")
        dur_by_id.update({i: v for i, v in zip(df["ID"], d) if pd.notna(v)})
    dur = {}
    for model in models:
        df = all_data.get((model, PRIMARY_MODE))
        if df is None:
            continue
        df = df.copy()
        df["_dur"] = pd.to_numeric(df["Speech_Duration_seconds"], errors="coerce")
        df["_dur"] = df["_dur"].fillna(df["ID"].map(dur_by_id))
        df["_bucket"] = pd.cut(df["_dur"], bins=DURATION_BINS, labels=DURATION_LABELS)
        for bucket, gdf in df.groupby("_bucket", observed=True):
            dur.setdefault(str(bucket), {})[model] = round(_corpus_wer(gdf), 2)
    if dur:
        pd.DataFrame([{"duration_bucket": b, "model": m, "corpus_wer_pct": v}
                      for b, mv in dur.items() for m, v in mv.items()]
                     ).to_csv(os.path.join(out_dir, "comparison_by_duration.csv"), index=False)

    # ---- 3. quick-look charts --------------------------------------------------
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white",
        "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False,
    })

    # ranking: neutral bars, accent on the best model, 95% cluster-bootstrap CI
    # whiskers when statistics.py has already produced them
    ranked = sorted(((m, MODEL_DISPLAY.get(m, m), float(df_summary.loc[df_summary.model == m, PRIMARY_MODE].values[0]))
                     for m in chart_models
                     if pd.notna(df_summary.loc[df_summary.model == m, PRIMARY_MODE].values[0])),
                    key=lambda t: t[2], reverse=True)
    if ranked:
        keys, names, vals = zip(*ranked)
        ci = {}
        stats_path = os.path.join(out_dir, f"statistics_{PRIMARY_MODE}.csv")
        if os.path.exists(stats_path):
            sdf = pd.read_csv(stats_path)
            ci = {r["model"]: (float(r["ci_lo_pct"]), float(r["ci_hi_pct"])) for _, r in sdf.iterrows()}
        fig, ax = plt.subplots(figsize=(9, 0.52 * len(names) + 1.4))
        best = min(vals)
        colors = ["#0072B2" if v == best else "#ADBDCC" for v in vals]
        ax.barh(names, vals, color=colors, height=0.62)
        for y, (k, v) in enumerate(zip(keys, vals)):
            label_x = v
            if k in ci:
                lo, hi = ci[k]
                ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none",
                            ecolor="#33383D", elinewidth=1.3, capsize=3.5)
                label_x = hi
            ax.text(label_x + max(vals) * 0.012, y, f"{v:.2f}%", va="center",
                    fontsize=10.5, fontweight="bold" if v == best else "normal",
                    color="#0072B2" if v == best else "#33383D")
        ax.margins(x=0.14)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ci_note = " with 95% cluster-bootstrap CI" if ci else ""
        ax.set_xlabel(f"Corpus WER (%){ci_note}")
        ax.set_title(f"{spec.display}: corpus WER, {PRIMARY_MODE} mode (lower is better)",
                     loc="left", pad=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "wer_by_model.png"))
        plt.close(fig)

    # by mode
    if chart_models:
        fig, ax = plt.subplots(figsize=(10, 6))
        _grouped_bar(ax, modes, chart_models,
                     lambda m, md: float(df_summary.loc[df_summary.model == m, md].values[0])
                     if pd.notna(df_summary.loc[df_summary.model == m, md].values[0])
                     else float("nan"))
        ax.set_xticklabels(modes, rotation=15)
        ax.set_ylabel("WER (%)")
        ax.legend(fontsize=8)
        ax.set_title(f"WER by model and mode: {spec.display}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "wer_by_model_and_mode.png"))
        plt.close(fig)

    # per-utterance distribution (small multiples)
    dm = [m for m in chart_models if (m, PRIMARY_MODE) in all_data]
    if dm:
        ncols = 2
        nrows = math.ceil(len(dm) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.4 * nrows + 1), sharex=True, sharey=True)
        axes = axes.flatten()
        bins = [i * 5 for i in range(21)]
        for i, m in enumerate(dm):
            ax = axes[i]
            w = (all_data[(m, PRIMARY_MODE)]["wer"].dropna().clip(upper=1.0) * 100).values
            if len(w):
                ax.hist(w, bins=bins, weights=[100.0 / len(w)] * len(w), color=MODEL_COLOR.get(m), edgecolor="white")
                med = float(pd.Series(w).median())
                ax.axvline(med, color="#d62728", linestyle="--", linewidth=1)
                ax.text(med + 2, ax.get_ylim()[1] * 0.85, f"median {med:.0f}%", fontsize=8, color="#d62728")
            ax.set_title(MODEL_DISPLAY.get(m, m), fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        for j in range(len(dm), len(axes)):
            axes[j].axis("off")
        fig.supxlabel("WER (%), bin width 5%")
        fig.supylabel("% of utterances")
        fig.suptitle(f"Per-utterance WER distribution ({PRIMARY_MODE}): {spec.display}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "wer_distribution.png"))
        plt.close(fig)

    # by duration
    if dur and chart_models:
        buckets = [b for b in DURATION_LABELS if b in dur]
        fig, ax = plt.subplots(figsize=(10, 6))
        _grouped_bar(ax, buckets, chart_models,
                     lambda m, b: dur.get(b, {}).get(m, float("nan")))
        ax.set_xticklabels(buckets)
        ax.set_ylabel("WER (%)")
        ax.legend(fontsize=8)
        ax.set_title(f"WER by duration ({PRIMARY_MODE}): {spec.display}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "wer_by_duration.png"))
        plt.close(fig)

    # one chart per subgroup dim
    for col, label in spec.subgroup_dims:
        gd = {}
        for m in models:
            df = all_data.get((m, PRIMARY_MODE))
            if df is None or col not in df.columns:
                continue
            for gval, gdf in df.groupby(col):
                gd.setdefault(str(gval), {})[m] = round(_corpus_wer(gdf), 2)
        if not gd or not chart_models:
            continue
        groups = sorted(gd.keys())
        fig, ax = plt.subplots(figsize=(10, 6))
        _grouped_bar(ax, groups, chart_models,
                     lambda m, g: gd.get(g, {}).get(m, float("nan")))
        ax.set_xticklabels(groups, rotation=15)
        ax.set_ylabel("WER (%)")
        ax.legend(fontsize=8)
        ax.set_title(f"WER by {label} ({PRIMARY_MODE}): {spec.display}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"wer_by_{col}.png"))
        plt.close(fig)

    # ---- 4. summary report + top20 aggregation --------------------------------
    lines = [f"# WER Evaluation Summary: {spec.display}", "",
             "## Corpus WER (%) by model and mode (+ primary-mode CER)", "",
             build_md_table(df_summary), "", "## Best model per mode", ""]
    for mode in modes:
        valid = df_summary[df_summary[mode].notna()]
        if not valid.empty:
            best = valid.loc[valid[mode].idxmin()]
            lines.append(f"- **{mode}**: {best['display']} ({best[mode]:.2f}%)")
    with open(os.path.join(out_dir, "summary_report.md"), "w") as f:
        f.write("\n".join(lines))

    for mode in modes:
        combined = []
        for m in models:
            p = os.path.join(stage2_dir(dataset), f"top_20_high_wer_{m}_{mode}.csv")
            if os.path.exists(p):
                d = pd.read_csv(p)
                if "model" not in d.columns:
                    d.insert(0, "model", m)
                combined.append(d)
        if combined:
            pd.concat(combined, ignore_index=True).to_csv(
                os.path.join(out_dir, f"top_20_high_wer_all_models_{mode}.csv"), index=False)

    print(f"\n[compare_all] {spec.display}: tables + charts written to {out_dir}\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tie")
    main(ap.parse_args().dataset)
