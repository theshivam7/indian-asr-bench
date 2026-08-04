"""Build the README's headline figure: the three-corpus leaderboard, side by side.

One panel per dataset, models on a shared row order, corpus WER with its 95%
cluster-bootstrap confidence interval. The point of showing all three together is the
paper's point: the same nine systems reorder between corpora, and the error bars show
when a reorder is inside the noise. TIE's panel in particular has the leader and the
runner-up overlapping, which is what makes its verdicts normalizer-fragile.

Colour comes from utils.registry, the same colourblind-safe (Okabe-Ito) mapping the
paper figures use, so a model is the same colour everywhere it appears.

Reads results/<dataset>/analysis/statistics_<mode>.csv, so the figure cannot disagree
with the committed numbers.

Usage:
    python analysis/make_overview_figure.py

Writes results/benchmark_overview.png (tracked, embedded in README.md) and
.svg (untracked). The PNG is byte-reproducible across runs; the SVG is not, because
matplotlib derives its clip-path ids per run, so an identical rerun yields a 300-line
diff of renamed ids. Versioning that would bury real changes, so the SVG is generated
for local use and gitignored.
"""

import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import analysis_dir, results_dir
from utils.registry import MODEL_COLOR, MODEL_ORDER, PRIMARY_MODE, get_dataset

DATASETS = ("tie", "svarah", "aesrc")

mpl.rcParams.update({
    "savefig.dpi": 200, "figure.dpi": 150,
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.alpha": 0.30, "grid.linewidth": 0.5,
})


def main(mode: str = PRIMARY_MODE) -> None:
    panels = []
    for ds in DATASETS:
        p = os.path.join(analysis_dir(ds), f"statistics_{mode}.csv")
        if os.path.exists(p):
            panels.append((ds, pd.read_csv(p)))
    if not panels:
        print("[overview] no statistics_*.csv found; run analysis/statistics.py first")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 3.5))
    for ax, (ds, df) in zip([axes] if len(panels) == 1 else axes, panels):
        # Sort by WER so each panel reads best-to-worst top-down; the reordering
        # between panels is the thing worth seeing.
        df = df.sort_values("corpus_wer_pct", ascending=False).reset_index(drop=True)
        y = range(len(df))
        for yi, r in zip(y, df.itertuples()):
            c = MODEL_COLOR.get(r.model, "#666666")
            ax.plot([r.ci_lo_pct, r.ci_hi_pct], [yi, yi], color=c, linewidth=2.2,
                    solid_capstyle="round", alpha=0.55, zorder=2)
            ax.plot(r.corpus_wer_pct, yi, "o", ms=6.5, color=c,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=3)
            ax.annotate(f"{r.corpus_wer_pct:.2f}", (r.ci_hi_pct, yi), xytext=(5, 0),
                        textcoords="offset points", fontsize=7.5, va="center", color=c)
        ax.set_yticks(list(y))
        ax.set_yticklabels(df["display"], fontsize=8)
        ax.set_xlabel("Corpus WER (%)")
        unit = df["cluster_unit"].iloc[0]
        ax.set_title(f"{get_dataset(ds).display}\n"
                     f"{df['n_clips'].iloc[0]:,} clips, {df['n_clusters'].iloc[0]} {unit} clusters",
                     fontsize=9)
        ax.set_xlim(0, df["ci_hi_pct"].max() * 1.20)
        ax.grid(axis="x", alpha=0.30); ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
        ax.tick_params(axis="both", length=3, width=0.6)

    fig.suptitle("Corpus WER with 95% cluster-bootstrap confidence intervals "
                 f"({mode})", fontsize=10.5, y=1.02)
    fig.tight_layout(w_pad=1.6)
    base = os.path.normpath(os.path.join(results_dir("tie"), "..", "benchmark_overview"))
    for ext in ("png", "svg"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[overview] wrote {base}.png / .svg")


if __name__ == "__main__":
    main()
