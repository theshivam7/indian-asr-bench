"""Recompute the human-review statistics quoted in SUMMARY.md from review_sheet.csv.

For the 49 reviewed TIE clips: mean WER against the original reference and against
the corrected one, the paired drop with a seeded bootstrap CI, a two-sided Wilcoxon
signed-rank test on the paired drop, and the same per model with Holm correction.

The Wilcoxon p-value uses the normal approximation with tie correction (n = 49, well
inside its range), so it needs no scipy. Zero differences are dropped, the standard
convention.

Usage:
    python analysis/tie_validation/review_stats.py
Writes results/tie/analysis/human_review_stats.md
"""

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.io_helpers import analysis_dir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(HERE, "review_sheet.csv")
MODELS = ("large", "parakeet", "parakeet_ctc", "qwen3", "medium")
B = 10000
SEED = 42


def wilcoxon_two_sided(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    n = d.size
    if n == 0:
        return 1.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_plus = ranks[d > 0].sum()
    mean = n * (n + 1) / 4
    counts = pd.Series(ranks).value_counts().to_numpy()
    tie_term = (counts**3 - counts).sum() / 48
    var = n * (n + 1) * (2 * n + 1) / 24 - tie_term
    z = (w_plus - mean) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, pvals[i] * (len(pvals) - rank))
        adjusted[i] = min(running, 1.0)
    return adjusted


def main() -> None:
    df = pd.read_csv(SHEET)
    before, after = df["avg_wer"].to_numpy(float), df["avg_wer_true"].to_numpy(float)
    drop = before - after
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(drop), size=(B, len(drop)))
    boot = drop[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_all = wilcoxon_two_sided(drop)

    per_model = []
    for m in MODELS:
        d = df[f"wer_{m}"].to_numpy(float) - df[f"wer_{m}_true"].to_numpy(float)
        per_model.append((m, d.mean(), wilcoxon_two_sided(d)))
    p_holm = holm([p for _, _, p in per_model])

    lines = [
        "# Human review statistics: TIE, 49 clips",
        "",
        f"Source: `analysis/tie_validation/review_sheet.csv`. Bootstrap B={B}, seed {SEED}.",
        "Wilcoxon signed-rank, two-sided, normal approximation with tie correction.",
        "",
        f"- Mean WER against the original reference: {before.mean():.1f}%",
        f"- Mean WER against the corrected reference: {after.mean():.1f}%",
        f"- Mean drop: {drop.mean():.1f} pp (95% bootstrap CI {lo:.1f} to {hi:.1f} pp)",
        f"- Wilcoxon p: {p_all:.2e}",
        f"- Clips that improve: {int((drop > 0).sum())} of {len(drop)}",
        "",
        "| Model | Mean drop (pp) | Wilcoxon p | p (Holm) |",
        "|---|:---:|:---:|:---:|",
    ]
    for (m, mean_d, p), ph in zip(per_model, p_holm):
        lines.append(f"| {m} | {mean_d:.1f} | {p:.1e} | {ph:.1e} |")
    out = os.path.join(analysis_dir("tie"), "human_review_stats.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
