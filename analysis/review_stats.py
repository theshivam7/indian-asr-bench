"""Recompute the human-review statistics quoted in SUMMARY.md from a review sheet.

For each reviewed corpus: mean WER against the original reference and against the
corrected one, the paired drop with a seeded bootstrap CI, a two-sided Wilcoxon
signed-rank test on the paired drop, the same per model with Holm correction, and
a count of the error labels the reviewer settled on.

The Wilcoxon p-value uses the normal approximation with tie correction (n is 28 to
60, well inside its range), so it needs no scipy. Zero differences are dropped, the
standard convention.

Usage:
    python analysis/review_stats.py --dataset tie
Writes results/<dataset>/analysis/human_review_stats.md
"""

import argparse
import collections
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import analysis_dir  # noqa: E402
from analysis.statistics import _holm as holm  # noqa: E402
from analysis.review_common import REVIEW_FOLDERS as FOLDERS, REVIEW_MODELS as MODELS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(FOLDERS))
    args = ap.parse_args()

    sheet = os.path.join(HERE, FOLDERS[args.dataset], "review_sheet.csv")
    df = pd.read_csv(sheet)
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

    labels = collections.Counter()
    for value in df["error_type"].fillna(""):
        for tag in (t.strip() for t in str(value).split(",")):
            if tag:
                labels[tag] += 1
    decisions = collections.Counter(
        str(v).split(" - ")[0] for v in df["reviewer_decision"].fillna("(none)")
    )

    lines = [
        f"# Human review statistics: {args.dataset.upper()}, {len(df)} clips",
        "",
        f"Source: `analysis/{FOLDERS[args.dataset]}/review_sheet.csv`. Bootstrap B={B}, seed {SEED}.",
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

    lines += ["", "| Reviewer verdict | Clips |", "|---|:---:|"]
    for name, count in decisions.most_common():
        lines.append(f"| {name} | {count} |")
    lines += ["", "| Error label | Clips |", "|---|:---:|"]
    for name, count in labels.most_common():
        lines.append(f"| {name} | {count} |")

    out = os.path.join(analysis_dir(args.dataset), "human_review_stats.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
