"""Score the completed validation annotation sheet against the classifier.

See PROTOCOL.md. Joins annotation_sheet.csv (labels filled in) with
validation_manifest.csv and reports:

  * precision of the consensus artifact flag (stratum A), Wilson 95% CI
  * per-category precision (clip_over_run -> B, content_mismatch -> C/D)
  * stratum-weighted recall estimate + bootstrap CI (labeled an estimate)
  * Cohen's kappa on the binary artifact decision (if label_2 filled on overlap)

Writes validation_results.md next to the inputs.

Usage:
    python analysis/validation/score_validation.py --dataset tie
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ARTIFACT_LABELS = {"B", "C", "D"}          # human labels meaning "artifact"
CATEGORY_TO_LABELS = {"clip_over_run": {"B"}, "content_mismatch": {"C", "D"}}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round((centre - half) * 100, 1), round((centre + half) * 100, 1)


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    po = (a == b).mean()
    pe = sum(((a == c).mean()) * ((b == c).mean()) for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tie")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    base = os.path.join(os.path.dirname(__file__), args.dataset)
    manifest = pd.read_csv(os.path.join(base, "validation_manifest.csv"))
    sheet = pd.read_csv(os.path.join(base, "annotation_sheet.csv"))
    df = manifest.merge(sheet[["item", "label", "label_2"]], on="item")

    df["label"] = df["label"].astype(str).str.strip().str.upper()
    labeled = df[df["label"].isin({"A", "B", "C", "D", "E"})]
    if labeled.empty:
        sys.exit("[validation] no filled labels found in annotation_sheet.csv")
    unsure = labeled[labeled["label"] == "E"]
    d = labeled[labeled["label"] != "E"].copy()
    d["human_artifact"] = d["label"].isin(ARTIFACT_LABELS)

    lines = [f"# Artifact-classifier validation: {args.dataset}", ""]
    lines.append(f"{len(labeled)} items labeled ({len(unsure)} 'unsure' excluded from metrics).")
    lines.append("")

    # --- Precision (stratum A = every consensus-flagged clip) ---
    a = d[d["stratum"] == "A"]
    tp = int(a["human_artifact"].sum())
    lo, hi = wilson_ci(tp, len(a))
    prec = tp / len(a) * 100 if len(a) else float("nan")
    lines += ["## Precision of the consensus artifact flag", "",
              f"{tp}/{len(a)} flagged clips confirmed as artifacts by the blind annotator: "
              f"**precision {prec:.1f}% (95% Wilson CI {lo}–{hi}%)**.", ""]

    # Per-category precision (exact-category agreement, not just binary)
    lines += ["| Predicted category | n | human-confirmed same category | binary artifact confirmed |",
              "|---|---|---|---|"]
    for cat, labels in CATEGORY_TO_LABELS.items():
        sub = a[a["predicted_category"] == cat]
        same = int(sub["label"].isin(labels).sum())
        binv = int(sub["human_artifact"].sum())
        lines.append(f"| {cat} | {len(sub)} | {same} | {binv} |")
    lines.append("")

    # --- Recall estimate (stratum-weighted false-negative extrapolation) ---
    # Estimated artifacts in the corpus = TP(A) + sum over strata B/C/D of
    # (stratum FN rate x stratum size). Recall = TP(A) / estimate.
    rng = np.random.default_rng(42)
    est_terms, boot_terms = [], []
    for s in ("B", "C", "D"):
        sub = d[d["stratum"] == s]
        if sub.empty:
            continue
        size = int(sub["stratum_size"].iloc[0])
        rate = sub["human_artifact"].mean()
        est_terms.append(rate * size)
        boot = rng.choice(sub["human_artifact"].to_numpy(), size=(args.bootstrap, len(sub)))
        boot_terms.append(boot.mean(axis=1) * size)
    fn_est = sum(est_terms)
    recall = tp / (tp + fn_est) * 100 if (tp + fn_est) else float("nan")
    if boot_terms:
        fn_boot = np.sum(boot_terms, axis=0)
        rec_boot = tp / (tp + fn_boot) * 100
        r_lo, r_hi = np.percentile(rec_boot, [2.5, 97.5])
    else:
        r_lo = r_hi = float("nan")
    lines += ["## Recall estimate (stratum-weighted)", "",
              f"Estimated missed artifacts (false negatives) across strata B/C/D: "
              f"**{fn_est:.1f}** clips -> estimated recall "
              f"**{recall:.1f}% (bootstrap 95% CI {r_lo:.1f}–{r_hi:.1f}%)**. "
              f"This is an extrapolated estimate: stratum D samples "
              f"{len(d[d['stratum']=='D'])} of "
              f"{int(d[d['stratum']=='D']['stratum_size'].iloc[0]) if not d[d['stratum']=='D'].empty else 0} "
              f"clips, so its contribution is wide by construction.", ""]

    # --- Inter-annotator agreement (optional) ---
    df["label_2"] = df["label_2"].astype(str).str.strip().str.upper()
    both = df[(df["overlap"] == "yes") & df["label"].isin({"A", "B", "C", "D"})
              & df["label_2"].isin({"A", "B", "C", "D"})]
    if len(both) >= 10:
        k_bin = cohens_kappa(both["label"].isin(ARTIFACT_LABELS),
                             both["label_2"].isin(ARTIFACT_LABELS))
        k_cat = cohens_kappa(both["label"], both["label_2"])
        lines += ["## Inter-annotator agreement", "",
                  f"{len(both)} double-annotated items: Cohen's κ = **{k_bin:.2f}** "
                  f"(binary artifact), {k_cat:.2f} (4-way category).", ""]
    else:
        lines += ["## Inter-annotator agreement", "",
                  "_No second annotator labels found (label_2 empty), single-annotator audit._", ""]

    out = os.path.join(base, "validation_results.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[validation] wrote {out}")


if __name__ == "__main__":
    main()
