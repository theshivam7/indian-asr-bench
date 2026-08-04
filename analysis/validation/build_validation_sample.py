"""Build the blind, stratified validation sample for the artifact classifier.

See PROTOCOL.md. Reads the full-corpus consensus table written by
analysis/error_analysis.py plus the per-model Stage-2 tables, draws the four
strata (seeded), and writes into analysis/validation/<dataset>/:

    validation_manifest.csv   item -> clip ID, stratum, predicted category (KEY, do not
                              show to the annotator)
    annotation_sheet.csv      item, audio file, reference, empty label/notes columns
                              (randomized order, blind)
    audio/item_###.wav        16 kHz mono clips (requires the HF dataset cache;
                              run on the cluster login node if not cached locally)

Usage:
    python analysis/validation/build_validation_sample.py --dataset tie
    python analysis/validation/build_validation_sample.py --dataset tie --no-audio
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.registry import PRIMARY_MODE, MODEL_BY_KEY, models_for_dataset, get_dataset
from utils.io_helpers import stage2_dir, analysis_dir
from analysis.error_analysis import classify, ARTIFACT_CATEGORIES

CAP_A = 60   # census cap: all consensus-flagged clips
CAP_B = 30   # census cap: borderline (>=1 individual model flag, no consensus flag)
N_C = 40     # sample: high-WER never-flagged
N_D = 20     # sample: random unflagged
OVERLAP_FRACTION = 0.5  # share of items marked for the optional second annotator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tie")
    ap.add_argument("--mode", default=PRIMARY_MODE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-audio", action="store_true", help="skip WAV extraction")
    args = ap.parse_args()

    spec = get_dataset(args.dataset)
    out_dir = os.path.join(os.path.dirname(__file__), args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cons_path = os.path.join(analysis_dir(args.dataset), f"error_analysis_full_{args.mode}.csv")
    if not os.path.exists(cons_path):
        sys.exit(f"[validation] {cons_path} not found, run analysis/error_analysis.py first.")
    cons = pd.read_csv(cons_path)
    cons["ID"] = cons["ID"].astype(str)

    # Per-model individual flags (for the borderline stratum) + reference text.
    per_model_flagged: set[str] = set()
    references: dict[str, str] = {}
    for m in models_for_dataset(args.dataset):
        if not MODEL_BY_KEY[m].chart:
            continue
        path = os.path.join(stage2_dir(args.dataset), args.mode, f"wer_{m}_{args.mode}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["ID"] = df["ID"].astype(str)
        for cid, rec, rat, ref, ref_n in zip(df["ID"], df["ref_recall"], df["length_ratio"],
                                             df["reference_raw"], df["reference"]):
            n_words = len(ref_n.split()) if isinstance(ref_n, str) else 0
            if classify(rec, rat, n_words) in ARTIFACT_CATEGORIES:
                per_model_flagged.add(cid)
            references.setdefault(cid, str(ref))

    # Stratum A = consensus ARTIFACT flags only; short_ref clips are unclassifiable
    # by the instrument being validated, so they don't belong in its precision sample.
    flagged = cons[cons["category"].isin(ARTIFACT_CATEGORIES)]
    strat_a = flagged["ID"].tolist()[:CAP_A]
    classifiable = cons[cons["category"] != "short_ref"]  # short_ref: outside the instrument's domain
    borderline = sorted((per_model_flagged - set(flagged["ID"])) & set(classifiable["ID"]))
    strat_b = list(rng.permutation(borderline))[:CAP_B]
    rest = classifiable[~classifiable["ID"].isin(set(strat_a) | set(strat_b))].sort_values("wer_mean", ascending=False)
    strat_c = rest["ID"].head(N_C).tolist()
    pool_d = rest["ID"].iloc[N_C:].tolist()
    strat_d = list(rng.choice(pool_d, size=min(N_D, len(pool_d)), replace=False))

    pred = dict(zip(cons["ID"], cons["category"]))
    stratum_sizes = {"A": len(flagged), "B": len(borderline),
                     "C": min(N_C, len(rest)), "D": len(pool_d)}
    rows = []
    for stratum, ids in (("A", strat_a), ("B", strat_b), ("C", strat_c), ("D", strat_d)):
        for cid in ids:
            rows.append({"ID": cid, "stratum": stratum,
                         "stratum_size": stratum_sizes[stratum],
                         "predicted_category": pred.get(cid, "unflagged"),
                         "consensus_flagged": "yes" if stratum == "A" else "no"})
    manifest = pd.DataFrame(rows)

    # Randomize presentation order; assign blind item numbers; mark the κ overlap set.
    manifest = manifest.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    manifest.insert(0, "item", [f"item_{i+1:03d}" for i in range(len(manifest))])
    overlap = rng.random(len(manifest)) < OVERLAP_FRACTION
    manifest["overlap"] = np.where(overlap, "yes", "no")
    manifest.to_csv(os.path.join(out_dir, "validation_manifest.csv"), index=False)

    sheet = pd.DataFrame({
        "item": manifest["item"],
        "audio_file": manifest["item"] + ".wav",
        "reference": [references.get(c, "") for c in manifest["ID"]],
        "overlap": manifest["overlap"],
        "label": "", "notes": "", "label_2": "",
    })
    sheet.to_csv(os.path.join(out_dir, "annotation_sheet.csv"), index=False)
    print(f"[validation] {len(manifest)} items "
          f"(A={len(strat_a)} census-flagged, B={len(strat_b)} borderline, "
          f"C={len(strat_c)} high-WER unflagged, D={len(strat_d)} random)")
    print(f"  manifest (KEY, keep from annotator): {out_dir}/validation_manifest.csv")
    print(f"  blind sheet:                         {out_dir}/annotation_sheet.csv")

    if args.no_audio:
        print("  --no-audio: skipping WAV extraction.")
        return

    from utils.datasets import load_eval, extract_ids
    from utils.io_helpers import audio_to_wav_16k

    ds, dspec = load_eval(args.dataset)
    pos = {cid: i for i, cid in enumerate(extract_ids(ds, dspec))}
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    missing = 0
    for item, cid in zip(manifest["item"], manifest["ID"]):
        if cid not in pos:
            print(f"  [WARN] clip {cid} not found in the eval split, skipped")
            missing += 1
            continue
        audio_to_wav_16k(ds[pos[cid]][dspec.audio_col], os.path.join(audio_dir, f"{item}.wav"))
    print(f"  wrote {len(manifest) - missing} WAVs to {audio_dir}/ ({missing} missing)")


if __name__ == "__main__":
    main()
