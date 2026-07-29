"""Extract 16 kHz mono WAVs for the clips listed in a human-review CSV.

Must run where the HF dataset cache already exists (NSCC) — TIE_shorts is too
large to download to a laptop. Reads sample_id from --csv, loads the TIE eval
split from the cache (no network needed once cached), writes one WAV per row
into --out-dir, and rewrites the CSV's audio_path column to the WAVs' absolute
paths.

Usage (on NSCC, inside a qsub -I session, whisper_medium_ft env active):
    python extract_review_audio.py \\
        --csv human_review_common_high_wer.csv \\
        --out-dir human_review_audio
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="human_review_audio")
    args = ap.parse_args()

    from utils.datasets import load_eval, extract_ids
    from utils.io_helpers import audio_to_wav_16k

    df = pd.read_csv(args.csv)
    df["sample_id"] = df["sample_id"].astype(str)

    ds, dspec = load_eval("tie")
    pos = {cid: i for i, cid in enumerate(extract_ids(ds, dspec))}

    out_dir = args.out_dir  # kept relative: an absolute path would bake this
    os.makedirs(out_dir, exist_ok=True)  # machine's home directory into the CSV

    audio_paths = []
    missing = 0
    for cid in df["sample_id"]:
        if cid not in pos:
            print(f"  [WARN] clip {cid} not found in the eval split — skipped")
            audio_paths.append("")
            missing += 1
            continue
        wav_path = os.path.join(out_dir, f"{cid}.wav")
        audio_to_wav_16k(ds[pos[cid]][dspec.audio_col], wav_path)
        audio_paths.append(wav_path)

    df["audio_path"] = audio_paths
    df.to_csv(args.csv, index=False)
    print(f"[extract] wrote {len(df) - missing} WAVs to {out_dir}/ ({missing} missing)")
    print(f"[extract] rewrote audio_path in {args.csv}")


if __name__ == "__main__":
    main()
