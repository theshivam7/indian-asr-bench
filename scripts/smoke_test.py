"""CPU-only preflight for a dataset: run this BEFORE any GPU job.

Exercises the exact code paths Stage 1 uses — adapter load (including the
Audio(decode=False) cast and fail-early ID/schema validation in utils.datasets),
per-sample ID extraction, and audio decode via decode_audio_value — on a handful
of samples. Exits non-zero with an actionable message on the first failure, so a
misconfigured environment dies in seconds on the login node instead of hours
into a queued GPU job.

Usage:
    python scripts/smoke_test.py --dataset svarah
    python scripts/smoke_test.py --dataset tie --samples 5
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="svarah")
    ap.add_argument("--samples", type=int, default=3, help="number of samples to decode")
    args = ap.parse_args()

    import datasets as hf_datasets
    print(f"[smoke] datasets=={hf_datasets.__version__}")
    try:
        import soundfile as sf
        print(f"[smoke] soundfile=={sf.__version__} (libsndfile {sf.__libsndfile_version__})")
    except ImportError:
        print("[smoke] FATAL: soundfile not installed — required to decode bytes-stored audio.\n"
              "        pip install soundfile==0.13.1")
        return 1

    from utils.datasets import load_eval, extract_ids
    from utils.io_helpers import decode_audio_value, sample_id, build_sample_row

    # load_eval already runs schema validation + ID uniqueness + probe decode.
    ds, spec = load_eval(args.dataset)

    ids = extract_ids(ds, spec)
    print(f"[smoke] {len(ids)} ids, first 3: {ids[:3]}")

    n = min(args.samples, len(ds))
    for i in range(n):
        sample = ds[i]
        sid = sample_id(sample, spec)
        assert sid == ids[i], (
            f"sample_id() and extract_ids() disagree at row {i}: {sid!r} != {ids[i]!r} "
            f"— per-sample and columnar ID paths must be identical."
        )
        samples_arr, sr = decode_audio_value(sample[spec.audio_col], target_sr=16000)
        ref = str(sample.get(spec.gold_ref_col) or "").strip()
        row = build_sample_row(sample, sid, ref, "smoke-test-hypothesis", spec=spec,
                               split=spec.splits["eval"])
        print(f"[smoke] sample {i}: id={sid}  audio={len(samples_arr)/sr:.2f}s@{sr}Hz  "
              f"ref_words={len(ref.split())}  row_cols={len(row)}")
        if len(samples_arr) == 0:
            print(f"[smoke] FATAL: sample {i} decoded to zero-length audio.")
            return 1
        if not ref:
            print(f"[smoke] WARNING: sample {i} has an empty reference (will be skipped in Stage 1).")

    print(f"\n[smoke] OK — dataset '{args.dataset}' is ready for Stage 1.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("\n[smoke] FAILED — fix the above before submitting any GPU job.")
        sys.exit(1)
