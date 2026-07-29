"""Build a non-blind human review sample of TIE_shorts clips that are hard for
several strong models at once.

Unlike build_validation_sample.py (which blind-tests the artifact classifier),
this shows the annotator every model's hypothesis so they can judge whether a
clip is hard because of the audio/reference or because of the model. Clips are
selected by requiring several of the strongest available models to agree a
clip is hard, which filters out model-specific failure modes.

Usage:
    python analysis/validation/tie/build_common_high_wer_review.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from utils.io_helpers import stage2_dir

MODE = "transcript_clean"
REQUIRED_MODELS = ["large", "parakeet", "parakeet_ctc", "qwen3"]
BONUS_MODEL = "medium"
ALL_MODELS = REQUIRED_MODELS + [BONUS_MODEL]
WER_THRESHOLD = 40.0  # percent
MIN_MODELS_FLAGGED = 3  # of len(REQUIRED_MODELS)

OUT_PATH = os.path.join(os.path.dirname(__file__), "human_review_common_high_wer.csv")


def load_model(model: str) -> pd.DataFrame:
    path = os.path.join(stage2_dir("tie"), MODE, f"wer_{model}_{MODE}.csv")
    df = pd.read_csv(path)
    df["ID"] = df["ID"].astype(str)
    df["wer_pct"] = df["wer"] * 100
    return df


def main() -> None:
    tables = {m: load_model(m) for m in ALL_MODELS}

    base = tables[REQUIRED_MODELS[0]][
        ["ID", "Speaker_ID", "Native_Region", "Discipline_Group", "Topic",
         "Speech_Duration_seconds", "reference_raw"]
    ].copy()
    base = base.rename(columns={
        "ID": "sample_id", "Speaker_ID": "speaker_id", "Native_Region": "native_region",
        "Discipline_Group": "discipline_group", "Topic": "topic",
        "Speech_Duration_seconds": "duration_seconds", "reference_raw": "reference",
    })

    flagged_counts = {}
    for m in ALL_MODELS:
        t = tables[m][["ID", "hypothesis_raw", "wer_pct"]].rename(
            columns={"ID": "sample_id", "hypothesis_raw": f"hyp_{m}", "wer_pct": f"wer_{m}"}
        )
        base = base.merge(t, on="sample_id", how="left")
        flagged_counts[m] = int((tables[m]["wer_pct"] > WER_THRESHOLD).sum())

    base["n_models_flagged"] = sum(
        (base[f"wer_{m}"] > WER_THRESHOLD).astype(int) for m in REQUIRED_MODELS
    )

    common = base[base["n_models_flagged"] >= MIN_MODELS_FLAGGED].copy()
    common = common.sort_values("n_models_flagged", ascending=False)

    common.insert(1, "audio_filename", common["sample_id"] + ".wav")
    common.insert(2, "audio_path", "")  # filled in by extract_review_audio.py on NSCC
    common["reviewer_decision"] = ""
    common["error_category"] = ""
    common["reviewer_notes"] = ""

    ordered_cols = (
        ["sample_id", "audio_filename", "audio_path", "reference"]
        + [f"hyp_{m}" for m in ALL_MODELS]
        + [f"wer_{m}" for m in ALL_MODELS]
        + ["n_models_flagged", "speaker_id", "native_region", "discipline_group",
           "topic", "duration_seconds", "reviewer_decision", "error_category", "reviewer_notes"]
    )
    common = common[ordered_cols]
    common.to_csv(OUT_PATH, index=False)

    print(f"[review] threshold: WER > {WER_THRESHOLD:.0f}% on {MODE}, required models: "
          f"{', '.join(REQUIRED_MODELS)} (+{BONUS_MODEL} as a bonus signal, not required)")
    print("[review] per-model flagged clip counts (WER > threshold):")
    for m in ALL_MODELS:
        req = "required" if m in REQUIRED_MODELS else "bonus"
        print(f"    {m:14s} {flagged_counts[m]:4d}  ({req})")
    for k in (2, 3, 4):
        n = int((base["n_models_flagged"] >= k).sum())
        print(f"[review] clips with >= {k} of {len(REQUIRED_MODELS)} required models flagged: {n}")
    print(f"[review] wrote {len(common)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
