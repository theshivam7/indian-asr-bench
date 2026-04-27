"""
Stage 2: Normalization + WER Computation.

Reads raw transcripts from results/stage1_raw_transcripts/
Applies 4 evaluation modes and computes WER.
Writes results to results/stage2_processed/{mode}/

No GPU needed. Re-run any time to change normalization.

Modes:
    transcript_raw   — Transcript as-is vs Whisper as-is (baseline)
    transcript_clean — Transcript normalized vs Whisper normalized (gold standard)
    hf_raw           — Normalised_Transcript as-is vs Whisper as-is
    hf_clean         — Normalised_Transcript normalized vs Whisper normalized

Each output CSV shows before/after normalization for both reference and hypothesis:
    reference_raw, reference, hypothesis_raw, hypothesis

Usage:
    python normalize_and_score.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils.normalize import MODES, normalize_text, get_reference_source
from utils.wer_compute import compute_sample_wer, compute_corpus_wer

MODELS = ("base", "medium", "large", "youtube")
STAGE1_DIR = os.path.join(os.path.dirname(__file__), "results", "stage1_raw_transcripts")
STAGE2_DIR = os.path.join(os.path.dirname(__file__), "results", "stage2_processed")


def load_raw(model: str) -> pd.DataFrame:
    path = os.path.join(STAGE1_DIR, f"wer_{model}_raw.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} samples for {model}")
    return df


def process(df: pd.DataFrame, model: str, mode: str) -> tuple[list[dict], dict]:
    """Compute WER for one model × mode."""
    is_hf = "hf" in mode
    is_clean = "clean" in mode

    rows = []
    for _, row in df.iterrows():
        # Pick reference source based on mode
        if is_hf:
            ref_raw = str(row.get("normalised_transcript_raw") or "").strip()
        else:
            ref_raw = str(row.get("transcript_raw") or "").strip()

        hyp_raw = str(row.get("hypothesis_raw") or "").strip()

        if not ref_raw:
            continue

        # Apply normalization for *_clean modes
        ref = normalize_text(ref_raw) if is_clean else ref_raw
        hyp = normalize_text(hyp_raw) if is_clean else hyp_raw

        if not ref:
            continue

        wer = compute_sample_wer(ref, hyp)

        rows.append({
            "split": row.get("split", "test"),
            "ID": row.get("ID", ""),
            "Speaker_ID": row.get("Speaker_ID", ""),
            "Gender": row.get("Gender", ""),
            "Speech_Class": row.get("Speech_Class", ""),
            "Native_Region": row.get("Native_Region", ""),
            "Speech_Duration_seconds": row.get("Speech_Duration_seconds", ""),
            "Discipline_Group": row.get("Discipline_Group", ""),
            "Topic": row.get("Topic", ""),
            "model": model,
            "mode": mode,
            "reference_source": get_reference_source(mode),
            "reference_raw": ref_raw,
            "reference": ref,
            "hypothesis_raw": hyp_raw,
            "hypothesis": hyp,
            "wer": round(wer, 4),
        })

    refs = [r["reference"] for r in rows]
    hyps = [r["hypothesis"] for r in rows]
    wers = [r["wer"] for r in rows]
    stats = compute_corpus_wer(refs, hyps, per_sample_wers=wers)
    return rows, stats


def save_csv(rows: list[dict], model: str, mode: str) -> str:
    out_dir = os.path.join(STAGE2_DIR, mode)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wer_{model}_{mode}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def save_top20(rows: list[dict], model: str, mode: str) -> None:
    df = pd.DataFrame(rows).sort_values("wer", ascending=False).head(20)
    df.to_csv(os.path.join(STAGE2_DIR, f"top_20_high_wer_{model}_{mode}.csv"), index=False)


def build_md_table(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(row[c]) if pd.notna(row[c]) else "N/A" for c in cols) + " |"
            for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


# --------------- Main ---------------

print("=" * 70)
print("STAGE 2: Normalization + WER Computation")
print("=" * 70)
print(f"Reading from: {STAGE1_DIR}")
print(f"Writing to:   {STAGE2_DIR}\n")

os.makedirs(STAGE2_DIR, exist_ok=True)

all_summary = []

for model in MODELS:
    print(f"\n--- Model: whisper-{model} ---")
    df_raw = load_raw(model)
    if df_raw is None:
        continue

    for mode in MODES:
        print(f"  [{mode}] ...", end=" ", flush=True)
        rows, stats = process(df_raw, model, mode)
        save_csv(rows, model, mode)
        save_top20(rows, model, mode)
        print(f"corpus_wer={stats['corpus_wer']*100:.2f}%  mean={stats['mean_wer']*100:.2f}%  median={stats['median_wer']*100:.2f}%")

        all_summary.append({
            "model": model,
            "mode": mode,
            "reference_source": get_reference_source(mode),
            "normalized": "yes" if "clean" in mode else "no",
            "corpus_wer_pct": round(stats["corpus_wer"] * 100, 2),
            "mean_wer_pct": round(stats["mean_wer"] * 100, 2),
            "median_wer_pct": round(stats["median_wer"] * 100, 2),
            "std_wer_pct": round(stats["std_wer"] * 100, 2),
            "p90_wer_pct": round(stats["p90_wer"] * 100, 2),
            "p95_wer_pct": round(stats["p95_wer"] * 100, 2),
            "num_samples": stats["num_samples"],
            "total_ref_words": stats["total_ref_words"],
            "total_errors": stats["total_errors"],
        })

# --------------- Save summary ---------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

df_summary = pd.DataFrame(all_summary)
pivot = df_summary.pivot_table(
    index="model", columns="mode", values="corpus_wer_pct", aggfunc="first"
)[list(MODES)]
print(pivot.to_string())

summary_csv = os.path.join(STAGE2_DIR, "wer_summary_all_models.csv")
df_summary.to_csv(summary_csv, index=False)

summary_md = os.path.join(STAGE2_DIR, "wer_summary_all_models.md")
with open(summary_md, "w") as f:
    f.write("# WER Summary — All Models × All Modes\n\n")
    f.write("## Corpus WER (%) Matrix\n\n")
    f.write(build_md_table(pivot.reset_index()) + "\n\n")
    f.write("## Mode Descriptions\n\n")
    f.write("| Mode | Reference | Before norm | After norm | Symmetric? | Purpose |\n")
    f.write("|------|-----------|-------------|------------|------------|---------|\n")
    f.write("| `transcript_raw` | Transcript | as-is | as-is | Yes | Upper bound baseline |\n")
    f.write("| `transcript_clean` | Transcript | Transcript | normalized | Yes | Gold standard (paper primary) |\n")
    f.write("| `hf_raw` | Normalised_Transcript | as-is | as-is | Yes | HuggingFace normalization as-is |\n")
    f.write("| `hf_clean` | Normalised_Transcript | Normalised_Transcript | normalized | Yes | HF + our normalizer |\n\n")
    f.write("## CSV Columns\n\n")
    f.write("Each result CSV contains:\n\n")
    f.write("| Column | Description |\n")
    f.write("|--------|-------------|\n")
    f.write("| `reference_raw` | Reference text **before** normalization |\n")
    f.write("| `reference` | Reference text **after** normalization (used for WER) |\n")
    f.write("| `hypothesis_raw` | Raw Whisper output **before** normalization |\n")
    f.write("| `hypothesis` | Whisper output **after** normalization (used for WER) |\n")
    f.write("| `wer` | Per-sample WER |\n\n")
    f.write("In `*_raw` modes: `reference_raw == reference` and `hypothesis_raw == hypothesis`.\n")

print(f"\nSummary saved: {summary_csv}")
print(f"Markdown saved: {summary_md}")
print("\nDone.")
