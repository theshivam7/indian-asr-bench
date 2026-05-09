"""
Clip-aligned WER for YouTube captions.

Full-video YouTube captions cannot be directly compared to short clip references
(~20s, ~50 words vs full lecture ~6000 words). This script uses a sliding window
with incremental Jaccard similarity to locate the clip within the full hypothesis,
extracts that window, then computes WER identically to Whisper models.

Alignment strategy:
  - Window size = max(ref_len, int(ref_len * WINDOW_MULTIPLIER))
  - Slide word-by-word over hypothesis
  - Score each window using Jaccard word overlap with reference (NOT WER — avoids circularity)
  - Extract best-matching window
  - Compute WER on that window

Output:
  - results/stage1_raw_transcripts/wer_youtube_aligned_raw.csv
    Same schema as other stage1 CSVs. Available samples get aligned hypothesis;
    unavailable samples keep empty hypothesis.
  - Console summary: raw vs aligned WER across all 4 modes.

After running this script, run normalize_and_score.py and compare_all.py to
include youtube_aligned in the full model comparison.

Usage:
    python task4_youtube_captions/align_youtube_captions.py
"""

import os
import sys
from collections import Counter

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.normalize import normalize_text, MODES, get_reference_source
from utils.wer_compute import compute_sample_wer, compute_corpus_wer

STAGE1_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "stage1_raw_transcripts")
WINDOW_MULTIPLIER = 1.5  # aligned window = ref_len * this factor


# ── Sliding-window Jaccard alignment ─────────────────────────────────────────

def align_hypothesis(ref_text: str, hyp_text: str) -> str:
    """Extract the span of hyp_text that best matches ref_text by word overlap.

    Uses O(M) incremental Jaccard — alignment criterion is word set overlap,
    NOT WER, so evaluation and alignment are independent.

    Returns the best-matching substring of hyp_text (as a string).
    If hypothesis is shorter than the window, returns hypothesis unchanged.
    """
    ref_words = ref_text.lower().split()
    hyp_words = hyp_text.lower().split()
    n_ref = len(ref_words)
    n_hyp = len(hyp_words)
    window_size = max(n_ref, int(n_ref * WINDOW_MULTIPLIER))

    if n_hyp <= window_size:
        return hyp_text  # already short enough — return as-is

    ref_counter = Counter(ref_words)

    # Initialize first window
    win_counter = Counter(hyp_words[:window_size])
    intersection = sum(min(ref_counter[w], win_counter[w]) for w in ref_counter)

    best_intersection = intersection
    best_start = 0

    for start in range(1, n_hyp - window_size + 1):
        out_word = hyp_words[start - 1]
        in_word  = hyp_words[start + window_size - 1]

        # Remove out_word from window
        if out_word in ref_counter:
            r = ref_counter[out_word]
            w_before = win_counter[out_word]
            intersection += min(r, w_before - 1) - min(r, w_before)
        win_counter[out_word] -= 1
        if win_counter[out_word] == 0:
            del win_counter[out_word]

        # Add in_word to window
        if in_word in ref_counter:
            r = ref_counter[in_word]
            w_before = win_counter.get(in_word, 0)
            intersection += min(r, w_before + 1) - min(r, w_before)
        win_counter[in_word] += 1

        if intersection > best_intersection:
            best_intersection = intersection
            best_start = start

    # Return original-casing words from hyp_text
    orig_words = hyp_text.split()
    return " ".join(orig_words[best_start : best_start + window_size])


# ── Load stage1 YouTube CSV ───────────────────────────────────────────────────

raw_path = os.path.join(STAGE1_DIR, "wer_youtube_raw.csv")
if not os.path.exists(raw_path):
    print(f"ERROR: {raw_path} not found. Run Colab notebook first.")
    sys.exit(1)

df = pd.read_csv(raw_path)
print(f"Loaded {len(df)} samples from wer_youtube_raw.csv")

available = df["hypothesis_raw"].fillna("").str.strip() != ""
print(f"  Available (have captions): {available.sum()}")
print(f"  Unavailable:               {(~available).sum()}")

# ── Align hypotheses ──────────────────────────────────────────────────────────

print(f"\nAligning {available.sum()} hypotheses (window = ref_len × {WINDOW_MULTIPLIER}) ...")

aligned_hyps = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="aligning"):
    hyp_raw = str(row.get("hypothesis_raw") or "").strip()
    ref_raw = str(row.get("transcript_raw") or "").strip()

    if not hyp_raw or not ref_raw:
        aligned_hyps.append("")
        continue

    aligned = align_hypothesis(ref_raw, hyp_raw)
    aligned_hyps.append(aligned)

df_aligned = df.copy()
df_aligned["hypothesis_raw"] = aligned_hyps

out_path = os.path.join(STAGE1_DIR, "wer_youtube_aligned_raw.csv")
df_aligned.to_csv(out_path, index=False)
print(f"Saved aligned stage1 CSV → {out_path}")


# ── Compute and compare WER: full vs aligned, raw vs clean ───────────────────

print("\n" + "=" * 70)
print("WER COMPARISON: full-video vs clip-aligned hypothesis")
print("(computed on available samples only — 190 with captions)")
print("=" * 70)

rows_full    = df[available].copy()
rows_aligned = df_aligned[available].copy()

comparison = []

for mode in MODES:
    is_hf    = "hf"    in mode
    is_clean = "clean" in mode
    ref_col  = "normalised_transcript_raw" if is_hf else "transcript_raw"

    def build_pairs(source_df):
        refs, hyps, wers = [], [], []
        for _, row in source_df.iterrows():
            ref_raw = str(row.get(ref_col) or "").strip()
            hyp_raw = str(row.get("hypothesis_raw") or "").strip()
            if not ref_raw or not hyp_raw:
                continue
            ref = normalize_text(ref_raw) if is_clean else ref_raw
            hyp = normalize_text(hyp_raw) if is_clean else hyp_raw
            if not ref:
                continue
            refs.append(ref)
            hyps.append(hyp)
            wers.append(compute_sample_wer(ref, hyp))
        return refs, hyps, wers

    refs_f, hyps_f, wers_f = build_pairs(rows_full)
    refs_a, hyps_a, wers_a = build_pairs(rows_aligned)

    stats_f = compute_corpus_wer(refs_f, hyps_f, per_sample_wers=wers_f)
    stats_a = compute_corpus_wer(refs_a, hyps_a, per_sample_wers=wers_a)

    comparison.append({
        "mode":                    mode,
        "full_corpus_wer_pct":     round(stats_f["corpus_wer"] * 100, 2),
        "full_mean_wer_pct":       round(stats_f["mean_wer"]   * 100, 2),
        "full_median_wer_pct":     round(stats_f["median_wer"] * 100, 2),
        "aligned_corpus_wer_pct":  round(stats_a["corpus_wer"] * 100, 2),
        "aligned_mean_wer_pct":    round(stats_a["mean_wer"]   * 100, 2),
        "aligned_median_wer_pct":  round(stats_a["median_wer"] * 100, 2),
        "num_samples":             stats_a["num_samples"],
    })

    print(f"\n  [{mode}]")
    print(f"    Full video  — corpus:{stats_f['corpus_wer']*100:.2f}%  "
          f"mean:{stats_f['mean_wer']*100:.2f}%  median:{stats_f['median_wer']*100:.2f}%")
    print(f"    Clip-aligned — corpus:{stats_a['corpus_wer']*100:.2f}%  "
          f"mean:{stats_a['mean_wer']*100:.2f}%  median:{stats_a['median_wer']*100:.2f}%")

# ── Save comparison CSV ───────────────────────────────────────────────────────

analysis_dir = os.path.join(os.path.dirname(__file__), "..", "results", "analysis")
os.makedirs(analysis_dir, exist_ok=True)
comp_csv = os.path.join(analysis_dir, "youtube_alignment_comparison.csv")
pd.DataFrame(comparison).to_csv(comp_csv, index=False)
print(f"\nComparison saved → {comp_csv}")

print("""
Next steps:
  python normalize_and_score.py      # picks up wer_youtube_aligned_raw.csv
  python analysis/compare_all.py     # includes youtube_aligned in all charts
""")
