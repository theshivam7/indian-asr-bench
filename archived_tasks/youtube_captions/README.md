# Archived: YouTube Caption Evaluation

This folder contains the YouTube caption experiment conducted as part of the Indian ASR benchmark. It is **not part of the active benchmark** and is excluded from main comparisons.

## Why Archived

1. **Coverage:** Only 190/986 samples (19.3%) have English captions. Results are not representative of the full test set.
2. **Different methodology:** Requires sliding-window Jaccard alignment to locate ~20-second clips within full-video captions (~6,300 words). This is incomparable to direct ASR transcription.
3. **Not an ASR model:** YouTube manual captions are created by human transcribers post-production, not by an ASR engine in the traditional sense.

## Key Results

- Clip-aligned WER: **51.88%** (transcript_clean, n=190)
- Whisper Medium on same 190 samples: 13.67% → YouTube is **3.8× worse**
- Low variance (std 8.35%) = consistently poor, not occasionally bad
- Normalization has near-zero impact (<0.2 pp), errors are content/vocabulary mismatches, not formatting

## Contents

```
task_code/                          ← original task scripts
  fetch_youtube_captions.py         ← NSCC fetcher (IP-blocked locally)
  fetch_youtube_captions_colab.ipynb  ← Colab fetcher (use this)
  align_youtube_captions.py         ← sliding-window Jaccard alignment
  requirements.txt

results/
  wer_youtube_raw.csv               ← Stage 1: full-video captions (986 rows, 190 manual)
  wer_youtube_aligned_raw.csv       ← Stage 1.5: clip-aligned (190 rows)
  youtube_alignment_comparison.csv  ← full vs aligned WER comparison
  stage2/                           ← normalized WER results (4 modes)
```

## How to Re-Run

1. Open `task_code/fetch_youtube_captions_colab.ipynb` in Google Colab
2. Run all cells, downloads captions with checkpoint support
3. Download result → `results/wer_youtube_raw.csv`
4. Run alignment: `python task_code/align_youtube_captions.py`
