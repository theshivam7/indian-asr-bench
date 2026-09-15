# Archived: YouTube Caption Evaluation

This folder contains the YouTube caption experiment conducted as part of the Indian ASR benchmark. It is **not part of the active benchmark** and is excluded from main comparisons. The code is frozen and has not been updated for the `results/<dataset>/` layout.

## Why Archived

1. **Coverage:** Only 190/986 samples (19.3%) have English captions. Results are not representative of the full test set.
2. **Different methodology:** Requires sliding-window Jaccard alignment to locate ~20-second clips within full-video captions (~6,300 words). This is incomparable to direct ASR transcription.
3. **Not an ASR model:** YouTube manual captions are created by human transcribers post-production, not by an ASR engine in the traditional sense.

## Key Results

- Clip-aligned WER: **51.88%** (transcript_clean, n=190)
- Whisper Medium on the same 190 samples: 13.67%, so YouTube is **3.8x worse**
- Low variance (std 8.35%) = consistently poor, not occasionally bad
- Normalization has near-zero impact (<0.2 pp), errors are content/vocabulary mismatches, not formatting

## Contents

| Path | What it is |
|---|---|
| `task_code/fetch_youtube_captions.py` | cluster-side fetcher (YouTube blocks the local IP) |
| `task_code/fetch_youtube_captions_colab.ipynb` | Colab fetcher (use this) |
| `task_code/align_youtube_captions.py` | sliding-window Jaccard alignment |
| `task_code/requirements.txt` | dependencies |
| `results/wer_youtube_raw.csv` | Stage 1: full-video captions (986 rows, 190 manual) |
| `results/wer_youtube_aligned_raw.csv` | Stage 1.5: clip-aligned (190 rows) |
| `results/youtube_alignment_comparison.csv` | full vs aligned WER comparison |
| `results/stage2/` | normalized WER results (4 modes) |

## How to Re-Run

1. Open `task_code/fetch_youtube_captions_colab.ipynb` in Google Colab
2. Run all cells, downloads captions with checkpoint support
3. Download the result to `results/wer_youtube_raw.csv`
4. Run alignment: `python task_code/align_youtube_captions.py`
