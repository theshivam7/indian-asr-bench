"""
Extract audio clips for top 20 highest WER samples per model (transcript_clean).

Saves WAV files to: audio_analysis/clips/{model}/
Filename: {rank:02d}_WER{wer_pct:.0f}_{region}_{class}_{id}.wav

Uses HuggingFace streaming — downloads only the 34 needed clips, not full dataset.
Deduplicates: same ID across models saved once per model folder.

Usage:
    python audio_analysis/extract_top20_audio.py
"""

import os
import sys
import wave

import numpy as np
import pandas as pd
import librosa

# Script, not a module: every statement below runs at import time. A repo-wide import
# sweep would therefore start a dataset download or a network fetch on whatever machine
# the sweep runs on, and on a shared login node that risks a fair-share violation.
# Refuse the import rather than doing the work silently.
if __name__ != "__main__":
    raise ImportError(
        "extract_top20_audio.py is a script and must be run, not imported: python archived_tasks/audio_analysis/extract_top20_audio.py"
    )


REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)

MODELS = ("base", "medium", "large")
STAGE2_DIR = os.path.join(REPO_ROOT, "results/stage2_processed")
OUT_BASE = os.path.join(os.path.dirname(__file__), "clips")


def save_wav(audio_array: np.ndarray, sr: int, path: str) -> None:
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        sr = 16000
    audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


# Load all top20 CSVs and collect unique IDs needed
model_tops: dict[str, pd.DataFrame] = {}
all_needed_ids: set[str] = set()

for model in MODELS:
    csv = os.path.join(STAGE2_DIR, f"top_20_high_wer_{model}_transcript_clean.csv")
    if not os.path.exists(csv):
        print(f"[SKIP] {csv} not found")
        continue
    df = pd.read_csv(csv)
    model_tops[model] = df
    all_needed_ids.update(df["ID"].astype(str).tolist())
    out_dir = os.path.join(OUT_BASE, model)
    os.makedirs(out_dir, exist_ok=True)


print(f"Models: {list(model_tops.keys())}")
print(f"Unique IDs to fetch: {len(all_needed_ids)}")

# Print reference sheets before loading dataset
for model, df in model_tops.items():
    print(f"\n{'='*80}")
    print(f"TOP 20 HIGH WER — Whisper {model.upper()} (transcript_clean)")
    print("="*80)
    for i, (_, r) in enumerate(df.iterrows(), 1):
        print(f"\n#{i:2d}  WER={r['wer']*100:.1f}%  {r['Native_Region']}/{r['Speech_Class']}"
              f"  dur={r['Speech_Duration_seconds']:.1f}s  ID={r['ID']}")
        print(f"  REF: {r['reference_raw']}")
        print(f"  HYP: {r['hypothesis_raw']}")

# Load dataset once and extract all needed audio
print(f"\n\nStreaming dataset (downloads only needed clips, not full dataset)...")
import os as _os
from datasets import load_dataset as _load_dataset
_HF_CACHE = _os.path.expanduser("~/hf_cache")
ds = _load_dataset("raianand/TIE_shorts", split="test", streaming=True, cache_dir=_HF_CACHE)

# Build lookup: id → audio data
print(f"Scanning stream for {len(all_needed_ids)} needed IDs (stops early once all found)...")
id_to_audio: dict[str, dict] = {}

for sample in ds:
    sid = str(sample.get("ID", ""))
    if sid in all_needed_ids and sid not in id_to_audio:
        id_to_audio[sid] = {
            "array": np.array(sample["audio"]["array"], dtype=np.float32).flatten(),
            "sampling_rate": sample["audio"]["sampling_rate"],
        }
    if len(id_to_audio) == len(all_needed_ids):
        break

print(f"Found {len(id_to_audio)}/{len(all_needed_ids)} IDs in dataset\n")

# Save WAVs per model
for model, df in model_tops.items():
    out_dir = os.path.join(OUT_BASE, model)
    saved = 0
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        sid = str(r["ID"])
        if sid not in id_to_audio:
            print(f"  [MISSING] #{rank} {sid}")
            continue
        wer_pct = r["wer"] * 100
        fname = f"{rank:02d}_WER{wer_pct:.0f}_{r['Native_Region']}_{r['Speech_Class']}_{sid}.wav"
        out_path = os.path.join(out_dir, fname)
        audio = id_to_audio[sid]
        save_wav(audio["array"].copy(), audio["sampling_rate"], out_path)
        saved += 1

    print(f"Whisper {model}: {saved}/20 clips → {out_dir}/")

print(f"\nDone. All audio in: {OUT_BASE}/")
print("Subfolders: base/ medium/ large/")
