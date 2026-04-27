"""
Stage 1: ASR Transcription — Whisper Large.

Saves to results/stage1_raw_transcripts/wer_large_raw.csv.
DO NOT re-run unless you need new transcriptions.
Run normalize_and_score.py for WER evaluation.
"""

import os
import sys
import warnings

import pandas as pd
import torch
import whisper
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.transcribe import transcribe_sample
from utils.io_helpers import load_dataset_test, results_dir, stage1_raw_dir, save_checkpoint, remove_checkpoint

warnings.filterwarnings("ignore")

MODEL_NAME = "large"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading whisper-{MODEL_NAME} on device: {device} ...")
model = whisper.load_model(MODEL_NAME, device=device)
print("Model loaded.\n")

transcribe_kw = {"language": "en"}
if device == "cpu":
    transcribe_kw["fp16"] = False

ds = load_dataset_test()

checkpoint_path = os.path.join(results_dir(), f"wer_{MODEL_NAME}_partial.csv")
completed_ids: set[str] = set()
checkpoint_rows: list[dict] = []

if os.path.exists(checkpoint_path):
    df_partial = pd.read_csv(checkpoint_path)
    completed_ids = set(df_partial["ID"].astype(str).tolist())
    checkpoint_rows = df_partial.to_dict("records")
    print(f"  Resuming from checkpoint: {len(completed_ids)} samples already done\n")

all_rows: list[dict] = []

print(f"--- Processing test split ({len(ds)} samples) ---")

for sample in tqdm(ds, desc="test (transcribing)"):
    transcript = (sample.get("Transcript") or "").strip()
    if not transcript:
        continue

    sample_id = sample.get("ID", "")

    hyp_raw = None
    if str(sample_id) in completed_ids:
        ckpt_row = next((r for r in checkpoint_rows if str(r["ID"]) == str(sample_id)), None)
        if ckpt_row is not None:
            hyp_raw = str(ckpt_row.get("hypothesis_raw") or "")

    if hyp_raw is None:
        hyp_raw = transcribe_sample(model, sample, transcribe_kw)

    row = {
        "split": "test",
        "ID": sample_id,
        "Speaker_ID": sample.get("Speaker_ID", ""),
        "Gender": sample.get("Gender", ""),
        "Speech_Class": sample.get("Speech_Class", ""),
        "Native_Region": sample.get("Native_Region", ""),
        "Speech_Duration_seconds": sample.get("Speech_Duration_seconds") or "",
        "Discipline_Group": sample.get("Discipline_Group", ""),
        "Topic": sample.get("Topic", ""),
        "transcript_raw": transcript,
        "normalised_transcript_raw": str(sample.get("Normalised_Transcript") or "").strip(),
        "hypothesis_raw": hyp_raw,
    }

    all_rows.append(row)
    checkpoint_rows.append(row)

    if len(all_rows) % 200 == 0:
        save_checkpoint(checkpoint_rows, MODEL_NAME)
        print(f"  [checkpoint] {len(all_rows)} samples saved")

out_path = os.path.join(stage1_raw_dir(), f"wer_{MODEL_NAME}_raw.csv")
pd.DataFrame(all_rows).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(all_rows)} samples)")
print("Run 'python normalize_and_score.py' for WER evaluation.")

remove_checkpoint(MODEL_NAME)
print("\nDone.")
