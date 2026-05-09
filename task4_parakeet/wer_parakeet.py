"""
Stage 1: ASR Transcription — NVIDIA Parakeet-TDT-0.6B-v2.

Saves to results/stage1_raw_transcripts/wer_parakeet_raw.csv.
DO NOT re-run unless you need new transcriptions.
Run normalize_and_score.py for WER evaluation.
"""

import logging
import os
import sys
import tempfile
import warnings
import wave

import librosa
import nemo.collections.asr as nemo_asr
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Suppress NeMo verbose logging (must be after nemo import so handlers exist)
logging.getLogger("nemo_logger").setLevel(logging.WARNING)
logging.getLogger("nemo").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import (
    load_dataset_test,
    results_dir,
    stage1_raw_dir,
    save_checkpoint,
    remove_checkpoint,
)

MODEL_NAME = "parakeet"
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"


def transcribe_sample_parakeet(model, sample: dict) -> str:
    """Transcribe a single HF dataset sample using Parakeet-TDT.

    Saves audio to a temp WAV file (NeMo expects a file path).
    Returns raw transcription string.
    """
    audio_data = sample["audio"]
    audio_array = np.array(audio_data["array"], dtype=np.float32).flatten()
    sr = audio_data["sampling_rate"]

    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        with wave.open(tmp_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

    try:
        output = model.transcribe([tmp_path])
        # NeMo returns List[str] (return_hypotheses=False, default) or
        # List[Hypothesis] (return_hypotheses=True). Handle both.
        result = output[0]
        text = result.text if hasattr(result, "text") else str(result)
        return text.strip()
    except Exception as e:
        sample_id = sample.get("ID", "?")
        print(f"  [WARN] Failed to transcribe {sample_id}: {e}")
        return ""
    finally:
        os.unlink(tmp_path)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_ID} on device: {device} ...")
model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
if device == "cuda":
    model = model.cuda()
model.eval()
print("Model loaded.\n")

ds = load_dataset_test()

checkpoint_path = os.path.join(results_dir(), f"wer_{MODEL_NAME}_partial.csv")
completed_ids: set[str] = set()
ckpt_map: dict[str, dict] = {}

if os.path.exists(checkpoint_path):
    df_partial = pd.read_csv(checkpoint_path)
    for r in df_partial.to_dict("records"):
        sid = str(r["ID"])
        completed_ids.add(sid)
        ckpt_map[sid] = r
    print(f"  Resuming from checkpoint: {len(completed_ids)} samples already done\n")

all_rows: list[dict] = []

print(f"--- Processing test split ({len(ds)} samples) ---")

for sample in tqdm(ds, desc="test (transcribing)"):
    transcript = (sample.get("Transcript") or "").strip()
    if not transcript:
        continue

    sample_id = str(sample.get("ID", ""))

    if sample_id in completed_ids:
        ckpt_row = ckpt_map.get(sample_id)
        hyp_raw = str(ckpt_row.get("hypothesis_raw") or "") if ckpt_row else ""
    else:
        hyp_raw = transcribe_sample_parakeet(model, sample)

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

    if len(all_rows) % 200 == 0:
        save_checkpoint(all_rows, MODEL_NAME)
        print(f"  [checkpoint] {len(all_rows)} samples saved")

out_path = os.path.join(stage1_raw_dir(), f"wer_{MODEL_NAME}_raw.csv")
pd.DataFrame(all_rows).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(all_rows)} samples)")
print("Run 'python normalize_and_score.py' for WER evaluation.")

remove_checkpoint(MODEL_NAME)
print("\nDone.")
