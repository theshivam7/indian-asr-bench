"""
Stage 1: ASR Transcription — NVIDIA Parakeet-TDT-0.6B-v2.

Saves to results/stage1_raw_transcripts/wer_parakeet_raw.csv.
DO NOT re-run unless you need new transcriptions.
Run normalize_and_score.py for WER evaluation.
"""

import logging
import os
import signal
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
BATCH_SIZE = 16      # NeMo batch transcription size; reduce to 8 if GPU OOM
CHECKPOINT_EVERY = 50


def _audio_to_wav(sample: dict, path: str) -> None:
    audio_data = sample["audio"]
    audio_array = np.array(audio_data["array"], dtype=np.float32).flatten()
    sr = audio_data["sampling_rate"]
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_int16.tobytes())


def transcribe_batch(model, samples: list) -> list:
    """Write temp WAVs, call NeMo batch transcribe, clean up. Returns list of hyp strings."""
    tmp_paths = []
    try:
        for sample in samples:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                _audio_to_wav(sample, tmp.name)
                tmp_paths.append(tmp.name)
        outputs = model.transcribe(tmp_paths, batch_size=len(tmp_paths))
        return [(out.text if hasattr(out, "text") else str(out)).strip() for out in outputs]
    except Exception as e:
        print(f"  [WARN] Batch transcription failed: {e}", flush=True)
        return [""] * len(samples)
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def _make_sigterm_handler(model_name, rows_ref):
    def handler(signum, frame):
        print(f"\n[SIGTERM] Job killed. Saving {len(rows_ref)} rows...", flush=True)
        if rows_ref:
            save_checkpoint(rows_ref, model_name)
            pd.DataFrame(rows_ref).to_csv(
                os.path.join(stage1_raw_dir(), f"wer_{model_name}_interrupted.csv"),
                index=False,
            )
            print("[SIGTERM] Checkpoint + interrupted CSV saved.", flush=True)
        sys.exit(0)
    return handler


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_ID} on device: {device} ...")
model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
if device == "cuda":
    model = model.cuda()
model.eval()
print("Model loaded.\n")

ds = load_dataset_test()

completed_ids: set = set()
ckpt_map: dict = {}

checkpoint_path = os.path.join(results_dir(), f"wer_{MODEL_NAME}_partial.csv")
if os.path.exists(checkpoint_path):
    df_partial = pd.read_csv(checkpoint_path)
    for r in df_partial.to_dict("records"):
        sid = str(r["ID"])
        completed_ids.add(sid)
        ckpt_map[sid] = r
    print(f"  Resuming from checkpoint: {len(completed_ids)} samples already done\n")

all_rows: list = []
signal.signal(signal.SIGTERM, _make_sigterm_handler(MODEL_NAME, all_rows))

print(f"--- Processing test split ({len(ds)} samples) ---")

pending_samples: list = []
pending_meta: list = []  # list of (sample_id, transcript, original_sample)


def _build_row(sample: dict, sample_id: str, transcript: str, hyp_raw: str) -> dict:
    return {
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


def flush_batch() -> None:
    if not pending_samples:
        return
    hyps = transcribe_batch(model, pending_samples)
    for sample, (sample_id, transcript), hyp_raw in zip(pending_samples, pending_meta, hyps):
        all_rows.append(_build_row(sample, sample_id, transcript, hyp_raw))
        if len(all_rows) % CHECKPOINT_EVERY == 0:
            save_checkpoint(all_rows, MODEL_NAME)
            print(f"  [checkpoint] {len(all_rows)} samples saved", flush=True)
    pending_samples.clear()
    pending_meta.clear()


for sample in tqdm(ds, desc="test (transcribing)"):
    transcript = (sample.get("Transcript") or "").strip()
    if not transcript:
        continue

    sample_id = str(sample.get("ID", ""))

    if sample_id in completed_ids:
        flush_batch()
        ckpt_row = ckpt_map.get(sample_id)
        hyp_raw = str(ckpt_row.get("hypothesis_raw") or "") if ckpt_row else ""
        all_rows.append(_build_row(sample, sample_id, transcript, hyp_raw))
    else:
        pending_samples.append(sample)
        pending_meta.append((sample_id, transcript))
        if len(pending_samples) >= BATCH_SIZE:
            flush_batch()

flush_batch()

out_path = os.path.join(stage1_raw_dir(), f"wer_{MODEL_NAME}_raw.csv")
pd.DataFrame(all_rows).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(all_rows)} samples)")
print("Run 'python normalize_and_score.py' for WER evaluation.")

remove_checkpoint(MODEL_NAME)
print("\nDone.")
