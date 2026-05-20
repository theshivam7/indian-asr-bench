"""
Stage 1: ASR Transcription — Qwen3-ASR-1.7B.

Saves to results/stage1_raw_transcripts/wer_qwen3_raw.csv.
DO NOT re-run unless you need new transcriptions.
Run normalize_and_score.py for WER evaluation.
"""

import os
import signal
import sys
import tempfile
import warnings
import wave

import librosa
import numpy as np
import pandas as pd
import torch
from qwen_asr import Qwen3ASRModel
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on systems where the
# cuDNN version does not match the CUDA runtime.
torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import (
    load_dataset_test,
    results_dir,
    stage1_raw_dir,
    build_sample_row,
    save_checkpoint,
    remove_checkpoint,
)

MODEL_NAME = "qwen3"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
CHECKPOINT_EVERY = 50


def transcribe_sample_qwen3(model, sample: dict) -> str:
    """Transcribe a single HF dataset sample using Qwen3-ASR.

    Saves audio to a temp WAV file and passes the path to the model.
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
        # language="English" for determinism; model auto-detects otherwise
        results = model.transcribe(audio=tmp_path, language="English")
        result = results[0]
        text = result.text if hasattr(result, "text") else str(result)
        return text.strip()
    except Exception as e:
        sample_id = sample.get("ID", "?")
        print(f"  [WARN] Failed to transcribe {sample_id}: {e}", flush=True)
        return ""
    finally:
        os.unlink(tmp_path)


def _make_sigterm_handler(model_name: str, rows_ref: list[dict]):
    """Return a SIGTERM handler that saves checkpoint and interrupted CSV before exit."""
    def handler(signum: int, frame) -> None:
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

# bfloat16 on GPU for efficiency; float32 on CPU for compatibility
if device == "cuda":
    model = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=512,
    )
else:
    model = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        max_new_tokens=512,
    )
print("Model loaded.\n")

ds = load_dataset_test()

completed_ids: set[str] = set()
ckpt_map: dict[str, dict] = {}

checkpoint_path = os.path.join(results_dir(), f"wer_{MODEL_NAME}_partial.csv")
if os.path.exists(checkpoint_path):
    df_partial = pd.read_csv(checkpoint_path)
    for r in df_partial.to_dict("records"):
        sid = str(r["ID"])
        completed_ids.add(sid)
        ckpt_map[sid] = r
    print(f"  Resuming from checkpoint: {len(completed_ids)} samples already done\n")

all_rows: list[dict] = []
signal.signal(signal.SIGTERM, _make_sigterm_handler(MODEL_NAME, all_rows))

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
        hyp_raw = transcribe_sample_qwen3(model, sample)

    all_rows.append(build_sample_row(sample, sample_id, transcript, hyp_raw))

    if len(all_rows) % CHECKPOINT_EVERY == 0:
        save_checkpoint(all_rows, MODEL_NAME)
        print(f"  [checkpoint] {len(all_rows)} samples saved", flush=True)

out_path = os.path.join(stage1_raw_dir(), f"wer_{MODEL_NAME}_raw.csv")
pd.DataFrame(all_rows).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(all_rows)} samples)")
print("Run 'python normalize_and_score.py' for WER evaluation.")

remove_checkpoint(MODEL_NAME)
print("\nDone.")
