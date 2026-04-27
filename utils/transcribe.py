"""Audio extraction and Whisper transcription."""

import os
import tempfile
import wave

import librosa
import numpy as np


def transcribe_sample(model, sample: dict, transcribe_kw: dict) -> str:
    """Transcribe a single HF dataset sample using a loaded Whisper model.

    Returns the raw (unnormalized) transcription string.
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
        result = model.transcribe(tmp_path, **transcribe_kw)
        return result["text"].strip()
    except Exception as e:
        sample_id = sample.get("ID", "?")
        print(f"  [WARN] Failed to transcribe {sample_id}: {e}")
        return ""
    finally:
        os.unlink(tmp_path)
