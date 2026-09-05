"""Audio extraction and Whisper transcription."""

import os
import tempfile

from utils.io_helpers import audio_to_wav_16k


def transcribe_sample(model, sample: dict, transcribe_kw: dict, audio_col: str = "audio") -> str:
    """Transcribe a single HF dataset sample using a loaded Whisper model.

    Audio is decoded via io_helpers (handles both raw-array and bytes storage,
    independent of datasets' Audio decode machinery). Returns the raw
    (unnormalized) transcription string.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        audio_to_wav_16k(sample[audio_col], tmp_path)
        result = model.transcribe(tmp_path, **transcribe_kw)
        return result["text"].strip()
    except Exception as e:
        raise RuntimeError("Whisper failed to transcribe a sample") from e
    finally:
        os.unlink(tmp_path)
