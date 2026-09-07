"""Audio extraction and Whisper transcription."""

import os
import tempfile
from contextlib import contextmanager

from utils.io_helpers import audio_to_wav_16k


@contextmanager
def temp_wavs(samples, audio_col: str = "audio"):
    """Write each sample's audio to a 16 kHz temp WAV, yield the paths, delete them after.

    Every engine takes a file path, so this is the one place the decode-and-stage
    step lives. Files are removed even when transcription raises.
    """
    paths = []
    try:
        for sample in samples:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                paths.append(tmp.name)
            audio_to_wav_16k(sample[audio_col], paths[-1])
        yield paths
    finally:
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass


def transcribe_sample(model, sample: dict, transcribe_kw: dict, audio_col: str = "audio") -> str:
    """Transcribe a single HF dataset sample using a loaded Whisper model.

    Audio is decoded via io_helpers (handles both raw-array and bytes storage,
    independent of datasets' Audio decode machinery). Returns the raw
    (unnormalized) transcription string.
    """
    try:
        with temp_wavs([sample], audio_col) as (path,):
            result = model.transcribe(path, **transcribe_kw)
            return result["text"].strip()
    except Exception as e:
        raise RuntimeError("Whisper failed to transcribe a sample") from e
