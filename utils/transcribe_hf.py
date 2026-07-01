"""HF-transformers transcription with chunked long-form decoding.

Analogue of utils/transcribe.py (which uses the openai-whisper package). Used for the
fine-tuned Whisper Medium model and the same-engine pretrained baseline (medium_hf).

We use transformers' ASR pipeline with chunk_length_s=30 so clips longer than Whisper's
30s receptive field are windowed automatically — matching how openai-whisper's
model.transcribe() handles long audio. Without this, generate() would silently truncate
anything past 30s.
"""

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

from utils.io_helpers import decode_audio_value

CHUNK_LENGTH_S = 30
# Leave stride at the pipeline default (chunk_length_s / 6 on each side) — symmetric
# left+right context overlap is what gives correct word boundaries when stitching chunks.


def build_asr_pipeline(model_path: str, device: str = None):
    """Load a Whisper model + processor and wrap them in a chunked ASR pipeline.

    model_path may be a local fine-tuned directory or an HF hub id (e.g. openai/whisper-medium).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Whisper model from: {model_path}  (device: {device})")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    # Inference: cache speeds up generation; remove any forced decoder ids so language/task
    # are supplied via generate_kwargs instead (avoids the forced_decoder_ids conflict).
    model.config.use_cache = True
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = model.to(device).to(torch_dtype)

    # pipeline wants an int device index (0 for cuda:0, -1 for cpu).
    pipe_device = 0 if device == "cuda" else -1

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=CHUNK_LENGTH_S,
        torch_dtype=torch_dtype,
        device=pipe_device,
    )
    print("Model loaded.\n")
    return pipe


def transcribe_sample_hf(pipe, sample: dict, audio_value: dict) -> str:
    """Transcribe one HF dataset sample with the chunked pipeline.

    audio_value is the raw arrow "audio" struct for this row (utils.io_helpers.
    raw_audio_column(ds)[idx].as_py()), passed explicitly rather than read from
    sample["audio"] — the caller strips "audio" from sample so plain dataset iteration
    never triggers datasets.Audio's decode (which needs torchcodec).

    Returns the raw (unnormalized) transcription string. Mirrors the error-handling
    behaviour of utils.transcribe.transcribe_sample.
    """
    try:
        # Inside the try: some rows have no embedded array, only a stale local path from
        # the original dataset upload (e.g. "E:\\TIE_shorts\\...") that isn't reachable
        # here, so decode_audio_value's soundfile fallback can raise for those rows too.
        audio_array, sr = decode_audio_value(audio_value)
        result = pipe(
            {"raw": audio_array, "sampling_rate": sr},
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
        return result["text"].strip()
    except Exception as e:
        sample_id = sample.get("ID", "?")
        print(f"  [WARN] Failed to transcribe {sample_id}: {e}")
        return ""
