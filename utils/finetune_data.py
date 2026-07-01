"""Preprocessing + data collator for Whisper fine-tuning (HF transformers).

Standard HuggingFace "Fine-Tune Whisper" recipe:
    - prepare_dataset: audio array -> log-mel input_features, Transcript -> label ids
    - DataCollatorSpeechSeq2SeqWithPadding: pad features + labels, mask pad tokens with -100

Used by task6_whisper_medium_ft/finetune.py. Kept here so the training script stays thin,
matching the project's "logic lives in utils/" structure.
"""

from dataclasses import dataclass
from typing import Any

import torch

from utils.io_helpers import decode_audio_value
from utils.normalize import strip_wrapping_quotes


def make_prepare_dataset(processor, raw_audio_column):
    """Return a `.map(..., input_columns=["Transcript"], with_indices=True)` function.

    raw_audio_column is the dataset's raw arrow "audio" ChunkedArray (utils.io_helpers.
    raw_audio_column). We pull each row's audio by index directly from arrow storage —
    bypassing datasets' Audio feature/decode entirely (datasets>=4.0 mandates torchcodec
    for that, a fragile torch/ffmpeg ABI dependency) — then resample to 16 kHz.
    input_columns=["Transcript"] keeps .map() from formatting the "audio" column at all
    while building each row, same trick already used for the .filter() calls above.
    """
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def prepare(transcript: str, idx: int) -> dict:
        audio_value = raw_audio_column[idx].as_py()
        audio_array, sr = decode_audio_value(audio_value, target_sr=16000)
        input_features = feature_extractor(
            audio_array, sampling_rate=sr
        ).input_features[0]

        # Targets come ONLY from the gold `Transcript` column (never Normalised_Transcript).
        # Many rows wrap the whole sentence in double quotes; strip just the leading/trailing
        # pair so the model doesn't learn to emit them. Interior quotes are left untouched.
        labels = tokenizer(strip_wrapping_quotes(transcript)).input_ids
        return {"input_features": input_features, "labels": labels}

    return prepare


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad a batch of {input_features, labels} for Whisper seq2seq training.

    Pads input_features (fixed-length log-mels) and label sequences separately, then
    replaces label padding with -100 so the loss ignores it. Strips a leading BOS token
    that the tokenizer prepends (the model adds it again at generation time).
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so it is ignored by the loss.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If BOS was prepended by the tokenizer, drop it — the model prepends it itself.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
