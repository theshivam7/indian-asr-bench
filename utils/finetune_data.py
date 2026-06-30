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


def strip_wrapping_quotes(text) -> str:
    """Strip a single leading/trailing pair of double quotes and surrounding whitespace.

    The TIE_shorts `Transcript` field often wraps the sentence in double quotes, e.g.
        "The second component is less than here ..."
    -> The second component is less than here ...
    Only an outer matched pair is removed; quotes inside the sentence are kept as-is.
    """
    s = (text or "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


def make_prepare_dataset(processor):
    """Return a `.map()` function that turns one raw sample into model inputs.

    Expects the audio column already cast to 16 kHz (datasets.Audio(sampling_rate=16000)),
    so no manual resampling is needed here.
    """
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def prepare(batch: dict) -> dict:
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Targets come ONLY from the gold `Transcript` column (never Normalised_Transcript).
        # Many rows wrap the whole sentence in double quotes; strip just the leading/trailing
        # pair so the model doesn't learn to emit them. Interior quotes are left untouched.
        transcript = strip_wrapping_quotes(batch.get("Transcript"))
        batch["labels"] = tokenizer(transcript).input_ids
        return batch

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
