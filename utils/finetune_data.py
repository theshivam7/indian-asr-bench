"""Preprocessing + data collator for Whisper fine-tuning (HF transformers).

Standard HuggingFace "Fine-Tune Whisper" recipe:
    - prepare_dataset: audio array -> log-mel input_features, Transcript -> label ids
    - DataCollatorSpeechSeq2SeqWithPadding: pad features + labels, mask pad tokens with -100

Used by finetune/finetune_medium.py. Kept here so the training script stays thin,
matching the project's "logic lives in utils/" structure.
"""

from dataclasses import dataclass
from typing import Any

import torch

from utils.io_helpers import decode_audio_value, raw_audio_column as _raw_audio_column
from utils.normalize import strip_wrapping_quotes

MAX_AUDIO_SECONDS = 30


def has_usable_text(transcript) -> bool:
    return bool((transcript or "").strip())


def within_duration(dur, max_seconds: float = MAX_AUDIO_SECONDS) -> bool:
    # Fail closed on missing/unparseable duration: keeping such a clip risks pairing
    # 30s-truncated audio (the feature extractor's hard cap) with its full, untruncated
    # transcript. Fixed 2026-07-08 during the tiny/small capacity-study audit — historical
    # runs (medium's official/disjoint/size-matched fine-tunes) predate this fix and are
    # documented as-run.
    try:
        return dur is not None and float(dur) <= max_seconds
    except (TypeError, ValueError):
        return False


def has_audio_array(raw_col):
    # Check validity at the pyarrow level (raw_col[idx]["array"].is_valid) rather than
    # raw_col[idx].as_py() — .as_py() would fully materialize every row's audio samples
    # into Python objects just to immediately discard them, roughly doubling this pass's
    # cost on top of the identical work .map() does right after for the surviving rows.
    def _check(_transcript, idx):
        return raw_col[idx]["array"].is_valid

    return _check


def filter_tie_split(ds, has_duration_col: bool = True,
                      duration_col: str = "Speech_Duration_seconds",
                      transcript_col: str = "Transcript",
                      max_seconds: float = MAX_AUDIO_SECONDS, label: str = ""):
    """Apply TIE's standard clip-usability filters, in the correct order.

    Order matters (fixed 2026-07-08): text/duration filtering must run BEFORE the
    no-embedded-audio filter, and BOTH must run before any downstream random subset
    selection (speaker-disjoint / size-matched / max-train-samples caps) — otherwise a
    sampled clip lacking audio silently shrinks the realized subset below its nominal
    size (historically observed: disjoint seed 42 realized 566 of a nominal 567 clips).

    Some TIE_shorts rows have no embedded audio array — only a stale local path from the
    original dataset upload (e.g. "E:\\TIE_shorts\\...") that isn't reachable here; those
    are dropped rather than crashing the run.

    Returns a flatten_indices()-materialized dataset, ready for raw arrow audio access.
    Shared by finetune_medium.py and finetune_stepwise.py so both drop exactly the same clips.
    """
    n_before = len(ds)
    if has_duration_col:
        ds = ds.filter(
            lambda transcript, dur: has_usable_text(transcript) and within_duration(dur, max_seconds),
            input_columns=[transcript_col, duration_col],
        )
    else:
        ds = ds.filter(has_usable_text, input_columns=[transcript_col])
    print(f"  {label}: {n_before} -> {len(ds)} after dropping empty / >{max_seconds}s clips")

    # filter() may apply a lazy indices overlay instead of physically reordering the
    # underlying arrow table. flatten_indices() materializes a fresh, physically-ordered
    # table so the raw arrow "audio" access below (by row index) lines up correctly.
    ds = ds.flatten_indices()
    n_before = len(ds)
    ds = ds.filter(
        has_audio_array(_raw_audio_column(ds)), input_columns=[transcript_col], with_indices=True,
    )
    print(f"  {label}: {n_before} -> {len(ds)} after dropping clips with no embedded audio")
    ds = ds.flatten_indices()
    return ds


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
