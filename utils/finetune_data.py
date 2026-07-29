"""Preprocessing + data collator for Whisper fine-tuning (HF transformers).

Standard HuggingFace "Fine-Tune Whisper" recipe:
    - prepare_dataset: audio array -> log-mel input_features, transcript -> label ids
    - DataCollatorSpeechSeq2SeqWithPadding: pad features + labels, mask pad tokens with -100
    - filter_finetune_split / filter_tie_split: clip-usability filtering per dataset
    - seed_everything: one call that seeds every RNG a training run touches

Used by finetune/finetune_medium.py and finetune/finetune_tiny_small.py. Kept here so
the training scripts stay thin, matching the project's "logic lives in utils/" structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from utils.io_helpers import decode_audio_value, raw_audio_column as _raw_audio_column
from utils.normalize import strip_wrapping_quotes

MAX_AUDIO_SECONDS = 30


def seed_everything(seed: int) -> None:
    """Seed every RNG a fine-tuning run touches, before the model is built.

    `Seq2SeqTrainingArguments(seed=...)` only takes effect when the Trainer is
    constructed, which is after `from_pretrained()` and after split filtering, so
    anything random before that point stays uncontrolled unless it is seeded here.
    Training entrypoints call this first, and still pass seed= and data_seed= to the
    training arguments (the Trainer reseeds at its own start, and data_seed is what
    actually drives the sampler's shuffle order).

    This deliberately does NOT set torch.backends.cudnn.deterministic: run-to-run
    kernel nondeterminism is part of the variance the multi-seed study is meant to
    measure, and forcing determinism would both understate that variance and change
    the recipe relative to the single-seed runs already reported.

    torch/numpy are imported inside the function to keep this module importable
    without the GPU stack, matching the TYPE_CHECKING guard above.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    # Validity check at the pyarrow level rather than .as_py(): materializing every row's
    # audio into Python objects just to discard them would roughly double this pass's cost.
    # TIE stores decoded arrays ({"array", ...}); bytes-stored datasets (AESRC) store
    # {"bytes", "path"} - check whichever field the storage actually has.
    field_names = {f.name for f in raw_col.type}
    field = "array" if "array" in field_names else "bytes"

    def _check(_transcript, idx):
        return raw_col[idx][field].is_valid

    return _check


# AESRC's mirror stores every clip as fixed-format WAV (16 kHz, 16-bit, mono, 78-byte
# header; verified corpus-wide in docs/AESRC2020_INDIAN_ANALYSIS.md), so exact duration
# is (byte_length - header) / byte_rate, with no decoding.
_WAV_HEADER_BYTES = 78
_WAV_BYTES_PER_SECOND = 32000


def _filter_by_bytes_duration(ds, max_seconds: float, label: str = ""):
    """Drop clips longer than max_seconds using byte-derived WAV durations.

    Used when a dataset has no duration column but stores fixed-format WAV bytes.
    The format assumption is verified against one decoded clip; a mismatch raises
    rather than silently mis-filtering. `ds` must already be flatten_indices()-
    materialized (raw arrow access by row index).
    """
    import pyarrow.compute as pc

    col = _raw_audio_column(ds).combine_chunks()
    lengths = pc.fill_null(pc.binary_length(col.field("bytes")), 0).to_pylist()
    seconds = [(n - _WAV_HEADER_BYTES) / _WAV_BYTES_PER_SECOND for n in lengths]

    probe_idx = next((i for i, n in enumerate(lengths) if n > 0), None)
    if probe_idx is not None:
        samples, sr = decode_audio_value(col[probe_idx].as_py())
        actual = len(samples) / sr
        if abs(actual - seconds[probe_idx]) > 0.05:
            raise ValueError(
                f"byte-derived duration ({seconds[probe_idx]:.2f}s) disagrees with decoded "
                f"duration ({actual:.2f}s) for clip {probe_idx}; the fixed-WAV-format "
                f"assumption does not hold for this dataset.")

    keep = [i for i, s in enumerate(seconds) if s <= max_seconds]
    if len(keep) < len(ds):
        print(f"  {label}: {len(ds)} -> {len(keep)} after dropping >{max_seconds}s clips "
              f"(byte-derived durations)")
        ds = ds.select(keep).flatten_indices()
    return ds


def _filter_core(ds, transcript_col: str, duration_col: str | None,
                 max_seconds: float, label: str, bytes_duration: bool = False):
    """Shared clip-usability filtering for fine-tuning splits.

    Order matters (fixed 2026-07-08): text/duration filtering must run BEFORE the
    no-embedded-audio filter, and BOTH must run before any downstream random subset
    selection (e.g. max-train-samples caps), otherwise a sampled clip lacking audio
    silently shrinks the realized subset below its nominal size.

    Returns a flatten_indices()-materialized dataset, ready for raw arrow audio access
    (filter() applies a lazy indices overlay; raw arrow access needs physical ordering).
    """
    n_before = len(ds)
    if duration_col:
        ds = ds.filter(
            lambda transcript, dur: has_usable_text(transcript) and within_duration(dur, max_seconds),
            input_columns=[transcript_col, duration_col],
        )
    else:
        ds = ds.filter(has_usable_text, input_columns=[transcript_col])
    print(f"  {label}: {n_before} -> {len(ds)} after dropping empty / >{max_seconds}s clips")
    ds = ds.flatten_indices()

    if duration_col is None and bytes_duration:
        ds = _filter_by_bytes_duration(ds, max_seconds, label=label)

    n_before = len(ds)
    ds = ds.filter(
        has_audio_array(_raw_audio_column(ds)), input_columns=[transcript_col], with_indices=True,
    )
    print(f"  {label}: {n_before} -> {len(ds)} after dropping clips with no embedded audio")
    return ds.flatten_indices()


def filter_finetune_split(ds, spec, max_seconds: float = MAX_AUDIO_SECONDS, label: str = ""):
    """Spec-driven clip-usability filters for any registry dataset.

    Transcript and duration columns come from the DatasetSpec. Datasets without a
    duration column but with bytes-stored audio (AESRC) get byte-derived durations,
    so a >30s clip can never pair feature-extractor-truncated audio with its full
    transcript.
    """
    return _filter_core(ds, spec.gold_ref_col, spec.duration_col, max_seconds, label,
                        bytes_duration=spec.audio_undecoded)


def filter_tie_split(ds, has_duration_col: bool = True,
                      duration_col: str = "Speech_Duration_seconds",
                      transcript_col: str = "Transcript",
                      max_seconds: float = MAX_AUDIO_SECONDS, label: str = ""):
    """TIE-specific wrapper around _filter_core (kept for finetune_medium.py).

    Some TIE_shorts rows have no embedded audio array, only a stale local path from
    the original dataset upload; those are dropped rather than crashing the run.
    TIE's validation split has no duration column, hence has_duration_col.
    """
    return _filter_core(ds, transcript_col, duration_col if has_duration_col else None,
                        max_seconds, label)


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
