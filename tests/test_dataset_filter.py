"""Integration tests for the registry row filter and fine-tune split filtering.

Builds a synthetic bytes-stored dataset shaped exactly like AESRC
(pengyizhou/accented_english: id / audio / transcription / speaker / accent) with
real WAV bytes, then exercises the adapter filter path (utils.datasets.load_split)
and the spec-driven fine-tune filters (utils.finetune_data) end-to-end, including
the row-alignment invariant that raw arrow audio access depends on.

Needs the `datasets` stack (datasets/pyarrow/soundfile); the whole module skips
cleanly in environments that only carry the light analysis dependencies.
"""

import io
import os
import sys
import wave

import pytest

pytest.importorskip("datasets")
pytest.importorskip("soundfile")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datasets as hf_datasets
from datasets import Audio, Dataset, Features, Value

from utils.datasets import load_split
from utils.finetune_data import filter_finetune_split, filter_tie_split
from utils.io_helpers import (
    build_sample_row,
    decode_audio_value,
    probe_audio_duration,
    raw_audio_column,
)
from utils.registry import get_dataset

SR = 16000


def _wav_bytes(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"\x00\x01" * int(SR * seconds))
    return buf.getvalue()


def _aesrc_like(rows: list[dict]) -> Dataset:
    """Dataset with AESRC's exact schema. `audio` entries are wav-bytes dicts or None."""
    features = Features({
        "id": Value("string"),
        "audio": Audio(sampling_rate=SR),
        "transcription": Value("string"),
        "speaker": Value("string"),
        "accent": Value("string"),
    })
    cols = {k: [r[k] for r in rows] for k in features}
    return Dataset.from_dict(cols, features=features)


def _row(rid: str, accent: str, seconds: float | None, text: str = "hello world") -> dict:
    audio = {"bytes": _wav_bytes(seconds), "path": f"{rid}.wav"} if seconds is not None else None
    return {"id": rid, "audio": audio, "transcription": text,
            "speaker": rid.rsplit("-", 1)[0], "accent": accent}


def _decoded_seconds(ds, idx: int) -> float:
    samples, sr = decode_audio_value(raw_audio_column(ds)[idx].as_py())
    return len(samples) / sr


@pytest.fixture
def mock_load_dataset(monkeypatch):
    """Patch datasets.load_dataset (looked up lazily inside load_split) per split."""
    def install(split_data: dict):
        def fake(hf_id, split=None, cache_dir=None, revision=None):
            assert split in split_data, f"unexpected split requested: {split}"
            return split_data[split]
        monkeypatch.setattr(hf_datasets, "load_dataset", fake)
    return install


def test_load_split_filters_to_indian_and_stays_row_aligned(mock_load_dataset):
    # Indian rows interleaved with other accents, each with a distinct duration so
    # a physical/logical row misalignment after filter+flatten cannot go unnoticed.
    rows = [
        _row("A-0001-S1", "AMERICAN", 0.5),
        _row("I-0001-S1", "INDIAN", 1.0),
        _row("B-0001-S1", "BRITISH", 0.5),
        _row("I-0002-S1", "INDIAN", 2.0),
        _row("I-0003-S1", "INDIAN", 3.0),
        _row("K-0001-S1", "KOREAN", 0.5),
    ]
    mock_load_dataset({"test": _aesrc_like(rows)})

    ds, spec = load_split("aesrc", "eval")
    assert spec.key == "aesrc"
    assert ds["id"] == ["I-0001-S1", "I-0002-S1", "I-0003-S1"]
    assert set(ds["accent"]) == {"INDIAN"}
    # The invariant every raw arrow audio consumer depends on: physical row i in
    # ds.data is logical row i.
    for i, expected in enumerate((1.0, 2.0, 3.0)):
        assert _decoded_seconds(ds, i) == pytest.approx(expected, abs=0.01)


def test_load_split_raises_when_filter_matches_nothing(mock_load_dataset):
    rows = [_row("A-0001-S1", "AMERICAN", 0.5), _row("B-0001-S1", "BRITISH", 0.5)]
    mock_load_dataset({"test": _aesrc_like(rows)})
    with pytest.raises(ValueError, match="matched 0 rows"):
        load_split("aesrc", "eval")


def test_filter_finetune_split_drops_unusable_clips(mock_load_dataset):
    rows = [
        _row("I-0001-S1", "INDIAN", 2.0),
        _row("I-0002-S1", "INDIAN", 1.5, text="   "),   # empty transcript
        _row("I-0003-S1", "INDIAN", 31.0),               # >30s, byte-derived duration
        _row("I-0004-S1", "INDIAN", None),               # no embedded audio
        _row("I-0005-S1", "INDIAN", 1.0),
    ]
    mock_load_dataset({"train": _aesrc_like(rows)})
    ds, spec = load_split("aesrc", "train")

    out = filter_finetune_split(ds, spec, label="train")
    assert out["id"] == ["I-0001-S1", "I-0005-S1"]
    for i, expected in enumerate((2.0, 1.0)):
        assert _decoded_seconds(out, i) == pytest.approx(expected, abs=0.01)


def test_filter_tie_split_array_storage_unchanged():
    # TIE-shaped rows: audio stored as raw arrays (not Audio-typed bytes), with a
    # duration column. Exercises has_audio_array's "array" field branch.
    n = SR  # 1s of samples
    ds = Dataset.from_dict({
        "Transcript": ["good clip", "", "too long", "no audio", "another good clip"],
        "Speech_Duration_seconds": [5.0, 5.0, 45.0, 5.0, 8.0],
        "audio": [
            {"array": [0.0] * n, "sampling_rate": SR, "path": ""},
            {"array": [0.0] * n, "sampling_rate": SR, "path": ""},
            {"array": [0.0] * n, "sampling_rate": SR, "path": ""},
            None,
            {"array": [0.0] * n, "sampling_rate": SR, "path": ""},
        ],
    })
    out = filter_tie_split(ds, has_duration_col=True, label="train")
    assert out["Transcript"] == ["good clip", "another good clip"]


def test_probe_audio_duration():
    assert probe_audio_duration({"bytes": _wav_bytes(2.0), "path": "x.wav"}) == pytest.approx(2.0, abs=0.01)
    assert probe_audio_duration({"array": [0.0] * SR, "sampling_rate": SR}) == pytest.approx(1.0)
    assert probe_audio_duration({"bytes": None, "path": None}) is None
    assert probe_audio_duration(None) is None


@pytest.mark.parametrize("arity", [1, 2])
def test_run_transcription_end_to_end(mock_load_dataset, monkeypatch, tmp_path, arity):
    """Stage-1 loop on the synthetic AESRC dataset: filter, derived durations,
    canonical CSV schema, and manifest timing, for both callback arities."""
    import pandas as pd

    from utils import io_helpers
    from utils.inference_loop import run_transcription

    rows = [
        _row("A-0001-S1", "AMERICAN", 0.5),
        _row("I-0001-S1", "INDIAN", 1.0),
        _row("I-0002-S1", "INDIAN", 2.0),
    ]
    mock_load_dataset({"test": _aesrc_like(rows)})
    monkeypatch.setattr(io_helpers, "_PROJECT_ROOT", str(tmp_path))

    if arity == 1:
        transcribe_one = lambda sample: "hello world"          # noqa: E731
    else:
        transcribe_one = lambda sample, raw_audio_value: "hello world"  # noqa: E731

    out_path = run_transcription("tiny", "aesrc", transcribe_one=transcribe_one)

    df = pd.read_csv(out_path)
    assert list(df["ID"]) == ["I-0001-S1", "I-0002-S1"]
    assert list(df["Accent"].unique()) == ["INDIAN"]
    assert list(df["Speaker_ID"]) == ["I-0001", "I-0002"]
    # Durations derived from the audio header (no duration column in the spec).
    assert df["Speech_Duration_seconds"].tolist() == pytest.approx([1.0, 2.0], abs=0.01)
    assert (df["hypothesis_raw"] == "hello world").all()

    import json
    manifest = json.loads(open(os.path.join(os.path.dirname(out_path),
                                            "wer_tiny_manifest.json")).read())
    assert manifest["dataset"] == "aesrc"
    assert manifest["clips_transcribed_this_run"] == 2
    assert manifest["audio_seconds_this_run"] == pytest.approx(3.0, abs=0.05)
    assert "seconds_per_audio_second" in manifest


def test_build_sample_row_duration_override():
    spec = get_dataset("aesrc")
    sample = {"id": "I-0001-S1", "transcription": "hello", "speaker": "I-0001", "accent": "INDIAN"}
    row = build_sample_row(sample, "I-0001-S1", "hello", "hyp", spec=spec, split="test",
                           duration=4.46641)
    assert row["Speech_Duration_seconds"] == 4.466   # rounded, not the empty string
    assert row["Speaker_ID"] == "I-0001"
    assert row["Accent"] == "INDIAN"
    # Without an override and no duration column, the field stays empty (old behavior).
    row = build_sample_row(sample, "I-0001-S1", "hello", "hyp", spec=spec, split="test")
    assert row["Speech_Duration_seconds"] == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
