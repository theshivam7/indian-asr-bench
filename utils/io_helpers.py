"""Dataset loading and CSV I/O utilities."""

import io
import os

import numpy as np
import pandas as pd

# `datasets` is only needed for Stage 1 transcription (loading the audio). It is imported
# lazily inside load_dataset_test() so the CPU-only Stage 2/3 pipeline (normalize_and_score.py,
# analysis/) — which only uses the CSV/markdown helpers below — does not require the heavy
# (GPU-side) `datasets`/`torch` stack just to recompute WER or draw charts.

# Resolve the HF cache from any of the common env vars so manual runs and PBS jobs
# agree. Defaulting to $HOME/.cache fills the (small) HOME quota on HPC clusters, so
# honour HF_DATASETS_CACHE / HF_CACHE / HF_HOME first — point any of them at scratch.
HF_CACHE = (
    os.environ.get("HF_DATASETS_CACHE")
    or os.environ.get("HF_CACHE")
    or (os.path.join(os.environ["HF_HOME"], "datasets") if os.environ.get("HF_HOME") else None)
    or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
)
os.makedirs(HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)


def load_dataset_test():
    """Load raianand/TIE_shorts test split (kept for backward compatibility).

    New code should use utils.datasets.load_eval(dataset_key) which is dataset-aware.
    """
    from datasets import load_dataset

    print("Loading dataset raianand/TIE_shorts (test split) ...")
    print(f"  Cache directory: {HF_CACHE}")
    ds = load_dataset("raianand/TIE_shorts", split="test", cache_dir=HF_CACHE)
    print(f"  Loaded {len(ds)} samples\n")
    return ds


def raw_audio_column(ds):
    """Return the "audio" column's underlying pyarrow ChunkedArray for a dataset.

    Indexing this (``raw_audio_column(ds)[i].as_py()``) returns a plain Python dict read
    straight from arrow storage, with NO involvement of datasets' Audio feature/decode
    machinery — so it works regardless of the exact nested array shape the dataset happens
    to store, and never touches torchcodec. Pair with input_columns=[...] (excluding
    "audio") on .map()/.filter() and row index access so audio is never auto-decoded during
    normal iteration either.
    """
    return ds.data.column("audio")


def decode_audio_value(audio_value: dict, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Return (mono float32 samples, sample_rate) from a raw (undecoded) audio struct.

    raianand/TIE_shorts stores audio pre-decoded as {"array": [...], "sampling_rate": N,
    "path": ...}, so we read that array directly — bypassing datasets>=4.0's Audio feature,
    which mandates torchcodec (a fragile torch/ffmpeg ABI triple) for any decode/encode/cast.
    The {bytes}/{path} branch is a soundfile fallback for datasets stored as encoded files.
    `.reshape(-1)` flattens whatever nesting the stored array actually has (e.g. a (1, N)
    single-channel wrapper), so this doesn't depend on knowing the exact shape in advance.
    Pass target_sr to resample (e.g. 16000 for Whisper); omit it to keep the native rate.
    """
    array = audio_value.get("array")
    if array is not None:
        samples = np.asarray(array, dtype=np.float32).reshape(-1)
        sr = int(audio_value.get("sampling_rate") or target_sr or 16000)
    else:
        # Fallback for encoded-file datasets: decode bytes/path with soundfile (libsndfile,
        # no PyTorch coupling). Imported lazily so lightweight importers of io_helpers don't
        # pull in the audio stack.
        import soundfile as sf

        source = io.BytesIO(audio_value["bytes"]) if audio_value.get("bytes") else audio_value["path"]
        samples, sr = sf.read(source, dtype="float32")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        import librosa  # lazy: only needed when the stored rate differs from target

        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return samples, sr


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def results_dir(dataset: str = "tie") -> str:
    """Return the per-dataset results directory: results/<dataset>."""
    return os.path.join(_PROJECT_ROOT, "results", dataset)


def stage1_raw_dir(dataset: str = "tie") -> str:
    """Stage 1 raw transcripts dir (immutable source of truth, always committed)."""
    d = os.path.join(results_dir(dataset), "stage1_raw_transcripts")
    os.makedirs(d, exist_ok=True)
    return d


def stage2_dir(dataset: str = "tie") -> str:
    """Stage 2 scored-output dir: results/<dataset>/stage2_processed."""
    d = os.path.join(results_dir(dataset), "stage2_processed")
    os.makedirs(d, exist_ok=True)
    return d


def analysis_dir(dataset: str = "tie") -> str:
    """Stage 3 analysis dir: results/<dataset>/analysis."""
    d = os.path.join(results_dir(dataset), "analysis")
    os.makedirs(d, exist_ok=True)
    return d


def sample_id(sample: dict, spec) -> str:
    """Extract a clean string ID for a sample.

    Usually spec.id_col is a plain string column. Some datasets (Svarah) have no
    separate id/filename field, so id_col points at the same HF column as
    audio_col; with audio_undecoded specs that column yields the raw
    {"bytes", "path"} storage dict — take the path's basename (stable across cache
    locations and runs). Must stay consistent with utils.datasets.extract_ids.
    Raises rather than returning an empty ID: a blank ID silently corrupts
    checkpoint-resume and every downstream per-clip join.
    """
    val = sample.get(spec.id_col, "")
    if isinstance(val, dict):
        val = os.path.basename(val.get("path") or "")
    sid = str(val)
    if not sid:
        raise ValueError(f"sample_id: empty id from column '{spec.id_col}' "
                         f"(dataset '{spec.key}') — refusing to emit a blank ID.")
    return sid


def audio_to_wav_16k(audio_value, wav_path: str) -> None:
    """Decode any raw audio value ({array,...} or {bytes,path}) to a 16 kHz mono
    int16 WAV file — the single audio path shared by all Stage-1 engines."""
    import wave

    samples, _ = decode_audio_value(audio_value, target_sr=16000)
    audio_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_int16.tobytes())


def build_sample_row(
    sample: dict,
    sample_id: str,
    transcript: str,
    hyp_raw: str,
    spec=None,
    split: str | None = None,
    alt_ref: str | None = None,
) -> dict:
    """Build the canonical raw-CSV row for any dataset, driven by its DatasetSpec.

    Metadata columns come from ``spec.metadata_cols`` (canonical_name -> HF source
    column); speaker and duration from ``spec.speaker_col`` / ``spec.duration_col``.
    Called with the old 4-arg signature it defaults to the TIE spec and reproduces
    the original TIE schema byte-for-byte (so the untouched task4/5/6 scripts keep
    working). ``alt_ref`` is the alternate reference (TIE Normalised_Transcript);
    if None it is pulled from ``spec.alt_ref_col`` when present.
    """
    from utils.registry import TIE

    if spec is None:
        spec = TIE
    if split is None:
        split = spec.splits.get("eval", "test")
    if alt_ref is None:
        alt_ref = sample.get(spec.alt_ref_col) if spec.alt_ref_col else ""

    row = {
        "split": split,
        "ID": sample_id,
        "Speaker_ID": sample.get(spec.speaker_col, "") if spec.speaker_col else "",
        "Speech_Duration_seconds": (sample.get(spec.duration_col) or "") if spec.duration_col else "",
    }
    for out_name, src_col in spec.metadata_cols.items():
        row[out_name] = sample.get(src_col, "")
    row["transcript_raw"] = transcript
    row["normalised_transcript_raw"] = str(alt_ref or "").strip()
    row["hypothesis_raw"] = hyp_raw
    return row


def build_md_table(df: pd.DataFrame) -> str:
    """Render a pandas DataFrame as a GitHub-Flavored Markdown table."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(row[c]) if pd.notna(row[c]) else "N/A" for c in cols) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def save_checkpoint(rows: list[dict], model_name: str, dataset: str = "tie") -> str:
    """Save a partial checkpoint CSV for crash recovery (transient, gitignored)."""
    out_dir = results_dir(dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wer_{model_name}_partial.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def remove_checkpoint(model_name: str, dataset: str = "tie") -> None:
    """Remove partial checkpoint CSV after successful completion."""
    out_path = os.path.join(results_dir(dataset), f"wer_{model_name}_partial.csv")
    if os.path.exists(out_path):
        os.unlink(out_path)
