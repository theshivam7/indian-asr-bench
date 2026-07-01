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
    """Load raianand/TIE_shorts test split."""
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


def results_dir() -> str:
    """Return the project-level results directory."""
    return os.path.join(os.path.dirname(__file__), "..", "results")


def stage1_raw_dir() -> str:
    """Return the Stage 1 raw transcripts directory (read-only after first run)."""
    d = os.path.join(results_dir(), "stage1_raw_transcripts")
    os.makedirs(d, exist_ok=True)
    return d


def build_sample_row(
    sample: dict,
    sample_id: str,
    transcript: str,
    hyp_raw: str,
) -> dict:
    """Build the standard output row dict shared by all transcription scripts."""
    return {
        "split": "test",
        "ID": sample_id,
        "Speaker_ID": sample.get("Speaker_ID", ""),
        "Gender": sample.get("Gender", ""),
        "Speech_Class": sample.get("Speech_Class", ""),
        "Native_Region": sample.get("Native_Region", ""),
        "Speech_Duration_seconds": sample.get("Speech_Duration_seconds") or "",
        "Discipline_Group": sample.get("Discipline_Group", ""),
        "Topic": sample.get("Topic", ""),
        "transcript_raw": transcript,
        "normalised_transcript_raw": str(sample.get("Normalised_Transcript") or "").strip(),
        "hypothesis_raw": hyp_raw,
    }


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


def save_checkpoint(rows: list[dict], model_name: str) -> str:
    """Save a partial checkpoint CSV for crash recovery."""
    out_dir = results_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wer_{model_name}_partial.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def remove_checkpoint(model_name: str) -> None:
    """Remove partial checkpoint CSV after successful completion."""
    out_path = os.path.join(results_dir(), f"wer_{model_name}_partial.csv")
    if os.path.exists(out_path):
        os.unlink(out_path)
