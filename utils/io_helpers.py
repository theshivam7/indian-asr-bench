"""Dataset loading and CSV I/O utilities."""

import os

import pandas as pd
from datasets import load_dataset

HF_CACHE = os.environ.get(
    "HF_DATASETS_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
)
os.makedirs(HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)


def load_dataset_test():
    """Load raianand/TIE_shorts test split."""
    print("Loading dataset raianand/TIE_shorts (test split) ...")
    print(f"  Cache directory: {HF_CACHE}")
    ds = load_dataset("raianand/TIE_shorts", split="test", cache_dir=HF_CACHE)
    print(f"  Loaded {len(ds)} samples\n")
    return ds


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
