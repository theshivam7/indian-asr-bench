"""Dataset loading and CSV I/O utilities."""

import os

import pandas as pd
from datasets import load_dataset

HF_CACHE = os.path.join(os.path.expanduser("~"), "hf_cache")
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


def save_mode_csv(
    rows: list[dict],
    model_name: str,
    mode: str,
) -> str:
    """Save per-sample results CSV for a given model and mode.

    Returns the output file path.
    """
    out_dir = os.path.join(results_dir(), mode)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wer_{model_name}_{mode}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def save_checkpoint(rows: list[dict], model_name: str) -> str:
    """Save a partial checkpoint CSV for crash recovery."""
    out_path = os.path.join(results_dir(), f"wer_{model_name}_partial.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def remove_checkpoint(model_name: str) -> None:
    """Remove partial checkpoint CSV after successful completion."""
    out_path = os.path.join(results_dir(), f"wer_{model_name}_partial.csv")
    if os.path.exists(out_path):
        os.unlink(out_path)


SUMMARY_COLUMNS = [
    "model", "mode", "reference_source",
    "corpus_wer", "mean_wer", "median_wer", "std_wer", "p90_wer", "p95_wer",
    "num_samples", "num_empty_hyps", "total_ref_words", "total_errors",
]


def save_summary_csv(summary_rows: list[dict], model_name: str) -> str:
    """Save per-model WER summary stats (one row per mode).

    Each row dict should contain keys from SUMMARY_COLUMNS.
    """
    out_path = os.path.join(results_dir(), f"wer_summary_{model_name}.csv")
    df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    df.to_csv(out_path, index=False)
    return out_path
