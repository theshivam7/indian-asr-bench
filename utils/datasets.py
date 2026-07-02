"""Dataset adapter — the only dataset-specific code in the pipeline.

`load_eval(dataset_key)` (and `load_split`) return the HF split plus its
DatasetSpec, after validating that every column the spec declares actually exists
in the dataset's features. This is what makes adding a dataset a one-line registry
change: everything downstream reads reference/metadata columns *through* the spec.

For a spec marked `verified=False` (currently Svarah, whose HF schema is gated and
unconfirmed), a missing column prints an actionable warning instead of crashing,
so the first real load on NSCC tells you exactly which names to fix in the registry.

`datasets` is imported lazily so the CPU-only Stage 2/3 pipeline never needs it.
"""

from utils.registry import get_dataset
from utils.io_helpers import HF_CACHE


def _validate_schema(spec, ds) -> None:
    cols = set(ds.column_names)
    expected = [spec.gold_ref_col, spec.id_col, spec.audio_col]
    if spec.alt_ref_col:
        expected.append(spec.alt_ref_col)
    if spec.speaker_col:
        expected.append(spec.speaker_col)
    if spec.duration_col:
        expected.append(spec.duration_col)
    expected += list(spec.metadata_cols.values())

    missing = [c for c in dict.fromkeys(expected) if c not in cols]
    if not missing:
        return

    msg = (
        f"[datasets] dataset '{spec.key}': declared columns not found in HF features: "
        f"{missing}\n           available columns: {sorted(cols)}\n"
        f"           -> fix the column names in the DatasetSpec in utils/registry.py."
    )
    if spec.verified:
        raise KeyError(msg)
    print("WARNING (provisional spec, verified=False):\n  " + msg.replace("\n", "\n  "))
    print("  Once the names match the real schema, set verified=True in the registry.")


def load_split(dataset_key: str, role: str = "eval"):
    """Load one split of a dataset. Returns (hf_dataset, DatasetSpec)."""
    from datasets import load_dataset

    spec = get_dataset(dataset_key)
    if role not in spec.splits:
        raise ValueError(f"Dataset '{dataset_key}' has no '{role}' split. Available: {spec.splits}")
    split = spec.splits[role]
    print(f"Loading {spec.hf_id} [{split}] ...  (cache: {HF_CACHE})")
    ds = load_dataset(spec.hf_id, split=split, cache_dir=HF_CACHE)
    print(f"  Loaded {len(ds)} samples")
    print(f"  Features: {ds.column_names}")
    _validate_schema(spec, ds)
    return ds, spec


def load_eval(dataset_key: str):
    """Load the evaluation split (the benchmark test set) for a dataset."""
    return load_split(dataset_key, "eval")
