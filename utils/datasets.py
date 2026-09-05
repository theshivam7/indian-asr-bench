"""Dataset adapter, the only dataset-specific code in the pipeline.

`load_eval(dataset_key)` (and `load_split`) return the HF split plus its
DatasetSpec, after validating that every column the spec declares actually exists
in the dataset's features. This is what makes adding a dataset a one-line registry
change: everything downstream reads reference/metadata columns *through* the spec.

For any future spec marked `verified=False`, a missing column prints an actionable
warning instead of crashing, so the first real load tells you exactly which names
to fix in the registry. All currently registered schemas are verified.

`datasets` is imported lazily so the CPU-only Stage 2/3 pipeline never needs it.
"""

from utils.registry import get_dataset
from utils.io_helpers import HF_CACHE, text_value


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


def extract_ids(ds, spec) -> list[str]:
    """Per-clip IDs for a whole split, WITHOUT materializing audio bytes.

    Reads the id column straight from arrow storage. If it is a struct column
    (Svarah: id_col == audio_col, storage {"bytes","path"}), pull the "path"
    field columnar-wise, loading ds[id_col] would decode/copy every audio blob.
    Must stay consistent with utils.io_helpers.sample_id (basename of path).
    """
    import os
    import pyarrow as pa

    col = ds.data.column(spec.id_col)
    if pa.types.is_struct(col.type):
        paths = col.combine_chunks().field("path").to_pylist()
        return [os.path.basename(p) if p else "" for p in paths]
    return [text_value(v) for v in col.to_pylist()]


def _validate_data(spec, ds) -> None:
    """Fail-early data checks: run before any GPU time is spent.

    Raises with an actionable message on: empty split, empty IDs, duplicate IDs
    (would corrupt checkpoint-resume and every downstream per-clip join), or a
    first sample whose audio cannot be decoded.
    """
    if len(ds) == 0:
        raise ValueError(f"[datasets] '{spec.key}' split loaded 0 samples, wrong split name or gated access?")

    ids = extract_ids(ds, spec)
    n_empty = sum(1 for i in ids if not i)
    if n_empty:
        raise ValueError(
            f"[datasets] '{spec.key}': {n_empty}/{len(ids)} samples have an EMPTY id "
            f"(id_col='{spec.id_col}'), checkpoint resume and Stage-2/3 joins would silently break."
        )
    if len(set(ids)) != len(ids):
        from collections import Counter
        dupes = [k for k, c in Counter(ids).most_common(5) if c > 1]
        raise ValueError(
            f"[datasets] '{spec.key}': {len(ids) - len(set(ids))} DUPLICATE ids in "
            f"id_col='{spec.id_col}' (e.g. {dupes}), per-clip joins in statistics/error analysis "
            f"would misalign. Pick a unique id column in the DatasetSpec."
        )

    if spec.audio_undecoded:
        from utils.io_helpers import decode_audio_value

        samples, sr = decode_audio_value(ds[0][spec.audio_col])
        if len(samples) == 0 or sr <= 0:
            raise ValueError(f"[datasets] '{spec.key}': probe decode of sample 0 returned no audio "
                             f"(sr={sr}), check soundfile install / audio codec.")
        print(f"  Probe decode OK: sample 0 -> {len(samples)/sr:.2f}s @ {sr}Hz")
    print(f"  Data checks OK: {len(ids)} samples, ids unique + non-empty")


def load_split(dataset_key: str, role: str = "eval"):
    """Load one split of a dataset. Returns (hf_dataset, DatasetSpec)."""
    from datasets import load_dataset

    spec = get_dataset(dataset_key)
    if role not in spec.splits:
        raise ValueError(f"Dataset '{dataset_key}' has no '{role}' split. Available: {spec.splits}")
    split = spec.splits[role]
    rev = f" @ {spec.hf_revision[:12]}" if spec.hf_revision else ""
    print(f"Loading {spec.hf_id} [{split}]{rev} ...  (cache: {HF_CACHE})")
    ds = load_dataset(spec.hf_id, split=split, cache_dir=HF_CACHE, revision=spec.hf_revision)
    print(f"  Loaded {len(ds)} samples")
    print(f"  Features: {ds.column_names}")
    if spec.audio_undecoded:
        # Access the audio column as its raw {"bytes","path"} storage dict. This is a
        # metadata-only cast: datasets' Audio decode machinery (torchcodec-mandatory in
        # datasets>=4) is never invoked; engines decode via io_helpers.decode_audio_value.
        from datasets import Audio

        ds = ds.cast_column(spec.audio_col, Audio(decode=False))
        print(f"  Cast '{spec.audio_col}' -> Audio(decode=False) (torchcodec-free bytes access)")
    ds = _apply_row_filter(spec, ds)
    _validate_schema(spec, ds)
    _validate_data(spec, ds)
    return ds, spec


def _apply_row_filter(spec, ds):
    """Keep only rows where spec.filter_col == spec.filter_value (e.g. AESRC accent).

    input_columns keeps the filter pass from touching the audio column. flatten_indices()
    is required afterwards: filter() applies a lazy indices overlay, and every raw arrow
    audio access (raw_audio_column) indexes ds.data by physical row, which would misalign
    with logical rows under an overlay.
    """
    if not spec.filter_col:
        return ds
    if spec.filter_col not in ds.column_names:
        raise KeyError(f"[datasets] '{spec.key}': filter_col '{spec.filter_col}' not in "
                       f"columns {sorted(ds.column_names)} - fix the DatasetSpec.")
    n_before = len(ds)
    ds = ds.filter(lambda v: v == spec.filter_value,
                   input_columns=[spec.filter_col]).flatten_indices()
    print(f"  Filter {spec.filter_col} == '{spec.filter_value}': {n_before} -> {len(ds)} rows")
    if len(ds) == 0:
        raise ValueError(f"[datasets] '{spec.key}': filter {spec.filter_col} == "
                         f"'{spec.filter_value}' matched 0 rows - wrong filter value?")
    return ds


def load_eval(dataset_key: str):
    """Load the evaluation split (the benchmark test set) for a dataset."""
    return load_split(dataset_key, "eval")
