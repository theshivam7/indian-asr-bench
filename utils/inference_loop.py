"""Shared Stage-1 transcription loop — dataset-aware, resumable, SIGTERM-safe.

Removes the per-model script duplication. An engine driver supplies a
``transcribe_one(sample) -> str`` callable (capturing its loaded model + decode
kwargs); this module handles everything dataset-related: loading the eval split
via the adapter, reading the reference/id through the DatasetSpec, checkpoint /
resume, building the canonical raw-CSV row, and writing the output.

Engines that must avoid datasets' Audio-feature decode (e.g. a conda env whose
pinned `datasets` version needs torchcodec for that path) can instead supply a
two-argument ``transcribe_one(sample, raw_audio_value) -> str``. The loop detects
this via the callable's arity: "audio" is removed from the per-row dict and the
matching raw arrow value is passed as the second argument instead. Existing
single-argument callbacks are unaffected either way.

Output: results/<dataset>/stage1_raw_transcripts/wer_<model>_raw.csv
"""

import inspect
import os
import signal
import sys

import pandas as pd
from tqdm import tqdm

from utils.datasets import load_eval
from utils.io_helpers import (
    stage1_raw_dir,
    results_dir,
    build_sample_row,
    sample_id,
    save_checkpoint,
    remove_checkpoint,
    write_run_manifest,
    raw_audio_column,
)


def _install_sigterm_handler(state: dict) -> None:
    """On SIGTERM (PBS walltime kill) dump progress so the run is resumable."""
    def handler(signum, frame):
        if state["rows"]:
            path = os.path.join(results_dir(state["dataset"]), f"wer_{state['model']}_interrupted.csv")
            pd.DataFrame(state["rows"]).to_csv(path, index=False)
            print(f"\n[SIGTERM] dumped {len(state['rows'])} rows to {path}", flush=True)
        sys.exit(143)
    signal.signal(signal.SIGTERM, handler)


def run_transcription(model_key: str, dataset_key: str, transcribe_one, *,
                      checkpoint_every: int = 200, manifest_extra: dict | None = None) -> str:
    """Transcribe a dataset's eval split with `transcribe_one` and write the raw CSV."""
    ds, spec = load_eval(dataset_key)
    split = spec.splits["eval"]
    out_dir = stage1_raw_dir(dataset_key)

    # A 2-arg transcribe_one opts into raw (undecoded) audio: strip "audio" from the row
    # dict so plain iteration never triggers datasets' Audio-feature decode, and pass the
    # matching raw arrow value as the second argument instead (see module docstring).
    callback_takes_raw_audio = len(inspect.signature(transcribe_one).parameters) >= 2
    raw_audio = raw_audio_column(ds) if callback_takes_raw_audio else None
    ds_iter = ds.remove_columns(["audio"]) if callback_takes_raw_audio else ds

    checkpoint_path = os.path.join(results_dir(dataset_key), f"wer_{model_key}_partial.csv")
    completed: set[str] = set()
    ckpt_map: dict[str, dict] = {}
    if os.path.exists(checkpoint_path):
        for r in pd.read_csv(checkpoint_path).to_dict("records"):
            sid = str(r["ID"])
            completed.add(sid)
            ckpt_map[sid] = r
        print(f"  Resuming from checkpoint: {len(completed)} samples already done\n")

    state = {"rows": [], "model": model_key, "dataset": dataset_key}
    _install_sigterm_handler(state)
    rows = state["rows"]

    print(f"--- {spec.display} [{split}] : {len(ds)} samples, model={model_key} ---")
    for idx, sample in enumerate(tqdm(ds_iter, desc=f"{dataset_key}:{model_key}")):
        transcript = str(sample.get(spec.gold_ref_col) or "").strip()
        if not transcript:
            continue
        sid = sample_id(sample, spec)
        if sid in completed:
            hyp_raw = str(ckpt_map.get(sid, {}).get("hypothesis_raw") or "")
        elif callback_takes_raw_audio:
            hyp_raw = transcribe_one(sample, raw_audio[idx].as_py())
        else:
            hyp_raw = transcribe_one(sample)
        rows.append(build_sample_row(sample, sid, transcript, hyp_raw, spec=spec, split=split))
        if len(rows) % checkpoint_every == 0:
            save_checkpoint(rows, model_key, dataset_key)

    out_path = os.path.join(out_dir, f"wer_{model_key}_raw.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    manifest = write_run_manifest(model_key, dataset_key, spec, extra=manifest_extra)
    print(f"\nSaved: {out_path}  ({len(rows)} samples)")
    print(f"Manifest: {manifest}")
    print("Run 'python normalize_and_score.py --dataset %s' for scoring." % dataset_key)
    remove_checkpoint(model_key, dataset_key)
    return out_path
