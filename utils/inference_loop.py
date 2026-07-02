"""Shared Stage-1 transcription loop — dataset-aware, resumable, SIGTERM-safe.

Removes the per-model script duplication. An engine driver supplies a
``transcribe_one(sample) -> str`` callable (capturing its loaded model + decode
kwargs); this module handles everything dataset-related: loading the eval split
via the adapter, reading the reference/id through the DatasetSpec, checkpoint /
resume, building the canonical raw-CSV row, and writing the output.

Output: results/<dataset>/stage1_raw_transcripts/wer_<model>_raw.csv
"""

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
    save_checkpoint,
    remove_checkpoint,
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


def run_transcription(model_key: str, dataset_key: str, transcribe_one, *, checkpoint_every: int = 200) -> str:
    """Transcribe a dataset's eval split with `transcribe_one` and write the raw CSV."""
    ds, spec = load_eval(dataset_key)
    split = spec.splits["eval"]
    out_dir = stage1_raw_dir(dataset_key)

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
    for sample in tqdm(ds, desc=f"{dataset_key}:{model_key}"):
        transcript = str(sample.get(spec.gold_ref_col) or "").strip()
        if not transcript:
            continue
        sid = str(sample.get(spec.id_col, ""))
        if sid in completed:
            hyp_raw = str(ckpt_map.get(sid, {}).get("hypothesis_raw") or "")
        else:
            hyp_raw = transcribe_one(sample)
        rows.append(build_sample_row(sample, sid, transcript, hyp_raw, spec=spec, split=split))
        if len(rows) % checkpoint_every == 0:
            save_checkpoint(rows, model_key, dataset_key)

    out_path = os.path.join(out_dir, f"wer_{model_key}_raw.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(rows)} samples)")
    print("Run 'python normalize_and_score.py --dataset %s' for scoring." % dataset_key)
    remove_checkpoint(model_key, dataset_key)
    return out_path
