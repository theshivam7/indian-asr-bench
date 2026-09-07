"""
Stage 1: ASR transcription, NVIDIA Parakeet (NeMo), batched.

Drives both Parakeet models via the registry:
    --model parakeet       -> Parakeet-TDT-0.6B-v2 (transducer)
    --model parakeet_ctc   -> Parakeet-CTC-1.1B     (ctc; 2nd cannot-hallucinate witness)

Usage:
    python parakeet/wer_parakeet.py --model parakeet_ctc --dataset tie
    python parakeet/wer_parakeet.py --model parakeet     --dataset svarah

Writes results/<dataset>/stage1_raw_transcripts/wer_<model>_raw.csv.
NeMo transcribes in batches, so this keeps its own loop instead of
utils.inference_loop, but mirrors that loop's checkpointing, duration
derivation and manifest timing so the raw CSVs have the same schema.
"""

import argparse
import logging
import os
import signal
import sys
import time
import warnings

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.efficiency import cudnn_disabled, timed
from utils.io_helpers import positive_float, probe_audio_duration, text_value
from utils.transcribe import temp_wavs

BATCH_SIZE = 16
CHECKPOINT_EVERY = 50


def transcribe_batch(model, samples, audio_col):
    try:
        with temp_wavs(samples, audio_col) as paths:
            outputs = model.transcribe(paths, batch_size=len(paths))
            return [(o.text if hasattr(o, "text") else str(o)).strip() for o in outputs]
    except Exception as e:
        raise RuntimeError(
            f"Parakeet batch transcription failed for {len(samples)} clip(s)"
        ) from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="parakeet", choices=["parakeet", "parakeet_ctc"])
    ap.add_argument("--dataset", default="tie")
    args = ap.parse_args()

    logging.getLogger("nemo_logger").setLevel(logging.WARNING)
    logging.getLogger("nemo").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")

    import nemo.collections.asr as nemo_asr
    from utils.registry import MODEL_BY_KEY
    from utils.datasets import load_eval
    from utils.io_helpers import (results_dir, stage1_raw_dir, build_sample_row,
                                  sample_id, save_checkpoint, remove_checkpoint,
                                  write_run_manifest)

    model_key = args.model
    dataset = args.dataset
    model_id = MODEL_BY_KEY[model_key].model_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} ({MODEL_BY_KEY[model_key].display}) on {device} ...")
    load_timing: list[float] = []
    with cudnn_disabled(), timed(load_timing):
        model = nemo_asr.models.ASRModel.from_pretrained(model_id)
        if device == "cuda":
            model = model.cuda()
        model.eval()
    print(f"Model loaded in {load_timing[0]:.1f}s.\n")

    ds, spec = load_eval(dataset)
    split = spec.splits["eval"]
    audio_col = spec.audio_col
    # Datasets without a duration column (AESRC) get it from the audio header, the
    # same rule utils.inference_loop applies, so the column is never left empty.
    derive_duration = spec.duration_col is None and spec.audio_undecoded

    completed, ckpt_map = set(), {}
    checkpoint_path = os.path.join(results_dir(dataset), f"wer_{model_key}_partial.csv")
    if os.path.exists(checkpoint_path):
        for r in pd.read_csv(checkpoint_path).to_dict("records"):
            sid = text_value(r.get("ID"))
            if not sid:
                raise ValueError(f"Checkpoint {checkpoint_path} contains an empty ID")
            completed.add(sid)
            ckpt_map[sid] = r
        print(f"  Resuming: {len(completed)} samples already done\n")

    all_rows, pending, pending_meta = [], [], []
    timing = {"n_fresh": 0, "audio_seconds": 0.0}

    def _sigterm(signum, frame):
        if all_rows:
            save_checkpoint(all_rows, model_key, dataset)
        print(f"\n[SIGTERM] saved {len(all_rows)} rows", flush=True)
        sys.exit(143)
    signal.signal(signal.SIGTERM, _sigterm)

    def flush():
        if not pending:
            return
        try:
            hyps = transcribe_batch(model, pending, audio_col)
        except Exception:
            if all_rows:
                path = save_checkpoint(all_rows, model_key, dataset)
                print(f"\n[ERROR] saved {len(all_rows)} resumable rows to {path}", flush=True)
            raise
        for s, (sid, tr, duration), hyp in zip(pending, pending_meta, hyps):
            all_rows.append(build_sample_row(s, sid, tr, hyp, spec=spec, split=split,
                                             duration=duration))
            timing["n_fresh"] += 1
            if spec.duration_col:
                timing["audio_seconds"] += positive_float(s.get(spec.duration_col)) or 0.0
            elif duration:
                timing["audio_seconds"] += duration
            if len(all_rows) % CHECKPOINT_EVERY == 0:
                save_checkpoint(all_rows, model_key, dataset)
        pending.clear()
        pending_meta.clear()

    print(f"--- {spec.display} [{split}] : {len(ds)} samples, model={model_key} ---")
    t_start = time.monotonic()
    for sample in tqdm(ds, desc=f"{dataset}:{model_key}"):
        transcript = text_value(sample.get(spec.gold_ref_col))
        if not transcript:
            continue
        sid = sample_id(sample, spec)
        duration = probe_audio_duration(sample.get(audio_col)) if derive_duration else None
        if sid in completed:
            flush()
            hyp = text_value((ckpt_map.get(sid) or {}).get("hypothesis_raw"))
            all_rows.append(build_sample_row(sample, sid, transcript, hyp, spec=spec, split=split,
                                             duration=duration))
        else:
            pending.append(sample)
            pending_meta.append((sid, transcript, duration))
            if len(pending) >= BATCH_SIZE:
                flush()
    flush()

    # Wall-time over freshly transcribed clips only, as in utils.inference_loop.
    elapsed = time.monotonic() - t_start
    run_timing = {
        "elapsed_seconds": round(elapsed, 1),
        "clips_transcribed_this_run": timing["n_fresh"],
        "audio_seconds_this_run": round(timing["audio_seconds"], 1),
    }
    if timing["n_fresh"] and timing["audio_seconds"]:
        run_timing["seconds_per_audio_second"] = round(elapsed / timing["audio_seconds"], 4)

    out_path = os.path.join(stage1_raw_dir(dataset), f"wer_{model_key}_raw.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    write_run_manifest(model_key, dataset, spec,
                       extra={**run_timing,
                              "decode_kwargs": {"batch_size": BATCH_SIZE, "engine_defaults": "nemo"}})
    print(f"\nSaved: {out_path}  ({len(all_rows)} samples)")
    remove_checkpoint(model_key, dataset)
    print("Done.")


if __name__ == "__main__":
    main()
