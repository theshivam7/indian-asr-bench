"""
Stage 1: ASR transcription, NVIDIA Parakeet (NeMo), batched.

Drives both Parakeet models via the registry:
    --model parakeet       -> Parakeet-TDT-0.6B-v2 (transducer)
    --model parakeet_ctc   -> Parakeet-CTC-1.1B     (ctc; 2nd cannot-hallucinate witness)

Usage:
    python parakeet/wer_parakeet.py --model parakeet_ctc --dataset tie
    python parakeet/wer_parakeet.py --model parakeet     --dataset svarah
    python parakeet/wer_parakeet.py --model parakeet --efficiency   # timing, not transcripts

Writes results/<dataset>/stage1_raw_transcripts/wer_<model>_raw.csv, or with
--efficiency, results/<dataset>/efficiency/efficiency_<model>.json (see
utils/efficiency.py for the measurement protocol).
Uses batch NeMo transcription (its own loop, so not utils.inference_loop), but is
dataset-aware through the DatasetSpec.
"""

import argparse
import logging
import os
import signal
import sys
import tempfile
import warnings

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.efficiency import (DEFAULT_CLIPS, DEFAULT_SEED, DEFAULT_WARMUP, count_parameters,
                              run_efficiency_benchmark, timed)

BATCH_SIZE = 16
CHECKPOINT_EVERY = 50


def transcribe_batch(model, samples, audio_col):
    from utils.io_helpers import audio_to_wav_16k

    tmp_paths = []
    try:
        for s in samples:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_to_wav_16k(s[audio_col], tmp.name)
                tmp_paths.append(tmp.name)
        outputs = model.transcribe(tmp_paths, batch_size=len(tmp_paths))
        return [(o.text if hasattr(o, "text") else str(o)).strip() for o in outputs]
    except Exception as e:
        print(f"  [WARN] batch transcription failed: {e}", flush=True)
        return [""] * len(samples)
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="parakeet", choices=["parakeet", "parakeet_ctc"])
    ap.add_argument("--dataset", default="tie")
    ap.add_argument("--efficiency", action="store_true",
                    help="measure speed/memory on a seeded clip subset instead of transcribing the split")
    ap.add_argument("--clips", type=int, default=DEFAULT_CLIPS,
                    help="--efficiency: number of measured clips (default %(default)s)")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help="--efficiency: untimed warmup clips (default %(default)s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="--efficiency: subset seed, keep identical across models (default %(default)s)")
    args = ap.parse_args()

    logging.getLogger("nemo_logger").setLevel(logging.WARNING)
    logging.getLogger("nemo").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.enabled = False  # avoid CUDNN_STATUS_NOT_INITIALIZED on LSTM load

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
    with timed(load_timing):
        model = nemo_asr.models.ASRModel.from_pretrained(model_id)
        if device == "cuda":
            model = model.cuda()
        model.eval()
    print(f"Model loaded in {load_timing[0]:.1f}s.\n")

    if args.efficiency:
        from utils.registry import get_dataset

        eff_audio_col = get_dataset(dataset).audio_col

        # Measured one clip at a time, not at BATCH_SIZE. Per-clip latency is only
        # comparable across engines if every engine sees the same single-stream
        # conditions, and Whisper/Qwen3 have no batched path here. Parakeet's
        # throughput under batching is therefore strictly better than reported;
        # `batched_throughput_available` records that so the paper does not read
        # this as Parakeet's ceiling.
        def transcribe_one(sample: dict) -> str:
            return transcribe_batch(model, [sample], eff_audio_col)[0]

        run_efficiency_benchmark(
            model_key, dataset, transcribe_one,
            n_clips=args.clips, warmup=args.warmup, seed=args.seed,
            model_load_seconds=load_timing[0], param_count=count_parameters(model),
            extra={"decode_kwargs": {"batch_size": 1, "engine_defaults": "nemo"},
                   "batched_throughput_available": True,
                   "production_batch_size": BATCH_SIZE},
        )
        print("\nDone.")
        return

    ds, spec = load_eval(dataset)
    split = spec.splits["eval"]
    audio_col = spec.audio_col

    completed, ckpt_map = set(), {}
    checkpoint_path = os.path.join(results_dir(dataset), f"wer_{model_key}_partial.csv")
    if os.path.exists(checkpoint_path):
        for r in pd.read_csv(checkpoint_path).to_dict("records"):
            sid = str(r["ID"]); completed.add(sid); ckpt_map[sid] = r
        print(f"  Resuming: {len(completed)} samples already done\n")

    all_rows, pending, pending_meta = [], [], []

    def _sigterm(signum, frame):
        if all_rows:
            save_checkpoint(all_rows, model_key, dataset)
            pd.DataFrame(all_rows).to_csv(
                os.path.join(stage1_raw_dir(dataset), f"wer_{model_key}_interrupted.csv"), index=False)
        print(f"\n[SIGTERM] saved {len(all_rows)} rows", flush=True)
        sys.exit(143)
    signal.signal(signal.SIGTERM, _sigterm)

    def flush():
        if not pending:
            return
        hyps = transcribe_batch(model, pending, audio_col)
        for s, (sid, tr), hyp in zip(pending, pending_meta, hyps):
            all_rows.append(build_sample_row(s, sid, tr, hyp, spec=spec, split=split))
            if len(all_rows) % CHECKPOINT_EVERY == 0:
                save_checkpoint(all_rows, model_key, dataset)
        pending.clear(); pending_meta.clear()

    print(f"--- {spec.display} [{split}] : {len(ds)} samples, model={model_key} ---")
    for sample in tqdm(ds, desc=f"{dataset}:{model_key}"):
        transcript = str(sample.get(spec.gold_ref_col) or "").strip()
        if not transcript:
            continue
        sid = sample_id(sample, spec)
        if sid in completed:
            flush()
            hyp = str((ckpt_map.get(sid) or {}).get("hypothesis_raw") or "")
            all_rows.append(build_sample_row(sample, sid, transcript, hyp, spec=spec, split=split))
        else:
            pending.append(sample); pending_meta.append((sid, transcript))
            if len(pending) >= BATCH_SIZE:
                flush()
    flush()

    out_path = os.path.join(stage1_raw_dir(dataset), f"wer_{model_key}_raw.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    write_run_manifest(model_key, dataset, spec,
                       extra={"decode_kwargs": {"batch_size": BATCH_SIZE, "engine_defaults": "nemo"}})
    print(f"\nSaved: {out_path}  ({len(all_rows)} samples)")
    remove_checkpoint(model_key, dataset)
    print("Done.")


if __name__ == "__main__":
    main()
