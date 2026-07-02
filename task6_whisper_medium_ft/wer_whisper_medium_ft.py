"""
Stage 1: ASR Transcription — fine-tuned Whisper Medium (and same-engine baseline).

One parameterized driver, selected by the MODEL_NAME env var:
    MODEL_NAME=medium_ft  -> load the fine-tuned model from models/whisper_medium_ft/
    MODEL_NAME=medium_hf  -> load pretrained openai/whisper-medium (same HF engine baseline)

Both transcribe the SAME `test` split through the SAME chunked HF pipeline (utils.transcribe_hf),
so the only difference is the model weights. This isolates the true fine-tuning gain from any
decoding/engine differences. medium_hf lands near the existing openai-whisper medium number
(14.72% transcript_clean) and serves as the apples-to-apples baseline for the comparison.

Saves to results/tie/stage1_raw_transcripts/wer_{MODEL_NAME}_raw.csv (same schema as all other models),
so Stage 2 (normalize_and_score.py) and Stage 3 (analysis/) treat it as just another model.

Resumable — re-running picks up from the last checkpoint.

Usage:
    MODEL_NAME=medium_ft python task6_whisper_medium_ft/wer_whisper_medium_ft.py
    MODEL_NAME=medium_hf python task6_whisper_medium_ft/wer_whisper_medium_ft.py
"""

import os
import sys
import warnings

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.transcribe_hf import build_asr_pipeline, transcribe_sample_hf
from utils.io_helpers import (
    load_dataset_test,
    results_dir,
    stage1_raw_dir,
    build_sample_row,
    save_checkpoint,
    remove_checkpoint,
    raw_audio_column,
)

warnings.filterwarnings("ignore")

MODEL_NAME = os.environ.get("MODEL_NAME", "medium_ft")
CHECKPOINT_EVERY = 200

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_FT_DIR = os.path.join(REPO_ROOT, "models", "whisper_medium_ft")

# Resolve which weights to load. MODEL_SOURCE (an explicit path/HF id) overrides the
# per-name default, so any local fine-tuned variant (e.g. the speaker-disjoint model,
# MODEL_NAME=medium_ft_disjoint) can be evaluated without editing this script.
_source = os.environ.get("MODEL_SOURCE")
if _source:
    model_path = _source
elif MODEL_NAME == "medium_ft":
    model_path = os.environ.get("FT_OUTPUT_DIR", DEFAULT_FT_DIR)
elif MODEL_NAME == "medium_hf":
    model_path = os.environ.get("FT_BASE_MODEL", "openai/whisper-medium")
else:
    sys.exit(f"[ERROR] MODEL_NAME='{MODEL_NAME}' needs an explicit MODEL_SOURCE "
             f"(path or HF id), or use 'medium_ft' / 'medium_hf'.")

# Local model dirs must exist; HF ids (contain no path separator) are downloaded.
if os.sep in model_path and not os.path.isdir(model_path):
    sys.exit(f"[ERROR] model weights not found at {model_path}. Run finetune.py first.")

print(f"=== Stage 1 transcription: {MODEL_NAME}  (weights: {model_path}) ===\n")
pipe = build_asr_pipeline(model_path)

ds = load_dataset_test()
# Read audio straight from arrow storage by row index, bypassing datasets.Audio's decode
# entirely (datasets>=4.0 mandates torchcodec for that — a fragile torch/ffmpeg ABI
# dependency on HPC). ds_meta drops "audio" so plain iteration below never formats/decodes
# it; raw_audio is indexed separately per row via decode_audio_value in transcribe_sample_hf.
raw_audio = raw_audio_column(ds)
ds_meta = ds.remove_columns(["audio"])

checkpoint_path = os.path.join(results_dir(), f"wer_{MODEL_NAME}_partial.csv")
completed_ids: set[str] = set()
ckpt_map: dict[str, dict] = {}

if os.path.exists(checkpoint_path):
    df_partial = pd.read_csv(checkpoint_path)
    for r in df_partial.to_dict("records"):
        sid = str(r["ID"])
        completed_ids.add(sid)
        ckpt_map[sid] = r
    print(f"  Resuming from checkpoint: {len(completed_ids)} samples already done\n")

all_rows: list[dict] = []

print(f"--- Processing test split ({len(ds_meta)} samples) ---")

for idx, sample in enumerate(tqdm(ds_meta, desc=f"test ({MODEL_NAME})")):
    transcript = (sample.get("Transcript") or "").strip()
    if not transcript:
        continue

    sample_id = sample.get("ID", "")

    if str(sample_id) in completed_ids:
        hyp_raw = str(ckpt_map.get(str(sample_id), {}).get("hypothesis_raw") or "")
    else:
        hyp_raw = transcribe_sample_hf(pipe, sample, raw_audio[idx].as_py())

    all_rows.append(build_sample_row(sample, str(sample_id), transcript, hyp_raw))

    if len(all_rows) % CHECKPOINT_EVERY == 0:
        save_checkpoint(all_rows, MODEL_NAME)
        print(f"  [checkpoint] {len(all_rows)} samples saved")

out_path = os.path.join(stage1_raw_dir(), f"wer_{MODEL_NAME}_raw.csv")
pd.DataFrame(all_rows).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(all_rows)} samples)")
print("Run 'python normalize_and_score.py' for WER evaluation.")

remove_checkpoint(MODEL_NAME)
print("\nDone.")
