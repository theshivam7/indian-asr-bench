"""
Stage 1: ASR Transcription, fine-tuned Whisper models (Tiny/Small/Medium) and their
same-engine pretrained baselines.

One parameterized driver, selected by the MODEL_NAME env var. Two names are built in:
    MODEL_NAME=medium_ft  -> load the fine-tuned model from models/whisper_medium_ft/
    MODEL_NAME=medium_hf  -> load pretrained openai/whisper-medium (same HF engine baseline)
Any other registry `hf_whisper` key also works out of the box (tiny_ft, tiny_hf, small_ft,
small_hf, ...), resolved via utils.registry, no code change needed to add one.

Both transcribe the SAME `test` split through the SAME chunked HF pipeline (utils.transcribe_hf),
so the only difference is the model weights. This isolates the true fine-tuning gain from any
decoding/engine differences.

Saves to results/<dataset>/stage1_raw_transcripts/wer_{MODEL_NAME}_raw.csv (same schema as all
other models), so Stage 2 (normalize_and_score.py) and Stage 3 (analysis/) treat it as just
another model.

Uses the shared utils.inference_loop (dataset-aware, resumable, SIGTERM-safe), same as
whisper_asr/run_whisper.py and qwen3/wer_qwen3.py. The chunked HF pipeline needs
the raw (undecoded) audio value rather than datasets' Audio-feature decode, so this driver
opts into inference_loop's two-argument transcribe_one(sample, raw_audio_value) form.

DATASET selects the registry dataset whose eval split is transcribed (default: tie).

Usage:
    MODEL_NAME=medium_ft python finetune/evaluate_finetuned.py
    MODEL_NAME=medium_hf python finetune/evaluate_finetuned.py
    MODEL_NAME=tiny_ft MODEL_SOURCE=models/whisper_tiny_ft python finetune/evaluate_finetuned.py
    DATASET=aesrc MODEL_NAME=tiny_aesrc_ft MODEL_SOURCE=models/whisper_tiny_aesrc_ft python finetune/evaluate_finetuned.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.transcribe_hf import build_asr_pipeline, transcribe_sample_hf
from utils.inference_loop import run_transcription
from utils.registry import MODEL_BY_KEY

warnings.filterwarnings("ignore")

MODEL_NAME = os.environ.get("MODEL_NAME", "medium_ft")
DATASET = os.environ.get("DATASET", "tie")

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_FT_DIR = os.path.join(REPO_ROOT, "models", "whisper_medium_ft")

# Resolve which weights to load. MODEL_SOURCE (an explicit path/HF id) overrides the
# per-name default, so any local fine-tuned variant can be evaluated without editing
# this script.
_source = os.environ.get("MODEL_SOURCE")
_is_local_dir = False   # set True when the path must already exist on disk
if _source:
    model_path = _source
elif MODEL_NAME == "medium_ft":
    model_path = os.environ.get("FT_OUTPUT_DIR", DEFAULT_FT_DIR)
elif MODEL_NAME == "medium_hf":
    model_path = os.environ.get("FT_BASE_MODEL", "openai/whisper-medium")
elif MODEL_NAME in MODEL_BY_KEY and MODEL_BY_KEY[MODEL_NAME].engine == "hf_whisper":
    # Any other registry hf_whisper entry (e.g. tiny_hf/tiny_ft/small_hf/small_ft)
    # resolves via its registry model_id, an HF id used as-is, or a "models/..."
    # local path made relative to the repo root (weights on this cluster live
    # under $SCRATCH, not in the repo, so MODEL_SOURCE remains the usual override).
    _mid = MODEL_BY_KEY[MODEL_NAME].model_id
    if "/" in _mid and not _mid.startswith("models"):
        model_path = _mid                              # HF hub id, downloaded on demand
    else:
        model_path = os.path.join(REPO_ROOT, _mid)     # registry-local weights
        _is_local_dir = True
else:
    sys.exit(f"[ERROR] MODEL_NAME='{MODEL_NAME}' needs an explicit MODEL_SOURCE "
             f"(path or HF id), or use 'medium_ft' / 'medium_hf'.")

# Local model dirs must exist; HF ids are downloaded. Distinguish by the project's own
# convention (every local weight path in the registry starts with "models/", see
# utils/registry.py) rather than by presence of a path separator: HF namespaced ids like
# "openai/whisper-medium" also contain "/", so that check alone misfires on every HF-hosted
# pretrained baseline (medium_hf, tiny_hf, small_hf, ...) and would incorrectly exit before
# ever calling build_asr_pipeline. Registry-resolved local paths become absolute after the
# REPO_ROOT join, so the branch above marks them explicitly rather than re-matching the prefix.
if (_is_local_dir or model_path.startswith(("models/", "models" + os.sep))) \
        and not os.path.isdir(model_path):
    sys.exit(f"[ERROR] model weights not found at {model_path}. "
             f"Run finetune_medium.py or finetune_tiny_small.py first.")

print(f"=== Stage 1 transcription: {MODEL_NAME} on {DATASET}  (weights: {model_path}) ===\n")
pipe = build_asr_pipeline(model_path)

run_transcription(
    MODEL_NAME, DATASET,
    transcribe_one=lambda sample, raw_audio_value: transcribe_sample_hf(pipe, sample, raw_audio_value),
    manifest_extra={"decode_kwargs": {"engine_defaults": "hf_whisper chunked pipeline"}},
)
print("\nDone.")
