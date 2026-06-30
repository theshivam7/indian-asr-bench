"""
Stage 0: Fine-tune Whisper Medium on raianand/TIE_shorts (train split).

Full fine-tuning (all 769M params) following the standard HuggingFace Whisper recipe,
with the correctness + best-practice details that prevent the common pitfalls:

  - Train on `train`, select checkpoint on `validation`, NEVER touch `test`  (no leakage)
  - Targets = `Transcript` (gold ground truth, matches the benchmark)
  - Filter clips > 30s (feature extractor truncates audio to 30s; longer clips would
    pair truncated audio with full transcripts)
  - bf16 (A100-native, more stable than fp16) + gradient checkpointing (use_cache=False)
  - SpecAugment + weight decay + early stopping  (overfitting guards on a ~3k dataset)
  - Epoch cap high, EarlyStopping(patience) decides  (also guards underfitting)
  - load_best_model_at_end by validation WER, computed with the SAME normalization as the
    final benchmark metric (utils.normalize.normalize_text)
  - Resumable: auto-detects the latest checkpoint and resumes

Output: models/whisper_medium_ft/  (best model + processor, ready for HF upload)

Usage:
    python task6_whisper_medium_ft/finetune.py

Tunable via env vars:
    FT_EPOCHS (10)  FT_BATCH (8)  FT_GRAD_ACCUM (2)  FT_LR (1e-5)  FT_PATIENCE (2)
    FT_OUTPUT_DIR (models/whisper_medium_ft)  FT_BASE_MODEL (openai/whisper-medium)
    MAX_TRAIN_SAMPLES (unset)  — subset training data for a quick smoke test
"""

import os
import sys
import warnings

import torch
from datasets import Audio, load_dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
import jiwer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import HF_CACHE
from utils.normalize import normalize_text
from utils.finetune_data import (
    DataCollatorSpeechSeq2SeqWithPadding,
    make_prepare_dataset,
)

warnings.filterwarnings("ignore")

# --------------- Config ---------------
BASE_MODEL = os.environ.get("FT_BASE_MODEL", "openai/whisper-medium")
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.environ.get(
    "FT_OUTPUT_DIR", os.path.join(REPO_ROOT, "models", "whisper_medium_ft")
)
EPOCHS = int(os.environ.get("FT_EPOCHS", "10"))
BATCH = int(os.environ.get("FT_BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("FT_GRAD_ACCUM", "2"))
LR = float(os.environ.get("FT_LR", "1e-5"))
PATIENCE = int(os.environ.get("FT_PATIENCE", "2"))
MAX_TRAIN_SAMPLES = os.environ.get("MAX_TRAIN_SAMPLES")
MAX_AUDIO_SECONDS = 30
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

print("=" * 70)
print("STAGE 0: Fine-tuning Whisper Medium on TIE_shorts")
print("=" * 70)
print(f"  base model : {BASE_MODEL}")
print(f"  output dir : {OUTPUT_DIR}")
print(f"  device     : {device}  (bf16={use_bf16})")
print(f"  epochs(cap): {EPOCHS}  batch: {BATCH}  grad_accum: {GRAD_ACCUM}  "
      f"eff_batch: {BATCH * GRAD_ACCUM}  lr: {LR}  patience: {PATIENCE}")
if MAX_TRAIN_SAMPLES:
    print(f"  SMOKE TEST : capping train to {MAX_TRAIN_SAMPLES} samples")
print()

# --------------- Processor & model ---------------
processor = WhisperProcessor.from_pretrained(
    BASE_MODEL, language="english", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

# Decoder prompt: language/task supplied at generate time; clear forced ids to avoid the
# "forced_decoder_ids conflict" error during predict_with_generate evaluation.
model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Gradient checkpointing is incompatible with the kv-cache — disable cache for training.
model.config.use_cache = False

# SpecAugment regularization (helps on the small ~3k-sample dataset).
model.config.apply_spec_augment = True
model.config.mask_time_prob = 0.05
model.config.mask_feature_prob = 0.05

# --------------- Data ---------------
print("Loading train + validation splits ...")
train_ds = load_dataset("raianand/TIE_shorts", split="train", cache_dir=HF_CACHE)
eval_ds = load_dataset("raianand/TIE_shorts", split="validation", cache_dir=HF_CACHE)


def has_usable_text(transcript) -> bool:
    return bool((transcript or "").strip())


def within_duration(dur) -> bool:
    try:
        return dur is None or float(dur) <= MAX_AUDIO_SECONDS
    except (TypeError, ValueError):
        return True


# Filter on the metadata columns only (input_columns) so the Audio column is NOT decoded
# here — decoding happens once, later, in .map(). Without input_columns, .filter() would
# decode every clip just to read text/duration and then discard it.
n_before = len(train_ds)
train_ds = train_ds.filter(
    lambda transcript, dur: has_usable_text(transcript) and within_duration(dur),
    input_columns=["Transcript", "Speech_Duration_seconds"],
)
print(f"  train: {n_before} -> {len(train_ds)} after dropping empty / >{MAX_AUDIO_SECONDS}s clips")
eval_ds = eval_ds.filter(has_usable_text, input_columns=["Transcript"])
print(f"  validation: {len(eval_ds)} samples")

if MAX_TRAIN_SAMPLES:
    train_ds = train_ds.select(range(min(int(MAX_TRAIN_SAMPLES), len(train_ds))))
    eval_ds = eval_ds.select(range(min(8, len(eval_ds))))

# Resample audio to 16 kHz on access (no manual librosa).
train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))

prepare = make_prepare_dataset(processor)
keep_remove = train_ds.column_names
print("Extracting features (this caches to disk) ...")
train_ds = train_ds.map(prepare, remove_columns=keep_remove, num_proc=1, desc="train features")
eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names, num_proc=1, desc="eval features")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# --------------- Metric (same normalization as the final benchmark) ---------------
tokenizer = processor.tokenizer


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Restore -100 -> pad before decoding.
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Apply the project's forward normalization to BOTH sides (matches transcript_clean).
    pred_norm = [normalize_text(p) for p in pred_str]
    label_norm = [normalize_text(l) for l in label_str]

    # Drop pairs whose reference normalized to empty (jiwer requires non-empty refs).
    refs, hyps = [], []
    for r, h in zip(label_norm, pred_norm):
        if r.strip():
            refs.append(r)
            hyps.append(h if h.strip() else " ")
    wer = jiwer.wer(refs, hyps) if refs else 1.0
    return {"wer": round(wer * 100, 4)}


# --------------- Training args ---------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    num_train_epochs=EPOCHS,
    gradient_checkpointing=True,
    bf16=use_bf16,
    fp16=not use_bf16 and device == "cuda",
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    dataloader_num_workers=int(os.environ.get("FT_NUM_WORKERS", "4")),
    seed=SEED,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)

# --------------- Train (resumable) ---------------
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}\n")

train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

# --------------- Save best model + processor ---------------
print("\nSaving best model + processor ...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
trainer.save_metrics("train", train_result.metrics)

eval_metrics = trainer.evaluate()
print(f"\nFinal best validation WER: {eval_metrics.get('eval_wer'):.4f}%")
print(f"Model saved to: {OUTPUT_DIR}")
print("\nNext: run transcription with")
print("  MODEL_NAME=medium_ft python task6_whisper_medium_ft/wer_whisper_medium_ft.py")
print("\nDone.")
