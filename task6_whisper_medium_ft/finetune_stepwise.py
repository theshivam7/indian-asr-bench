#!/usr/bin/env python3
"""Fine-tune Whisper on TIE_shorts using the professor's training recipe.

Adapted from the professor's `step4_train_whisper.py` (lab_clean_code, ASPIRE2A) for the
tiny/small capacity study — see results/tie/analysis/findings_tiny_small_ft.md. Kept
faithful to his script's structure and hyperparameters; the substantive changes are the
data source (TIE_shorts via HF, not local JSONL+wav manifests) and the disclosed
additions below, needed for this to produce a model our pipeline can evaluate.

Recipe (verbatim from step4_train_whisper.py): STEP-based training (not epoch-based, unlike
task6_whisper_medium_ft/finetune.py's medium study), max_steps=2000, warmup_steps=100,
lr=1e-5, batch=8, grad_accum=4 (effective batch 32), fp16, eval/save every 200 steps,
greedy predict_with_generate, checkpoint-selection WER computed with OpenAI's
EnglishTextNormalizer — NOT this project's `transcript_clean` normalizer used by
finetune.py's medium study. This is a disclosed recipe difference, not a bug; see the
findings report for the full list of deltas vs the medium recipe (effective batch 32 vs
16, fp16 vs bf16, no SpecAugment, no early stopping, different selection-metric normalizer).

Disclosed additions (his script lacked these; needed for a usable, comparable model):
  - load_best_model_at_end + metric_for_best_model="wer": his script has no best-checkpoint
    selection, so training would return the LAST checkpoint (~epoch 8-9 at max_steps=2000 on
    TIE's ~7.2k-clip train set) rather than the best one — risky given the medium study's
    finding that Whisper FT best-checkpoints on this dataset arrive at epoch 1.
  - explicit seed=42 (Trainer's own default; made explicit rather than implicit).
  - TIE-specific filtering (empty transcript / >30s / no embedded audio) via
    utils.finetune_data.filter_tie_split — his script assumed pre-filtered JSONL manifests.
  - trainer.save_model()/processor.save_pretrained() to the output dir root: his script
    left only rolling `checkpoint-N` dirs; our downstream eval (wer_whisper_medium_ft.py)
    expects a clean directory with the final model+processor at the root.
  - model.generation_config.language/task set explicitly to English/transcribe: his script
    only clears forced_decoder_ids and suppress_tokens, but never sets language/task on
    generation_config. Since the base checkpoints are multilingual (not .en), an unset
    generation_config.language leaves generate() free to fall back to language
    auto-detection during predict_with_generate eval — which could silently corrupt the
    very WER metric checkpoint selection depends on. This is the one place we deviate from
    "keep the recipe verbatim": it's the standard step in every published Whisper
    fine-tuning recipe (including finetune.py's own medium study) and its absence risks the
    whole run's checkpoint selection, not just a cosmetic recipe difference.

Usage:
    python task6_whisper_medium_ft/finetune_stepwise.py \\
        --base-model openai/whisper-tiny --output-dir models/whisper_tiny_ft
    python task6_whisper_medium_ft/finetune_stepwise.py \\
        --base-model openai/whisper-small --output-dir models/whisper_small_ft

CLI (matches his script's flags, plus --base-model/--output-dir/--max-train-samples):
    --max-steps (2000)  --lr (1e-5)  --batch-size (8)  --grad-accum (4)
    --max-train-samples (unset)  — subset training data for a quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import jiwer
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import HF_CACHE, raw_audio_column, decode_audio_value
from utils.normalize import strip_wrapping_quotes
from utils.finetune_data import DataCollatorSpeechSeq2SeqWithPadding, filter_tie_split

warnings.filterwarnings("ignore")


class TIEWhisperDataset(Dataset):
    """Torch Dataset over a filtered TIE_shorts split: reads audio straight from arrow
    storage (utils.io_helpers.decode_audio_value, bypassing datasets' Audio decode — same
    trick finetune.py uses) and extracts log-mel features on the fly in __getitem__,
    mirroring the professor's LocalWhisperDataset (which read local wavs via librosa).

    `ds` must already be filter_tie_split()-filtered and flatten_indices()-materialized so
    raw arrow row i matches logical row i.
    """

    def __init__(self, ds, processor: WhisperProcessor):
        self.raw_audio = raw_audio_column(ds)
        self.transcripts = ds["Transcript"]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.transcripts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        audio_value = self.raw_audio[idx].as_py()
        audio_array, sr = decode_audio_value(audio_value, target_sr=16000)
        feats = self.processor.feature_extractor(audio_array, sampling_rate=sr).input_features[0]
        # Targets come from the gold `Transcript` column; strip only a wrapping quote pair
        # (a known TIE data quirk — see utils.normalize.strip_wrapping_quotes), same as
        # finetune.py's medium study.
        labels = self.processor.tokenizer(strip_wrapping_quotes(self.transcripts[idx])).input_ids
        return {"input_features": feats, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="openai/whisper-tiny")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None,
                         help="Subset training data for a quick smoke test.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[finetune_stepwise] base_model={args.base_model} out={args.output_dir} device={device}")

    processor = WhisperProcessor.from_pretrained(
        args.base_model, language="English", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    # Transformers >=5: generation knobs live on generation_config, not model.config
    model.generation_config.forced_decoder_ids = None
    if hasattr(model.generation_config, "suppress_tokens"):
        model.generation_config.suppress_tokens = None
    # Disclosed addition (see module docstring): force English/transcribe explicitly so
    # predict_with_generate eval can't fall back to language auto-detection.
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"

    print("Loading TIE_shorts train/validation splits ...")
    train_hf = load_dataset("raianand/TIE_shorts", split="train", cache_dir=HF_CACHE)
    eval_hf = load_dataset("raianand/TIE_shorts", split="validation", cache_dir=HF_CACHE)

    if args.max_train_samples:
        # Smoke-test path: subset the RAW dataset BEFORE filtering. filter_tie_split()
        # internally calls flatten_indices() twice, which materializes every surviving
        # clip's full audio array into a new arrow table -- for TIE's ~7.2k-clip filtered
        # train split that's expensive enough to OOM-kill a memory-constrained login node
        # (observed 2026-07-09), and it happens regardless of --max-train-samples if the
        # cap is only applied afterward. 3x headroom on the raw slice comfortably survives
        # TIE's ~91% filter pass-rate for any reasonable smoke-test sample size.
        train_hf = train_hf.select(range(min(args.max_train_samples * 3, len(train_hf))))
        eval_hf = eval_hf.select(range(min(24, len(eval_hf))))
        print(f"  SMOKE TEST: pre-capped raw train to {len(train_hf)} rows before filtering")

    print("Filtering ...")
    train_hf = filter_tie_split(train_hf, has_duration_col=True, label="train")
    eval_hf = filter_tie_split(eval_hf, has_duration_col=False, label="validation")

    if args.max_train_samples:
        train_hf = train_hf.select(range(min(args.max_train_samples, len(train_hf)))).flatten_indices()
        eval_hf = eval_hf.select(range(min(8, len(eval_hf)))).flatten_indices()
        print(f"  SMOKE TEST: capped filtered train to {len(train_hf)} samples")

    train_ds = TIEWhisperDataset(train_hf, processor)
    eval_ds = TIEWhisperDataset(eval_hf, processor)
    print(f"  train: {len(train_ds)} clips   validation: {len(eval_ds)} clips")

    tokenizer = processor.tokenizer
    try:
        spelling_mapping = tokenizer.english_spelling_mapping
    except AttributeError:
        spelling_mapping = {}
    normalizer = EnglishTextNormalizer(spelling_mapping)

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm = [normalizer(s) for s in pred_str]
        label_norm = [normalizer(s) for s in label_str]

        # Drop pairs whose reference normalized to empty (jiwer requires non-empty refs);
        # mirrors the same guard in finetune.py's compute_metrics.
        refs, hyps = [], []
        for r, h in zip(label_norm, pred_norm):
            if r.strip():
                refs.append(r)
                hyps.append(h if h.strip() else " ")
        wer = jiwer.wer(refs, hyps) if refs else 1.0
        return {"wer": wer}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        max_steps=args.max_steps,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        predict_with_generate=True,
        save_total_limit=2,
        report_to=["none"],
        dataloader_num_workers=0,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        ),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    trainer.train()

    print("\nSaving best model + processor ...")
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    results = trainer.evaluate()
    summary = {
        "eval_wer": float(results["eval_wer"]),
        "output_dir": str(args.output_dir),
        "base_model": args.base_model,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
    }
    (args.output_dir / "eval_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    print("\nNext: run transcription with")
    print(f"  MODEL_NAME=<size>_ft MODEL_SOURCE={args.output_dir} "
          f"python task6_whisper_medium_ft/wer_whisper_medium_ft.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
