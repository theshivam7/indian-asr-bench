#!/usr/bin/env python3
"""Fine-tune Whisper on a registry dataset using a step-based training recipe.

Adapted from an externally supplied reference script (`step4_train_whisper.py`) for the
tiny/small capacity study, see results/tie/analysis/findings_tiny_small_ft.md. Kept
faithful to the source script's structure and hyperparameters; the substantive changes are
the data source (TIE_shorts via HF, not local JSONL+wav manifests) and the disclosed
additions below, needed for this to produce a model our pipeline can evaluate.

Recipe (verbatim from the source script): STEP-based training (not epoch-based, unlike
finetune/finetune_medium.py's medium study), max_steps=2000, warmup_steps=100,
lr=1e-5, batch=8, grad_accum=4 (effective batch 32), fp16, eval/save every 200 steps,
greedy predict_with_generate, checkpoint-selection WER computed with OpenAI's
EnglishTextNormalizer, NOT this project's `transcript_clean` normalizer used by
finetune.py's medium study. This is a disclosed recipe difference, not a bug; see the
findings report for the full list of deltas vs the medium recipe (effective batch 32 vs
16, fp16 vs bf16, no SpecAugment, no early stopping, different selection-metric normalizer).

Disclosed additions (the source script lacked these; needed for a usable, comparable model):
  - load_best_model_at_end + metric_for_best_model="wer": the source script has no
    best-checkpoint selection, so training would return the LAST checkpoint (~epoch 8-9 at
    max_steps=2000 on TIE's ~7.2k-clip train set) rather than the best one, risky given the
    medium study's finding that Whisper FT best-checkpoints on this dataset arrive at epoch 1.
  - explicit seed (--seed, default 42 = Trainer's own default; made explicit rather than
    implicit). The seed is also applied to python/numpy/torch before the model is built
    (utils.finetune_data.seed_everything) and passed as data_seed, so a whole run is
    reproducible from one number. See --seed below for the multi-seed study.
  - TIE-specific filtering (empty transcript / >30s / no embedded audio) via
    utils.finetune_data.filter_tie_split, the source script assumed pre-filtered JSONL
    manifests.
  - trainer.save_model()/processor.save_pretrained() to the output dir root: the source
    script left only rolling `checkpoint-N` dirs; our downstream eval (evaluate_finetuned.py)
    expects a clean directory with the final model+processor at the root.
  - model.generation_config.language/task set explicitly to English/transcribe: the source
    script only clears forced_decoder_ids and suppress_tokens, but never sets language/task on
    generation_config. Since the base checkpoints are multilingual (not .en), an unset
    generation_config.language leaves generate() free to fall back to language
    auto-detection during predict_with_generate eval, which could silently corrupt the
    very WER metric checkpoint selection depends on. This is the one place we deviate from
    "keep the recipe verbatim": it's the standard step in every published Whisper
    fine-tuning recipe (including finetune_medium.py's own medium study) and its absence risks the
    whole run's checkpoint selection, not just a cosmetic recipe difference.

Datasets: --dataset selects any registry dataset with train+validation splits.
    tie   (default), loads TIE_shorts directly, preserving the original capacity-study
          code path byte-for-byte (TIE's validation split has no duration column, so it
          cannot go through the adapter's schema validation).
    aesrc, loads via utils.datasets.load_split, which applies the registry's
          accent == "INDIAN" filter and schema/ID validation. All three sizes
          (tiny/small/medium) use this same recipe on AESRC. Note: AESRC's validation
          split shares the train split's 38 speakers, so checkpoint-selection WER
          measures fit, not speaker generalization; the speaker-disjoint test split is
          never touched during training.

Usage:
    python finetune/finetune_tiny_small.py \\
        --base-model openai/whisper-tiny --output-dir models/whisper_tiny_ft
    python finetune/finetune_tiny_small.py --dataset aesrc \\
        --base-model openai/whisper-medium --output-dir models/whisper_medium_aesrc_ft
    python finetune/finetune_tiny_small.py --dataset aesrc --seed 43 \\
        --base-model openai/whisper-tiny --output-dir models/whisper_tiny_aesrc_ft_seed43

CLI (matches the source script's flags, plus --dataset/--base-model/--output-dir/--seed/--max-train-samples):
    --max-steps (2000)  --lr (1e-5)  --batch-size (8)  --grad-accum (4)  --seed (42)
    --max-train-samples (unset), subset training data for a quick smoke test

Multi-seed study: one seed cannot bound run-to-run variance, so each size is trained
across several seeds and reported as mean +/- standard deviation. Drive that with
finetune/run_seeds.sh (which also handles per-seed output dirs, transcription and
scoring), then aggregate with analysis/compare_seeds.py.
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
from utils.registry import get_dataset
from utils.finetune_data import (
    DataCollatorSpeechSeq2SeqWithPadding,
    filter_finetune_split,
    filter_tie_split,
    seed_everything,
)

warnings.filterwarnings("ignore")


class WhisperFTDataset(Dataset):
    """Torch Dataset over a filtered fine-tuning split: reads audio straight from arrow
    storage (utils.io_helpers.decode_audio_value, bypassing datasets' Audio decode) and
    extracts log-mel features on the fly in __getitem__.

    `ds` must already be filtered and flatten_indices()-materialized so raw arrow row i
    matches logical row i.
    """

    def __init__(self, ds, processor: WhisperProcessor, transcript_col: str,
                 audio_col: str = "audio"):
        self.raw_audio = raw_audio_column(ds, audio_col)
        self.transcripts = ds[transcript_col]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.transcripts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        audio_value = self.raw_audio[idx].as_py()
        audio_array, sr = decode_audio_value(audio_value, target_sr=16000)
        feats = self.processor.feature_extractor(audio_array, sampling_rate=sr).input_features[0]
        # Targets come from the gold transcript column; strip only a wrapping quote pair
        # (a TIE data quirk, no-op for datasets without it), same as finetune_medium.py.
        labels = self.processor.tokenizer(strip_wrapping_quotes(self.transcripts[idx])).input_ids
        return {"input_features": feats, "labels": labels}


def load_finetune_splits(dataset_key: str):
    """Return raw (unfiltered) train + validation HF splits for a dataset.

    TIE loads directly (its validation split lacks the duration column the adapter's
    schema validation requires); every other dataset goes through utils.datasets.
    load_split, which applies any registry row filter and fail-early validation.
    """
    if dataset_key == "tie":
        spec = get_dataset(dataset_key)
        train_hf = load_dataset(spec.hf_id, split=spec.splits["train"], cache_dir=HF_CACHE,
                                revision=spec.hf_revision)
        eval_hf = load_dataset(spec.hf_id, split=spec.splits["validation"], cache_dir=HF_CACHE,
                               revision=spec.hf_revision)
        return train_hf, eval_hf

    from utils.datasets import load_split

    train_hf, _ = load_split(dataset_key, "train")
    eval_hf, _ = load_split(dataset_key, "validation")
    return train_hf, eval_hf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tie", help="registry dataset key (tie, aesrc, ...)")
    parser.add_argument("--base-model", default="openai/whisper-tiny")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42,
                         help="Training seed: RNG state for init order, dataloader shuffling "
                              "and dropout. Vary it to measure run-to-run variance.")
    parser.add_argument("--max-train-samples", type=int, default=None,
                         help="Subset training data for a quick smoke test.")
    args = parser.parse_args()

    spec = get_dataset(args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[finetune_tiny_small] dataset={args.dataset} base_model={args.base_model} "
          f"out={args.output_dir} seed={args.seed} device={device}")

    # Before from_pretrained(): see utils.finetune_data.seed_everything for why the
    # Trainer's own seed= is not sufficient on its own.
    seed_everything(args.seed)

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

    print(f"Loading {spec.display} train/validation splits ...")
    train_hf, eval_hf = load_finetune_splits(args.dataset)

    if args.max_train_samples:
        # Smoke-test path: subset the RAW dataset BEFORE filtering. The filters call
        # flatten_indices(), which materializes every surviving clip's full audio into a
        # new arrow table -- expensive enough to OOM-kill a memory-constrained node
        # (observed 2026-07-09), and it happens regardless of --max-train-samples if the
        # cap is only applied afterward. 3x headroom on the raw slice comfortably survives
        # the ~91% filter pass-rate for any reasonable smoke-test sample size.
        train_hf = train_hf.select(range(min(args.max_train_samples * 3, len(train_hf))))
        eval_hf = eval_hf.select(range(min(24, len(eval_hf))))
        print(f"  SMOKE TEST: pre-capped raw train to {len(train_hf)} rows before filtering")

    print("Filtering ...")
    if args.dataset == "tie":
        # Preserved TIE code path: validation split has no duration column.
        train_hf = filter_tie_split(train_hf, has_duration_col=True, label="train")
        eval_hf = filter_tie_split(eval_hf, has_duration_col=False, label="validation")
    else:
        train_hf = filter_finetune_split(train_hf, spec, label="train")
        eval_hf = filter_finetune_split(eval_hf, spec, label="validation")

    if args.max_train_samples:
        train_hf = train_hf.select(range(min(args.max_train_samples, len(train_hf)))).flatten_indices()
        eval_hf = eval_hf.select(range(min(8, len(eval_hf)))).flatten_indices()
        print(f"  SMOKE TEST: capped filtered train to {len(train_hf)} samples")

    train_ds = WhisperFTDataset(train_hf, processor, spec.gold_ref_col, spec.audio_col)
    eval_ds = WhisperFTDataset(eval_hf, processor, spec.gold_ref_col, spec.audio_col)
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
        # mirrors the same guard in finetune_medium.py's compute_metrics.
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
        seed=args.seed,
        # data_seed drives the train sampler's shuffle order. Left unset it falls back to
        # `seed`, so the two are redundant today, but stating it keeps the data order tied
        # to this run's seed even if the recipe later sets one of them independently.
        data_seed=args.seed,
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
        "dataset": args.dataset,
        "output_dir": str(args.output_dir),
        "base_model": args.base_model,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
    }
    (args.output_dir / "eval_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    print("\nNext: run transcription with")
    print(f"  DATASET={args.dataset} MODEL_NAME=<model_key> MODEL_SOURCE={args.output_dir} "
          f"python finetune/evaluate_finetuned.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
