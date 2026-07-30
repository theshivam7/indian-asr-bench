#!/usr/bin/env bash
# Multi-seed fine-tuning driver: train, transcribe and score one size across N seeds.
#
# A single seed cannot bound run-to-run variance, which is the main disclosed
# limitation of the fine-tuning study. This runs the same recipe repeatedly with
# only the seed changed, so analysis/compare_seeds.py can report mean and standard
# deviation across seeds alongside the within-run bootstrap CI. Those two numbers
# measure different things and are reported separately, never combined.
#
# Usage:
#     bash finetune/run_seeds.sh --size tiny
#     bash finetune/run_seeds.sh --size small --seeds 42,43,44,45,46,47 --dataset aesrc
#
# Then:
#     python analysis/compare_seeds.py --dataset aesrc
#
# Resumable: a seed whose scored output already exists is skipped, so a walltime
# kill loses at most the seed that was in flight. Delete that seed's output dir to
# force a retrain.

set -euo pipefail

SIZE=""
DATASET="aesrc"
SEEDS="42,43,44,45,46,47"   # 6 seeds: enough for a mean and a usable spread
MODELS_ROOT="${MODELS_ROOT:-models}"
SKIP_TRAINED="${SKIP_TRAINED:-1}"

usage() {
    cat >&2 <<'EOF'
Usage: bash finetune/run_seeds.sh --size {tiny|small|medium} [options]

  --size     SIZE      required: tiny, small or medium
  --dataset  KEY       dataset key (default: aesrc)
  --seeds    N,N,...   comma-separated seeds (default: 42,43,44,45,46,47)
  --models-root DIR    where checkpoints are written (default: models)
  --force              retrain even if a seed's output already exists

Each seed produces:
  <models-root>/whisper_<size>_<dataset>_ft_seed<N>/          checkpoint
  results/<dataset>/stage1_raw_transcripts/wer_<size>_<dataset>_ft_seed<N>_raw.csv
  results/<dataset>/stage2_processed/<mode>/wer_<size>_<dataset>_ft_seed<N>_<mode>.csv
EOF
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) SIZE="${2:-}"; shift 2 ;;
        --dataset) DATASET="${2:-}"; shift 2 ;;
        --seeds) SEEDS="${2:-}"; shift 2 ;;
        --models-root) MODELS_ROOT="${2:-}"; shift 2 ;;
        --force) SKIP_TRAINED=0; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

[[ -n "${SIZE}" ]] || usage
case "${SIZE}" in
    tiny|small|medium) ;;
    *) echo "--size must be tiny, small or medium (got '${SIZE}')" >&2; exit 2 ;;
esac

BASE_MODEL="openai/whisper-${SIZE}"
FT_KEY="${SIZE}_${DATASET}_ft"

echo "=== multi-seed fine-tuning ==="
echo "  size=${SIZE}  dataset=${DATASET}  seeds=${SEEDS}"
echo "  base=${BASE_MODEL}"
echo

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"

for SEED in "${SEED_ARRAY[@]}"; do
    SEED="$(echo "${SEED}" | tr -d '[:space:]')"
    [[ -n "${SEED}" ]] || continue

    RUN_KEY="${FT_KEY}_seed${SEED}"
    OUT_DIR="${MODELS_ROOT}/whisper_${SIZE}_${DATASET}_ft_seed${SEED}"
    RAW_CSV="results/${DATASET}/stage1_raw_transcripts/wer_${RUN_KEY}_raw.csv"

    echo "--- seed ${SEED} -> ${RUN_KEY} ---"

    # Each stage is skipped on its OWN completion marker. Treating the raw CSV as
    # "seed finished" and skipping the whole iteration was wrong: transcription writes
    # that file before scoring runs, so a seed whose scoring failed would be skipped
    # forever on every rerun and silently drop out of the aggregate.

    # Completion is decided by eval_results.json, which finetune_tiny_small.py writes
    # last, after both save_model() and processor.save_pretrained() have succeeded.
    # Directory existence is NOT a valid marker: the Trainer creates output_dir at
    # step 0, so a walltime kill or an OOM mid-training leaves a directory with
    # checkpoint-* subdirs and no model at the root. Skipping training on that basis
    # sends a never-saved checkpoint to the transcription step, which then fails on a
    # missing preprocessor_config.json after the GPU time has already been spent.
    if [[ "${SKIP_TRAINED}" == "1" && -f "${OUT_DIR}/eval_results.json" ]]; then
        echo "  trained checkpoint complete (${OUT_DIR}/eval_results.json), skipping training."
    else
        if [[ -d "${OUT_DIR}" ]]; then
            echo "  [WARN] ${OUT_DIR} exists but has no eval_results.json: partial run, retraining."
            echo "         Stale checkpoint-* subdirectories are not removed automatically;"
            echo "         delete the directory by hand if disk quota is tight."
        fi
        echo "  training ..."
        python finetune/finetune_tiny_small.py \
            --dataset "${DATASET}" \
            --seed "${SEED}" \
            --base-model "${BASE_MODEL}" \
            --output-dir "${OUT_DIR}"
    fi

    # inference_loop writes the raw CSV only after the whole split is transcribed
    # (partials go to a separate wer_<key>_partial.csv), so its presence is a valid
    # completion marker for this stage on its own.
    if [[ "${SKIP_TRAINED}" == "1" && -f "${RAW_CSV}" ]]; then
        echo "  transcripts present (${RAW_CSV}), skipping transcription."
    else
        echo "  transcribing eval split ..."
        DATASET="${DATASET}" MODEL_NAME="${RUN_KEY}" MODEL_SOURCE="${OUT_DIR}" \
            python finetune/evaluate_finetuned.py
    fi

    echo "  scoring ..."
    # Always re-scored rather than skipped: it is CPU-only, takes seconds, and is
    # idempotent, so there is nothing to save by guarding it and a partially scored
    # seed would otherwise need manual repair.
    # Per-seed keys are deliberately not in the registry, so they are scored by name.
    python normalize_and_score.py --dataset "${DATASET}" --models "${RUN_KEY}"

    echo "  seed ${SEED} done."
    echo
done

echo "All seeds finished. Aggregate with:"
echo "    python analysis/compare_seeds.py --dataset ${DATASET}"
