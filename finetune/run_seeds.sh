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

    if [[ "${SKIP_TRAINED}" == "1" && -f "${RAW_CSV}" ]]; then
        echo "  already scored (${RAW_CSV}), skipping. Use --force to redo."
        echo
        continue
    fi

    if [[ "${SKIP_TRAINED}" == "1" && -d "${OUT_DIR}" ]]; then
        echo "  checkpoint exists, skipping training."
    else
        echo "  training ..."
        python finetune/finetune_tiny_small.py \
            --dataset "${DATASET}" \
            --seed "${SEED}" \
            --base-model "${BASE_MODEL}" \
            --output-dir "${OUT_DIR}"
    fi

    echo "  transcribing eval split ..."
    DATASET="${DATASET}" MODEL_NAME="${RUN_KEY}" MODEL_SOURCE="${OUT_DIR}" \
        python finetune/evaluate_finetuned.py

    echo "  scoring ..."
    # Per-seed keys are deliberately not in the registry, so they are scored by name.
    python normalize_and_score.py --dataset "${DATASET}" --models "${RUN_KEY}"

    echo "  seed ${SEED} done."
    echo
done

echo "All seeds finished. Aggregate with:"
echo "    python analysis/compare_seeds.py --dataset ${DATASET}"
