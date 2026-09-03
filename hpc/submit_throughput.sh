#!/bin/bash
# Submit all nine pretrained systems. Each job runs all three datasets so queue
# usage stays manageable while every model/dataset result remains separate.
set -eu

: "${PROJECT:?export PROJECT=<NSCC project id>}"
DATASETS=${DATASETS:-tie:svarah:aesrc}
GIT_COMMIT=$(git rev-parse HEAD)

# Result files may legitimately be present or modified. Every other tracked or
# untracked file must match GIT_COMMIT so local modules cannot silently shadow
# the committed implementation recorded in the output.
DIRTY=$(git status --porcelain --untracked-files=all -- . ':(exclude)results/**')
if [ -n "${DIRTY}" ]; then
    echo "[FATAL] code/config differs from ${GIT_COMMIT}; commit or remove it before submitting:" >&2
    echo "${DIRTY}" >&2
    exit 1
fi

submit() {
    engine=$1
    model=$2
    qsub -P "${PROJECT}" \
      -v "ENGINE=${engine},MODEL=${model},DATASETS=${DATASETS},GIT_COMMIT=${GIT_COMMIT}" \
      hpc/job_throughput.pbs
}

for model in tiny base small medium large large_v3_turbo; do
    submit whisper "${model}"
done
submit parakeet parakeet
submit parakeet parakeet_ctc
submit qwen3 qwen3
