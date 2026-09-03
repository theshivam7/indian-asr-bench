#!/bin/bash
# Submit all nine pretrained systems. Each job runs all three datasets so queue
# usage stays manageable while every model/dataset result remains separate.
set -euo pipefail

: "${PROJECT:?export PROJECT=<NSCC project id>}"
DATASETS=${DATASETS:-tie:svarah:aesrc}
GIT_COMMIT=$(git rev-parse HEAD)
SUBMIT_DIR=$(pwd -P)
mkdir -p "${SUBMIT_DIR}/logs"
WHISPER_THROUGHPUT_ENV=${WHISPER_THROUGHPUT_ENV:-whisper_throughput}
PARAKEET_ENV=${PARAKEET_ENV:-parakeet}
QWEN3_ENV=${QWEN3_ENV:-qwen3}

# Result files may legitimately be present or modified. Every other tracked or
# untracked file must match GIT_COMMIT so local modules cannot silently shadow
# the committed implementation recorded in the output.
DIRTY=$(git status --porcelain --untracked-files=all -- . ':(exclude)results/**')
if [ -n "${DIRTY}" ]; then
    echo "[FATAL] code/config differs from ${GIT_COMMIT}; commit or remove it before submitting:" >&2
    echo "${DIRTY}" >&2
    exit 1
fi

# PBS snapshots this shell script but the Python modules remain in the shared
# checkout. Hash every runtime source/config file now and verify it again on the
# compute node, preventing a later git pull from falsifying run provenance.
source_digest() {
    {
        find utils throughput -maxdepth 1 -type f \
          \( -name '*.py' -o -name '*.txt' -o -name '*.sh' \) -print0
        printf '%s\0' hpc/job_throughput.pbs parakeet/requirements.txt qwen3/requirements.txt
    } | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
}
SOURCE_SHA256=$(source_digest)

# Fail on the login node before consuming a queued GPU job. Importing packages
# and reading metadata is lightweight; model loading/inference remains inside PBS.
check_common() {
    env_name=$1
    conda run -n "${env_name}" python -c \
      "import torch; from importlib.metadata import version; expected={'torch':'2.5.1','datasets':'4.8.5','pandas':'2.2.3','numpy':'1.26.4','soundfile':'0.13.1','librosa':'0.11.0','jiwer':'4.0.0','num2words':'0.5.14'}; actual={k:version(k) for k in expected}; assert actual==expected,(actual,expected); assert torch.version.cuda, 'CUDA-enabled torch build required'; print('${env_name}',actual,'torch_cuda',torch.version.cuda)"
}

echo "=== preflight: throughput environments ==="
check_common "${WHISPER_THROUGHPUT_ENV}"
conda run -n "${WHISPER_THROUGHPUT_ENV}" ffmpeg -version >/dev/null
conda run -n "${WHISPER_THROUGHPUT_ENV}" python -c \
  "from importlib.metadata import version; assert version('transformers')=='4.57.6' and version('accelerate')=='1.12.0' and version('safetensors')=='0.6.2'"
check_common "${PARAKEET_ENV}"
conda run -n "${PARAKEET_ENV}" python -c \
  "from importlib.metadata import version; assert version('nemo_toolkit')=='2.3.0'"
check_common "${QWEN3_ENV}"
conda run -n "${QWEN3_ENV}" python -c \
  "from importlib.metadata import version; assert version('qwen-asr')=='0.0.6' and version('transformers')=='4.57.6'"

WHISPER_CUDA=$(conda run -n "${WHISPER_THROUGHPUT_ENV}" python -c \
  "import torch; print(f'{torch.version.cuda}|{torch.backends.cudnn.version()}')" | tr -d '\r\n')
PARAKEET_CUDA=$(conda run -n "${PARAKEET_ENV}" python -c \
  "import torch; print(f'{torch.version.cuda}|{torch.backends.cudnn.version()}')" | tr -d '\r\n')
QWEN3_CUDA=$(conda run -n "${QWEN3_ENV}" python -c \
  "import torch; print(f'{torch.version.cuda}|{torch.backends.cudnn.version()}')" | tr -d '\r\n')
if [ "${WHISPER_CUDA}" != "${PARAKEET_CUDA}" ] || [ "${WHISPER_CUDA}" != "${QWEN3_CUDA}" ]; then
    echo "[FATAL] CUDA/cuDNN builds differ across environments:" >&2
    echo "  whisper=${WHISPER_CUDA} parakeet=${PARAKEET_CUDA} qwen3=${QWEN3_CUDA}" >&2
    exit 1
fi
echo "common CUDA|cuDNN=${WHISPER_CUDA}"
echo "=== preflight passed; submitting nine jobs ==="

submit() {
    engine=$1
    model=$2
    qsub -P "${PROJECT}" \
      -o "${SUBMIT_DIR}/logs/pbs_${engine}_${model}_${GIT_COMMIT:0:7}.out" \
      -v "ENGINE=${engine},MODEL=${model},DATASETS=${DATASETS},GIT_COMMIT=${GIT_COMMIT},SOURCE_SHA256=${SOURCE_SHA256},WHISPER_THROUGHPUT_ENV=${WHISPER_THROUGHPUT_ENV},PARAKEET_ENV=${PARAKEET_ENV},QWEN3_ENV=${QWEN3_ENV}" \
      hpc/job_throughput.pbs
}

for model in tiny base small medium large large_v3_turbo; do
    submit whisper "${model}"
done
submit parakeet parakeet
submit parakeet parakeet_ctc
submit qwen3 qwen3
