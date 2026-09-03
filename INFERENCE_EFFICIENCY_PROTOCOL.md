# Inference-efficiency protocol

## What the two measurements mean

Keep both results; do not merge their labels.

1. **Single-stream latency** (existing `utils/efficiency.py`): batch size 1,
   per-clip end-to-end latency, RTF, and PyTorch peak memory. This answers how
   quickly one request finishes. It does **not** claim maximum GPU throughput.
2. **Offline saturated throughput** (new `utils/throughput.py`): common
   duration-sorted workload, native batched inference, batch-size sweep, repeated
   timings, quality gate, and sampled GPU telemetry. This answers how much audio
   one exclusive A100 can process when requests are already available.

This follows the same separation used by MLPerf Inference: SingleStream is a
latency scenario and Offline is a throughput scenario. It is not a production
server benchmark; server performance additionally requires controlled request
arrivals, concurrency, queueing latency, and a stated latency SLO.

## Official implementation choices

| Family | Measured runtime | Fixed decoding/configuration | Documentation basis |
|---|---|---|---|
| Whisper Tiny through large-v3-turbo | Hugging Face Transformers ASR pipeline, FP16, SDPA | English, transcription, greedy (`do_sample=False`, one beam), no timestamps, inputs at most 30 s and no external chunking | OpenAI's reference `transcribe()` is single-file. The official Transformers pipeline exposes documented list batching; its documentation also warns that batching must be benchmarked because it is not always faster. Runtime identity is recorded in every result. |
| Parakeet TDT 0.6B v2 and CTC 1.1B | NVIDIA NeMo `ASRModel.transcribe`, AMP FP16 | Checkpoint-default greedy decoding, no timestamps | NVIDIA's official transcription example supports `batch_size`, AMP, duration presorting, warmups, and RTFx. The TDT model card reports its own RTFx at batch 128 and explicitly says batch size and audio duration affect it. |
| Qwen3-ASR 1.7B | Official `qwen-asr` Transformers backend, BF16, SDPA | English, no timestamps, `max_new_tokens=512` | The official API accepts lists and chunks them by `max_inference_batch_size`; its recommended precision is BF16. vLLM is deliberately not used in the controlled comparison because it would add an optimized serving-runtime advantage unavailable in the reference paths of every family. |

Primary sources:

- [OpenAI Whisper README and A100 speed/VRAM table](https://github.com/openai/whisper/blob/main/README.md)
- [OpenAI Whisper reference transcription implementation](https://github.com/openai/whisper/blob/main/whisper/transcribe.py)
- [Hugging Face Transformers pipeline batching documentation](https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching)
- [NVIDIA Parakeet-TDT-0.6B-v2 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NVIDIA Parakeet-CTC-1.1B model card](https://huggingface.co/nvidia/parakeet-ctc-1.1b)
- [NVIDIA NeMo transcription and RTFx example](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/transcribe_speech.py)
- [Qwen3-ASR official repository](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR-1.7B official model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen3-ASR official inference implementation](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)
- [MLPerf inference audit metrics by scenario](https://github.com/mlcommons/inference_policies/blob/master/MLPerf_Audit_Guidelines.adoc)
- [PyTorch CUDA synchronization](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html), [allocated memory](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html), and [reserved memory](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html)

## Pre-registered workload and timing rules

- Hardware: one exclusive A100 40 GB node (`place=excl`), one GPU per process.
  NSCC's `g1` route allocates 16 CPUs and 110 GB host RAM for this one-GPU job;
  CPU math-library threads are fixed to those 16 allocated CPUs for every model.
- Dataset: 512 evaluation clips with non-empty normalized references and duration
  of at most 30 seconds,
  selected with NumPy seed 42. The short-form limit prevents Whisper alone from
  entering an external/experimental chunking path during a batch comparison.
  Exact row indices, IDs, dataset revision, duration, and SHA-256 fingerprints are
  stored in every result. The aggregator rejects mismatched workloads.
- Audio: every selected clip is converted once to the same mono 16-bit 16 kHz WAV
  before timing. The timed region includes WAV read/decode, feature extraction,
  GPU inference, and text decoding. Dataset download, model load, and WAV
  conversion are separate fields.
- Ordering: the selected clips are sorted by `(duration, clip_id)` for every
  model. This gives each system the same padding-efficient offline queue; it is
  not used for the single-stream result.
- Batch sweep: `1,2,4,8,16,32,64,128`, stopping after the first OOM or failure.
- Warmup/repetition: 3 full batches per batch size covering the shortest, median,
  and longest duration buckets, then 3 complete timed passes over the 512 clips.
  CUDA is synchronized around every timed batch. Report the
  median, minimum, maximum, and coefficient of variation across the three passes;
  retain all individual measurements.
- Selection: find the highest median RTFx among quality-valid configurations,
  then choose the smallest batch within 1% of that maximum. This avoids choosing
  a much larger, higher-latency batch for measurement noise. Do not choose a
  batch merely because it fills more of the 40 GB card.

## Metrics to report

Primary paper metrics:

- **RTFx / audio seconds per wall-clock second**: total audio duration divided by
  processing time; higher is better.
- **Batching speedup**: best valid RTFx divided by batch-1 RTFx in the same runtime.
- **Utterances/s**: useful alongside RTFx because corpus clip lengths differ.
- **Completion latency p50/p95** at the selected batch: throughput has a latency
  cost, so report both.
- **Mean and p95 GPU SM utilization** sampled every 250 ms.
- **Peak device memory** from NVIDIA telemetry plus PyTorch allocated and reserved
  peaks. Reserved memory is not the same as utilization.
- **Quality at batch 1 and selected batch**: corpus WER/CER, empty outputs, and
  absolute WER change.

Secondary/reproducibility metrics: model-load time, mean power, approximate GPU
energy per pass, Wh per transcribed audio hour, audio seconds per GPU joule, GPU
name/UUID context, total VRAM, driver, CUDA/Torch/runtime
versions, node, git commit, OOM point, model ID, precision, attention backend,
language, token limit, and timestamp setting.

## Quality gate

A batch setting is eligible only when its corpus WER differs from the same
runtime's batch-1 WER by at most **0.10 absolute percentage point** and it creates
no additional empty hypotheses. Transcript hashes are also saved to show whether
outputs are exactly identical. Every timed repeat must produce the same transcript
hash. This prevents a faster or unstable configuration from winning by silently
changing or dropping predictions.

The 0.10-point tolerance is fixed before execution and is small enough to catch
meaningful decode drift on 512 clips while allowing harmless floating-point
differences. The report must show the observed delta rather than only `pass/fail`.

## Claims this supports (and does not support)

Supported: single-request latency at batch 1; maximum quality-preserving offline
throughput under the tested batch sweep; batching benefit; GPU occupancy, memory,
and approximate device energy on A100 40 GB.

Not supported: price in dollars without an explicit NSCC/GPU hourly rate;
multi-client production-server QPS or tail latency; TensorRT/Riva/vLLM-optimized
runtime comparisons; total server energy; performance on a different GPU. These
can be separate experiments, but must not be mixed into this controlled table.
