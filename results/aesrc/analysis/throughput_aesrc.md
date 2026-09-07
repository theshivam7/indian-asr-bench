# Offline throughput: aesrc

Best quality-valid batch size on the common duration-sorted workload, under the pre-registered gate. RTFx is audio seconds processed per wall-clock second; higher is better.

> **Read RTFx together with utterances/s.** The Whisper systems run the short-form Transformers path, which zero-pads every clip to 30 s, so their cost is per utterance and does not fall when clips get shorter. The NeMo systems pad to the longest clip in the batch, so their cost tracks real audio. RTFx divides by real audio seconds, which flatters the padded systems on short-clip corpora. Whisper's mean GPU utilization on the curated corpora is under 2%, so those numbers are largely bounded by CPU-side audio decode rather than by the A100.

| model_display | best_batch_size | best_rtfx_audio_s_per_s | best_rtfx_min | best_rtfx_max | batching_speedup_x | utterances_per_s | gpu_util_mean_pct | device_memory_peak_mib | estimated_gpu_wh_per_audio_hour | completion_latency_p95_s | best_wer_pct | wer_delta_pp_vs_batch1 | padded_rtfx_audio_s_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper Tiny | 128 | 66.755 | 66.497 | 66.783 | 1.793 | 14.726 | 1.96 | 4385.0 | 1.0889 | 8.7626 | 12.82 | 0.0379 | 441.785 |
| Whisper Base | 128 | 65.526 | 65.509 | 65.613 | 2.111 | 14.455 | 3.17 | 5671.0 | 1.0425 | 8.9239 | 9.7098 | 0.0 | 433.651 |
| Whisper Small | 128 | 62.309 | 62.191 | 62.422 | 2.631 | 13.745 | 6.78 | 11763.0 | 1.3492 | 9.3995 | 6.5807 | -0.019 | 412.361 |
| Whisper Medium | 128 | 56.24 | 56.129 | 56.303 | 3.424 | 12.407 | 16.75 | 34419.0 | 2.085 | 10.4151 | 4.8739 | 0.0 | 372.196 |
| Whisper Large-v3 | 64 | 48.287 | 45.374 | 48.315 | 3.704 | 10.652 | 27.75 | 31273.0 | 3.2149 | 6.0946 | 4.8739 | 0.0 | 319.563 |
| Whisper large-v3-turbo | 64 | 53.124 | 53.079 | 53.143 | 1.628 | 11.719 | 21.56 | 8427.0 | 2.4825 | 5.5156 | 5.4049 | 0.019 | 351.575 |
| Parakeet-TDT-0.6B-v2 | 128 | 1957.696 | 1897.155 | 2003.511 | 28.086 | 431.871 | 51.6 | 6789.0 | 0.1311 | 0.4063 | 5.4618 | -0.0758 | N/A |
| Parakeet-CTC-1.1B | 128 | 1492.29 | 1492.011 | 1529.921 | 33.384 | 329.202 | 61.29 | 7951.0 | 0.1735 | 0.5538 | 6.6566 | 0.038 | N/A |
| Qwen3-ASR-1.7B | 128 | 263.178 | 258.535 | 264.133 | 18.476 | 58.058 | 52.17 | 19189.0 | 0.6636 | 2.536 | 4.8929 | -0.0189 | N/A |
## Gate sensitivity

The pre-registered gate rejects any batch whose corpus WER moves more than 0.1 pp from batch 1 in either direction, and any batch that adds an empty hypothesis. Whisper pads to a fixed window so batching cannot move its output and the gate never binds; the NeMo systems pad dynamically, so it binds only on them. The columns below re-derive the selection with a one-sided tolerance of 0.5 pp (a batch that scores better than batch 1 is not treated as a failure). This is a post-hoc sensitivity check, not the pre-registered result; cite the table above.

No model's selection changes under the wider gate.

## Gate cost, every model

The same comparison with no quality filter at all: `tput_*` is the fastest batch measured for each model, and `gate_cost_x` is how much throughput the pre-registered gate gives up. Every Whisper row reads 1.00 on all three corpora: a fixed 30 s window makes Whisper's output batch-invariant, so the WER arm of the gate cannot bind on it. Values above 1.00 are therefore a cost borne only by the dynamically padded engines, which is why the gate is better read as a diagnostic than as a neutral selection rule. Values slightly below 1.00 are possible where the wider candidate set lets the within-1% tie rule pick a smaller batch; treat those as ties.

| model_display | best_batch_size | best_rtfx_audio_s_per_s | tput_batch_size | tput_rtfx_audio_s_per_s | tput_wer_delta_pp_vs_batch1 | gate_cost_x |
| --- | --- | --- | --- | --- | --- | --- |
| Whisper Tiny | 128 | 66.755 | 128 | 66.755 | 0.0379 | 1.0 |
| Whisper Base | 128 | 65.526 | 128 | 65.526 | 0.0 | 1.0 |
| Whisper Small | 128 | 62.309 | 128 | 62.309 | -0.019 | 1.0 |
| Whisper Medium | 128 | 56.24 | 128 | 56.24 | 0.0 | 1.0 |
| Whisper Large-v3 | 64 | 48.287 | 64 | 48.287 | 0.0 | 1.0 |
| Whisper large-v3-turbo | 64 | 53.124 | 64 | 53.124 | 0.019 | 1.0 |
| Parakeet-TDT-0.6B-v2 | 128 | 1957.696 | 128 | 1957.696 | -0.0758 | 1.0 |
| Parakeet-CTC-1.1B | 128 | 1492.29 | 128 | 1492.29 | 0.038 | 1.0 |
| Qwen3-ASR-1.7B | 128 | 263.178 | 128 | 263.178 | -0.0189 | 1.0 |

## Padded audio

RTFx divides by real audio seconds. The Whisper short-form path pads every clip to a fixed 30 s window, so it is charged for the corpus and actually processes the padded window. `padded_rtfx_audio_s_per_s` in the table above is RTFx times that padding multiplier, which is the rate the GPU actually sustained. It is blank for the NeMo and Qwen3 rows: NeMo pads to batch maximum, which is dynamic, and the Qwen3 backend does not record its windowing per batch, so neither is guessed.

Padding multiplier on this corpus: 6.62x (9 systems, 6 of them fixed-window).

Per-batch measurements for every model, including the ones the gate rejected and why, are in `throughput_aesrc_sweep.csv`.
