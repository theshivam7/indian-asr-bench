# Offline throughput: svarah

Best quality-valid batch size on the common duration-sorted workload, under the pre-registered gate. RTFx is audio seconds processed per wall-clock second; higher is better.

> **Read RTFx together with utterances/s.** The Whisper systems run the short-form Transformers path, which zero-pads every clip to 30 s, so their cost is per utterance and does not fall when clips get shorter. The NeMo systems pad to the longest clip in the batch, so their cost tracks real audio. RTFx divides by real audio seconds, which flatters the padded systems on short-clip corpora. Whisper's mean GPU utilization on the curated corpora is under 2%, so those numbers are largely bounded by CPU-side audio decode rather than by the A100.

| model_display | best_batch_size | best_rtfx_audio_s_per_s | best_rtfx_min | best_rtfx_max | batching_speedup_x | utterances_per_s | gpu_util_mean_pct | device_memory_peak_mib | estimated_gpu_wh_per_audio_hour | completion_latency_p95_s | best_wer_pct | wer_delta_pp_vs_batch1 | padded_rtfx_audio_s_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper Tiny | 128 | 78.742 | 78.674 | 78.993 | 1.877 | 14.863 | 1.77 | 4385.0 | 0.9411 | 8.9917 | 19.5105 | -0.0177 | 445.916 |
| Whisper Base | 128 | 76.919 | 76.458 | 76.983 | 2.185 | 14.519 | 3.04 | 5671.0 | 0.8855 | 9.2132 | 13.8347 | -0.0355 | 435.592 |
| Whisper Small | 128 | 72.602 | 72.391 | 72.683 | 2.664 | 13.704 | 7.54 | 11763.0 | 1.2452 | 9.9319 | 10.2519 | 0.0178 | 411.145 |
| Whisper Medium | 64 | 64.057 | 63.979 | 64.126 | 3.516 | 12.091 | 18.22 | 18195.0 | 1.8921 | 6.0866 | 7.8751 | -0.0178 | 362.755 |
| Whisper Large-v3 | 64 | 55.668 | 55.495 | 55.819 | 3.844 | 10.508 | 29.52 | 31273.0 | 2.9266 | 7.1769 | 6.9173 | 0.0354 | 315.248 |
| Whisper large-v3-turbo | 64 | 62.205 | 62.057 | 62.297 | 1.682 | 11.742 | 21.55 | 8429.0 | 2.0784 | 5.7618 | 8.1057 | 0.0177 | 352.267 |
| Parakeet-TDT-0.6B-v2 | 8 | 537.089 | 495.734 | 544.963 | 6.349 | 101.38 | 40.81 | 3777.0 | 0.205 | 0.1081 | 12.9301 | 0.0177 | N/A |
| Parakeet-CTC-1.1B | 4 | 201.897 | 189.837 | 202.808 | 3.765 | 38.11 | 36.76 | 4871.0 | 0.4644 | 0.1108 | 15.2182 | 0.0532 | N/A |
| Qwen3-ASR-1.7B | 64 | 202.375 | 202.322 | 204.624 | 13.597 | 38.2 | 62.52 | 18163.0 | 0.9313 | 4.3377 | 11.5466 | -0.071 | N/A |
## Gate sensitivity

The pre-registered gate rejects any batch whose corpus WER moves more than 0.1 pp from batch 1 in either direction, and any batch that adds an empty hypothesis. Whisper pads to a fixed window so batching cannot move its output and the gate never binds; the NeMo systems pad dynamically, so it binds only on them. The columns below re-derive the selection with a one-sided tolerance of 0.5 pp (a batch that scores better than batch 1 is not treated as a failure). This is a post-hoc sensitivity check, not the pre-registered result; cite the table above.

| model_display | best_batch_size | best_rtfx_audio_s_per_s | sens_batch_size | sens_rtfx_audio_s_per_s | sens_wer_delta_pp_vs_batch1 | sens_vs_published_x |
| --- | --- | --- | --- | --- | --- | --- |
| Parakeet-TDT-0.6B-v2 | 8 | 537.089 | 64 | 1580.31 | 0.1064 | 2.942 |
| Parakeet-CTC-1.1B | 4 | 201.897 | 64 | 1175.276 | 0.1064 | 5.821 |

## Gate cost, every model

The same comparison with no quality filter at all: `tput_*` is the fastest batch measured for each model, and `gate_cost_x` is how much throughput the pre-registered gate gives up. Every Whisper row reads 1.00 on all three corpora: a fixed 30 s window makes Whisper's output batch-invariant, so the WER arm of the gate cannot bind on it. Values above 1.00 are therefore a cost borne only by the dynamically padded engines, which is why the gate is better read as a diagnostic than as a neutral selection rule. Values slightly below 1.00 are possible where the wider candidate set lets the within-1% tie rule pick a smaller batch; treat those as ties.

| model_display | best_batch_size | best_rtfx_audio_s_per_s | tput_batch_size | tput_rtfx_audio_s_per_s | tput_wer_delta_pp_vs_batch1 | gate_cost_x |
| --- | --- | --- | --- | --- | --- | --- |
| Whisper Tiny | 128 | 78.742 | 128 | 78.742 | -0.0177 | 1.0 |
| Whisper Base | 128 | 76.919 | 128 | 76.919 | -0.0355 | 1.0 |
| Whisper Small | 128 | 72.602 | 128 | 72.602 | 0.0178 | 1.0 |
| Whisper Medium | 64 | 64.057 | 64 | 64.057 | -0.0178 | 1.0 |
| Whisper Large-v3 | 64 | 55.668 | 64 | 55.668 | 0.0354 | 1.0 |
| Whisper large-v3-turbo | 64 | 62.205 | 64 | 62.205 | 0.0177 | 1.0 |
| Parakeet-TDT-0.6B-v2 | 8 | 537.089 | 64 | 1580.31 | 0.1064 | 2.942 |
| Parakeet-CTC-1.1B | 4 | 201.897 | 64 | 1175.276 | 0.1064 | 5.821 |
| Qwen3-ASR-1.7B | 64 | 202.375 | 64 | 202.375 | -0.071 | 1.0 |

## Padded audio

RTFx divides by real audio seconds. The Whisper short-form path pads every clip to a fixed 30 s window, so it is charged for the corpus and actually processes the padded window. `padded_rtfx_audio_s_per_s` in the table above is RTFx times that padding multiplier, which is the rate the GPU actually sustained. It is blank for the NeMo and Qwen3 rows: NeMo pads to batch maximum, which is dynamic, and the Qwen3 backend does not record its windowing per batch, so neither is guessed.

Padding multiplier on this corpus: 5.66x (9 systems, 6 of them fixed-window).

Per-batch measurements for every model, including the ones the gate rejected and why, are in `throughput_svarah_sweep.csv`.
