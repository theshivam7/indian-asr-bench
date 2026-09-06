# Offline throughput: tie

> **Incomplete panel.** 8 of 9 systems. Missing: Qwen3-ASR-1.7B. Do not read this as a full comparison until the remaining runs land.

Best quality-valid batch size on the common duration-sorted workload, under the pre-registered gate. RTFx is audio seconds processed per wall-clock second; higher is better.

> **Read RTFx together with utterances/s.** The Whisper systems run the short-form Transformers path, which zero-pads every clip to 30 s, so their cost is per utterance and does not fall when clips get shorter. The NeMo systems pad to the longest clip in the batch, so their cost tracks real audio. RTFx divides by real audio seconds, which flatters the padded systems on short-clip corpora. Whisper's mean GPU utilization on the curated corpora is under 2%, so those numbers are largely bounded by CPU-side audio decode rather than by the A100.

| model_display | best_batch_size | best_rtfx_audio_s_per_s | best_rtfx_min | best_rtfx_max | batching_speedup_x | utterances_per_s | gpu_util_mean_pct | device_memory_peak_mib | estimated_gpu_wh_per_audio_hour | completion_latency_p95_s | best_wer_pct | wer_delta_pp_vs_batch1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper Tiny | 128 | 291.156 | 290.604 | 291.826 | 3.777 | 12.548 | 7.78 | 4387.0 | 0.3037 | 11.061 | 20.2096 | -0.0188 |
| Whisper Base | 128 | 289.366 | 289.31 | 289.949 | 4.493 | 12.471 | 10.1 | 5673.0 | 0.2982 | 11.9203 | 17.1669 | 0.0338 |
| Whisper Small | 128 | 279.1 | 277.774 | 279.503 | 6.725 | 12.029 | 13.84 | 11763.0 | 0.3955 | 10.8749 | 13.4405 | 0.0375 |
| Whisper Medium | 128 | 235.633 | 235.408 | 235.841 | 9.716 | 10.155 | 28.62 | 34419.0 | 0.673 | 12.9078 | 12.5202 | -0.0113 |
| Whisper Large-v3 | 128 | 192.765 | 192.584 | 193.054 | 10.198 | 8.308 | 40.49 | 38155.0 | 1.0217 | 15.8089 | 12.9597 | -0.0263 |
| Whisper large-v3-turbo | 128 | 249.957 | 242.461 | 250.017 | 3.263 | 10.773 | 22.62 | 14631.0 | 0.5467 | 12.1258 | 13.5757 | 0.0413 |
| Parakeet-TDT-0.6B-v2 | 128 | 2251.261 | 2203.297 | 2251.505 | 8.294 | 97.024 | 60.59 | 16257.0 | 0.104 | 1.6044 | 15.2098 | 0.0939 |
| Parakeet-CTC-1.1B | 1 | 228.145 | 217.468 | 228.36 | 1.0 | 9.833 | 38.11 | 4737.0 | 0.4304 | 0.1031 | 17.1744 | 0.0 |
## Gate sensitivity

The pre-registered gate rejects any batch whose corpus WER moves more than 0.1 pp from batch 1 in either direction, and any batch that adds an empty hypothesis. Whisper pads to a fixed window so batching cannot move its output and the gate never binds; the NeMo systems pad dynamically, so it binds only on them. The columns below re-derive the selection with a one-sided tolerance of 0.5 pp (a batch that scores better than batch 1 is not treated as a failure). This is a post-hoc sensitivity check, not the pre-registered result; cite the table above.

| model_display | best_batch_size | best_rtfx_audio_s_per_s | sens_batch_size | sens_rtfx_audio_s_per_s | sens_wer_delta_pp_vs_batch1 | sens_vs_published_x |
| --- | --- | --- | --- | --- | --- | --- |
| Parakeet-CTC-1.1B | 1 | 228.145 | 64 | 1719.067 | 0.3306 | 7.535 |

Per-batch measurements for every model, including the ones the gate rejected and why, are in `throughput_tie_sweep.csv`.
