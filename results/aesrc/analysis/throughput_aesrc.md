# Offline throughput: aesrc

> **Incomplete panel.** 7 of 9 systems. Missing: Whisper large-v3-turbo, Qwen3-ASR-1.7B. Do not read this as a full comparison until the remaining runs land.

Best quality-valid batch size on the common duration-sorted workload, under the pre-registered gate. RTFx is audio seconds processed per wall-clock second; higher is better.

> **Read RTFx together with utterances/s.** The Whisper systems run the short-form Transformers path, which zero-pads every clip to 30 s, so their cost is per utterance and does not fall when clips get shorter. The NeMo systems pad to the longest clip in the batch, so their cost tracks real audio. RTFx divides by real audio seconds, which flatters the padded systems on short-clip corpora. Whisper's mean GPU utilization on the curated corpora is under 2%, so those numbers are largely bounded by CPU-side audio decode rather than by the A100.

| model_display | best_batch_size | best_rtfx_audio_s_per_s | best_rtfx_min | best_rtfx_max | batching_speedup_x | utterances_per_s | gpu_util_mean_pct | device_memory_peak_mib | estimated_gpu_wh_per_audio_hour | completion_latency_p95_s | best_wer_pct | wer_delta_pp_vs_batch1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper Tiny | 128 | 66.755 | 66.497 | 66.783 | 1.793 | 14.726 | 1.96 | 4385.0 | 1.0889 | 8.7626 | 12.82 | 0.0379 |
| Whisper Base | 128 | 65.526 | 65.509 | 65.613 | 2.111 | 14.455 | 3.17 | 5671.0 | 1.0425 | 8.9239 | 9.7098 | 0.0 |
| Whisper Small | 128 | 62.309 | 62.191 | 62.422 | 2.631 | 13.745 | 6.78 | 11763.0 | 1.3492 | 9.3995 | 6.5807 | -0.019 |
| Whisper Medium | 128 | 56.24 | 56.129 | 56.303 | 3.424 | 12.407 | 16.75 | 34419.0 | 2.085 | 10.4151 | 4.8739 | 0.0 |
| Whisper Large-v3 | 64 | 48.287 | 45.374 | 48.315 | 3.704 | 10.652 | 27.75 | 31273.0 | 3.2149 | 6.0946 | 4.8739 | 0.0 |
| Parakeet-TDT-0.6B-v2 | 128 | 1957.696 | 1897.155 | 2003.511 | 28.086 | 431.871 | 51.6 | 6789.0 | 0.1311 | 0.4063 | 5.4618 | -0.0758 |
| Parakeet-CTC-1.1B | 128 | 1492.29 | 1492.011 | 1529.921 | 33.384 | 329.202 | 61.29 | 7951.0 | 0.1735 | 0.5538 | 6.6566 | 0.038 |
## Gate sensitivity

The pre-registered gate rejects any batch whose corpus WER moves more than 0.1 pp from batch 1 in either direction, and any batch that adds an empty hypothesis. Whisper pads to a fixed window so batching cannot move its output and the gate never binds; the NeMo systems pad dynamically, so it binds only on them. The columns below re-derive the selection with a one-sided tolerance of 0.5 pp (a batch that scores better than batch 1 is not treated as a failure). This is a post-hoc sensitivity check, not the pre-registered result; cite the table above.

No model's selection changes under the wider gate.

Per-batch measurements for every model, including the ones the gate rejected and why, are in `throughput_aesrc_sweep.csv`.
