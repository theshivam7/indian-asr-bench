# Inference efficiency: aesrc

Measured on NVIDIA A100-SXM4-40GB, driver 570.124.06, torch 2.5.1, 2.5.1+cu124, CUDA 11.8, 12.4.

Protocol: 200 clips sampled with seed 42 (fingerprint `a0aa7f0e3d06`), 3 untimed warmup clips, batch size 1.

RTF convention: processing_seconds / audio_seconds; lower is faster, <1 is faster than real time.

> **Comparability warnings**
> - Runs span 2 CUDA runtime versions (CUDA 11.8: base, large, large_v3_turbo, medium, small, tiny; CUDA 12.4: parakeet, parakeet_ctc, qwen3). The engines cannot share an environment, so this is expected; disclose it with the table rather than treating small cross-engine timing gaps as architectural.
> - Precision is not recorded in these results. Each engine runs at its reference implementation's default, which is not the same across engines (fp32-resident weights for the Whisper and Parakeet drivers, bf16 for Qwen3-ASR). Peak GPU memory is therefore not a like-for-like comparison; treat it as each system's default footprint, not as an architecture-controlled measurement.
> - cuDNN state during inference is not recorded. Results produced before the cuDNN fix in the Parakeet and Qwen3 drivers were timed with cuDNN disabled, while the Whisper runs had it enabled; those Parakeet and Qwen3 latencies are pessimistic. Re-run to remove this caveat.

| Model | Params | Arch | RTF | RTF p50 | Audio-s/s | Lat. p50 (s) | Lat. p95 (s) | Peak GPU (MiB) | Load (s) |
|---|---|---|---|---|---|---|---|---|---|
| Whisper Tiny | 39M | enc_dec | 0.0464 | 0.0474 | 21.556 | 0.2016 | 0.2463 | 205.4 | 0.432 |
| Whisper Base | 74M | enc_dec | 0.0593 | 0.0601 | 16.877 | 0.2606 | 0.3168 | 359.6 | 0.749 |
| Whisper Small | 244M | enc_dec | 0.0814 | 0.0833 | 12.282 | 0.3581 | 0.4326 | 1066.5 | 2.323 |
| Whisper Medium | 769M | enc_dec | 0.1195 | 0.1206 | 8.368 | 0.5207 | 0.6734 | 3171.8 | 6.65 |
| Whisper Large-v3 | 1.5B | enc_dec | 0.1457 | 0.148 | 6.865 | 0.6336 | 0.8397 | 6395.6 | 12.482 |
| Whisper large-v3-turbo | 809M | enc_dec | 0.0629 | 0.064 | 15.903 | 0.2759 | 0.3636 | 3328.7 | 6.294 |
| Parakeet-TDT-0.6B-v2 | 600M | transducer | 0.0127 | 0.0128 | 79.009 | 0.0548 | 0.0599 | 2456.9 | 8.946 |
| Parakeet-CTC-1.1B | 1.1B | ctc | 0.02 | 0.0193 | 50.036 | 0.0822 | 0.0901 | 4122.8 | 12.761 |
| Qwen3-ASR-1.7B | 1.7B | llm | 0.0798 | 0.0796 | 12.528 | 0.3639 | 0.4814 | 3955 | 11.259 |

Note: Parakeet-TDT-0.6B-v2, Parakeet-CTC-1.1B support batched inference but were measured one clip at a time, so every engine sees the same single-stream batch size. Batch size is the only thing equalized here; see the comparability warnings above for what is not. Throughput under batching is higher than reported here.
