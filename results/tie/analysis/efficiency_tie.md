# Inference efficiency: tie

Measured on NVIDIA A100-SXM4-40GB, driver 570.124.06, torch 2.5.1, 2.5.1+cu124, CUDA 11.8, 12.4.

Protocol: 200 clips sampled with seed 42 (fingerprint `46c6f70a710f`), 3 untimed warmup clips, batch size 1.

RTF convention: processing_seconds / audio_seconds; lower is faster, <1 is faster than real time.

> **Comparability warnings**
> - Runs span 2 CUDA runtime versions (CUDA 11.8: base, large, large_v3_turbo, medium, small, tiny; CUDA 12.4: parakeet, parakeet_ctc, qwen3). The engines cannot share an environment, so this is expected; disclose it with the table rather than treating small cross-engine timing gaps as architectural.

| Model | Params | Arch | RTF | RTF p50 | Audio-s/s | Lat. p50 (s) | Lat. p95 (s) | Peak GPU (MiB) | Load (s) |
|---|---|---|---|---|---|---|---|---|---|
| Whisper Tiny | 39M | enc_dec | 0.0214 | 0.0194 | 46.717 | 0.4851 | 0.6681 | 227 | 1.234 |
| Whisper Base | 74M | enc_dec | 0.0267 | 0.026 | 37.394 | 0.6498 | 0.8774 | 384.7 | 1.316 |
| Whisper Small | 244M | enc_dec | 0.0379 | 0.0383 | 26.371 | 0.9543 | 1.312 | 1094.7 | 2.03 |
| Whisper Medium | 769M | enc_dec | 0.0798 | 0.0788 | 12.525 | 1.9455 | 2.6116 | 3207.1 | 7.296 |
| Whisper Large-v3 | 1.5B | enc_dec | 0.1051 | 0.1024 | 9.518 | 2.5304 | 3.5524 | 6442.3 | 13.437 |
| Whisper large-v3-turbo | 809M | enc_dec | 0.0252 | 0.0203 | 39.637 | 0.4996 | 1.3522 | 3361 | 84.701 |
| Parakeet-TDT-0.6B-v2 | 600M | transducer | 0.0047 | 0.0038 | 214.745 | 0.0914 | 0.1194 | 2863.9 | 9.328 |
| Parakeet-CTC-1.1B | 1.1B | ctc | 0.0044 | 0.0044 | 225.362 | 0.1049 | 0.1373 | 4382.4 | 12.341 |
| Qwen3-ASR-1.7B | 1.7B | llm | 0.0736 | 0.0737 | 13.588 | 1.8575 | 2.593 | 4378.2 | 13.98 |

Note: Parakeet-TDT-0.6B-v2, Parakeet-CTC-1.1B support batched inference but were measured one clip at a time, so every engine is timed under identical single-stream conditions. Throughput under batching is higher than reported here.
