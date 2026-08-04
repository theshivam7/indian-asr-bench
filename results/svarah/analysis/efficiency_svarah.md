# Inference efficiency: svarah

Measured on NVIDIA A100-SXM4-40GB, driver 570.124.06, torch 2.5.1, 2.5.1+cu124, CUDA 11.8, 12.4.

Protocol: 200 clips sampled with seed 42 (fingerprint `3313dc8c2a93`), 3 untimed warmup clips, batch size 1.

RTF convention: processing_seconds / audio_seconds; lower is faster, <1 is faster than real time.

> **Comparability warnings**
> - Runs span 2 CUDA runtime versions (CUDA 11.8: base, large, large_v3_turbo, medium, small, tiny; CUDA 12.4: parakeet, parakeet_ctc, qwen3). The engines cannot share an environment, so this is expected; disclose it with the table rather than treating small cross-engine timing gaps as architectural.

| Model | Params | Arch | RTF | RTF p50 | Audio-s/s | Lat. p50 (s) | Lat. p95 (s) | Peak GPU (MiB) | Load (s) |
|---|---|---|---|---|---|---|---|---|---|
| Whisper Tiny | 39M | enc_dec | 0.0477 | 0.0513 | 20.977 | 0.2064 | 0.3401 | 210.2 | 1.041 |
| Whisper Base | 74M | enc_dec | 0.0498 | 0.0557 | 20.099 | 0.2185 | 0.3801 | 365.6 | 0.817 |
| Whisper Small | 244M | enc_dec | 0.0722 | 0.0793 | 13.855 | 0.3175 | 0.5662 | 1073.5 | 2.292 |
| Whisper Medium | 769M | enc_dec | 0.1191 | 0.1268 | 8.394 | 0.5047 | 0.9168 | 3190.7 | 6.239 |
| Whisper Large-v3 | 1.5B | enc_dec | 0.1488 | 0.1509 | 6.72 | 0.597 | 1.1918 | 6437.9 | 11.968 |
| Whisper large-v3-turbo | 809M | enc_dec | 0.0511 | 0.0555 | 19.588 | 0.2308 | 0.3306 | 3332.9 | 6.561 |
| Parakeet-TDT-0.6B-v2 | 600M | transducer | 0.0121 | 0.0138 | 82.65 | 0.0559 | 0.0632 | 2493.7 | 10.235 |
| Parakeet-CTC-1.1B | 1.1B | ctc | 0.0176 | 0.0206 | 56.757 | 0.0816 | 0.0866 | 4146.1 | 13.823 |
| Qwen3-ASR-1.7B | 1.7B | llm | 0.0929 | 0.104 | 10.759 | 0.4047 | 0.9173 | 4008.7 | 12.15 |

Note: Parakeet-TDT-0.6B-v2, Parakeet-CTC-1.1B support batched inference but were measured one clip at a time, so every engine is timed under identical single-stream conditions. Throughput under batching is higher than reported here.
