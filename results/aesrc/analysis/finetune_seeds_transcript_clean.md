# Multi-seed fine-tuning: aesrc (transcript_clean)

Each size trained repeatedly with only the seed changed, then scored through the identical HF pipeline as its own pretrained baseline.

The spread below is **across-seed** variation of the delta: how much the result moves when training is repeated. It is a different quantity from the within-run bootstrap CI in `finetune_capacity_summary.csv`, which describes sampling error over test clips. Report both, and do not pool them.

| Size | Params | Seeds | Baseline WER | FT WER (mean) | Δ mean (pp) | Δ SD (pp) | Δ min | Δ max |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Whisper Tiny | 39M | 6 | 17.447% | 10.597% | -6.85 | 1.029 | -7.337 | -4.751 |
| Whisper Small | 244M | 6 | 7.222% | 5.577% | -1.645 | 0.145 | -1.84 | -1.416 |
| Whisper Medium | 769M | 6 | 5.628% | 4.408% | -1.221 | 0.115 | -1.324 | -0.997 |

## Per-seed runs

The individual runs behind the means above. Listed so the aggregate is checkable: whether every run improved on its baseline, and how the SD was computed, are both questions the summary table cannot answer on its own.

| Size | Seed | Baseline WER | FT WER | Δ (pp) |
|---|:---:|:---:|:---:|:---:|
| Whisper Tiny | 42 | 17.447% | 12.695% | -4.751 |
| Whisper Tiny | 43 | 17.447% | 10.15% | -7.296 |
| Whisper Tiny | 44 | 17.447% | 10.179% | -7.268 |
| Whisper Tiny | 45 | 17.447% | 10.225% | -7.222 |
| Whisper Tiny | 46 | 17.447% | 10.11% | -7.337 |
| Whisper Tiny | 47 | 17.447% | 10.219% | -7.228 |
| Whisper Small | 42 | 7.222% | 5.583% | -1.639 |
| Whisper Small | 43 | 7.222% | 5.542% | -1.679 |
| Whisper Small | 44 | 7.222% | 5.657% | -1.565 |
| Whisper Small | 45 | 7.222% | 5.806% | -1.416 |
| Whisper Small | 46 | 7.222% | 5.491% | -1.731 |
| Whisper Small | 47 | 7.222% | 5.382% | -1.84 |
| Whisper Medium | 42 | 5.628% | 4.379% | -1.25 |
| Whisper Medium | 43 | 5.628% | 4.631% | -0.997 |
| Whisper Medium | 44 | 5.628% | 4.402% | -1.227 |
| Whisper Medium | 45 | 5.628% | 4.385% | -1.244 |
| Whisper Medium | 46 | 5.628% | 4.304% | -1.324 |
| Whisper Medium | 47 | 5.628% | 4.344% | -1.284 |
