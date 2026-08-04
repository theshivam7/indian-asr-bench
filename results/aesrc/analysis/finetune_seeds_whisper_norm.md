# Multi-seed fine-tuning: aesrc (whisper_norm)

Each size trained repeatedly with only the seed changed, then scored through the identical HF pipeline as its own pretrained baseline.

The spread below is **across-seed** variation of the delta: how much the result moves when training is repeated. It is a different quantity from the within-run bootstrap CI in `finetune_capacity_summary.csv`, which describes sampling error over test clips. Report both, and do not pool them.

| Size | Params | Seeds | Baseline WER | FT WER (mean) | Δ mean (pp) | Δ SD (pp) | Δ min | Δ max |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Whisper Tiny | 39M | 6 | 16.967% | 9.854% | -7.114 | 0.043 | -7.173 | -7.071 |
| Whisper Small | 244M | 6 | 6.905% | 5.248% | -1.657 | 0.132 | -1.806 | -1.453 |
| Whisper Medium | 769M | 6 | 5.259% | 4.106% | -1.153 | 0.091 | -1.236 | -0.98 |

## Per-seed runs

The individual runs behind the means above. Listed so the aggregate is checkable: whether every run improved on its baseline, and how the SD was computed, are both questions the summary table cannot answer on its own.

| Size | Seed | Baseline WER | FT WER | Δ (pp) |
|---|:---:|:---:|:---:|:---:|
| Whisper Tiny | 42 | 16.967% | 9.885% | -7.082 |
| Whisper Tiny | 43 | 16.967% | 9.811% | -7.156 |
| Whisper Tiny | 44 | 16.967% | 9.851% | -7.116 |
| Whisper Tiny | 45 | 16.967% | 9.885% | -7.082 |
| Whisper Tiny | 46 | 16.967% | 9.794% | -7.173 |
| Whisper Tiny | 47 | 16.967% | 9.897% | -7.071 |
| Whisper Small | 42 | 6.905% | 5.242% | -1.664 |
| Whisper Small | 43 | 6.905% | 5.253% | -1.652 |
| Whisper Small | 44 | 6.905% | 5.327% | -1.578 |
| Whisper Small | 45 | 6.905% | 5.453% | -1.453 |
| Whisper Small | 46 | 6.905% | 5.116% | -1.789 |
| Whisper Small | 47 | 6.905% | 5.099% | -1.806 |
| Whisper Medium | 42 | 5.259% | 4.079% | -1.179 |
| Whisper Medium | 43 | 5.259% | 4.279% | -0.98 |
| Whisper Medium | 44 | 5.259% | 4.102% | -1.157 |
| Whisper Medium | 45 | 5.259% | 4.108% | -1.151 |
| Whisper Medium | 46 | 5.259% | 4.022% | -1.236 |
| Whisper Medium | 47 | 5.259% | 4.045% | -1.214 |
