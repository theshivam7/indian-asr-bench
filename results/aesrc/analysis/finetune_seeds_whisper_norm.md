# Multi-seed fine-tuning: aesrc (whisper_norm)

Each size trained repeatedly with only the seed changed, then scored through the identical HF pipeline as its own pretrained baseline.

The spread below is **across-seed** variation of the delta: how much the result moves when training is repeated. It is a different quantity from the within-run bootstrap CI in `finetune_capacity_summary.csv`, which describes sampling error over test clips. Report both, and do not pool them.

| Size | Params | Seeds | Baseline WER | FT WER (mean) | Δ mean (pp) | Δ SD (pp) | Δ min | Δ max |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Whisper Tiny | 39M | 6 | 16.967% | 9.854% | -7.114 | 0.043 | -7.173 | -7.071 |
| Whisper Small | 244M | 6 | 6.905% | 5.248% | -1.657 | 0.132 | -1.806 | -1.453 |
| Whisper Medium | 769M | 6 | 5.259% | 4.106% | -1.153 | 0.091 | -1.236 | -0.98 |
