# Multi-seed fine-tuning: aesrc (transcript_clean)

Each size trained repeatedly with only the seed changed, then scored through the identical HF pipeline as its own pretrained baseline.

The spread below is **across-seed** variation of the delta: how much the result moves when training is repeated. It is a different quantity from the within-run bootstrap CI in `finetune_capacity_summary.csv`, which describes sampling error over test clips. Report both, and do not pool them.

| Size | Params | Seeds | Baseline WER | FT WER (mean) | Δ mean (pp) | Δ SD (pp) | Δ min | Δ max |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Whisper Tiny | 39M | 6 | 17.447% | 10.597% | -6.85 | 1.029 | -7.337 | -4.751 |
