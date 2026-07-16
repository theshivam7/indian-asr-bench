# Speaker Overlap Across Splits — AESRC2020 (Indian) (data-leakage disclosure)

| Split | Clips | Unique speakers |
|-------|------:|----------------:|
| train | 12820 | 38 |
| valid | 532 | 38 |
| test | 1731 | 481 |

## Train ∩ Test (the relevant leakage)

- Test speakers also present in train: **0 / 481** (0.0% of test speakers)
- Test clips spoken by a train-seen speaker: **0 / 1731** (0.0% of test clips)
- Validation speakers also present in train: **38 / 38** (100.0% of validation speakers)

## Interpretation

> No speaker overlap between train and test — the fine-tuning gain reflects genuine generalization to unseen speakers.
