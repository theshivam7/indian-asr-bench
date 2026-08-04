# Speaker overlap across splits: TIE_shorts (data-leakage disclosure)

| Split | Clips | Unique speakers |
|-------|------:|----------------:|
| train | 7884 | 331 |
| validation | 986 | 280 |
| test | 986 | 280 |

## Train ∩ Test (the relevant leakage)

- Test speakers also present in train: **280 / 280** (100.0% of test speakers)
- Test clips spoken by a train-seen speaker: **986 / 986** (100.0% of test clips)
- Validation speakers also present in train: **280 / 280** (100.0% of validation speakers)

## Interpretation

> **Speaker-matched fine-tuning**: 100.0% of test clips come from speakers also seen during training. The fine-tuning improvement therefore partly reflects speaker adaptation. This is disclosed rather than hidden: it reflects the dataset's own official splits, which we did not modify.
