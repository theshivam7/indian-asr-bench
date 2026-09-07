# Human review statistics: TIE, 49 clips

Source: `analysis/tie_validation/review_sheet.csv`. Bootstrap B=10000, seed 42.
Wilcoxon signed-rank, two-sided, normal approximation with tie correction.

- Mean WER against the original reference: 64.8%
- Mean WER against the corrected reference: 17.0%
- Mean drop: 47.8 pp (95% bootstrap CI 40.3 to 55.9 pp)
- Wilcoxon p: 1.18e-09
- Clips that improve: 48 of 49

| Model | Mean drop (pp) | Wilcoxon p | p (Holm) |
|---|:---:|:---:|:---:|
| large | 43.7 | 4.5e-09 | 9.0e-09 |
| parakeet | 51.8 | 1.1e-09 | 5.5e-09 |
| parakeet_ctc | 50.3 | 1.2e-09 | 5.5e-09 |
| qwen3 | 50.7 | 1.2e-09 | 5.5e-09 |
| medium | 42.2 | 3.5e-08 | 3.5e-08 |
