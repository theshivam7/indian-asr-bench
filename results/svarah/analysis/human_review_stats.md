# Human review statistics: SVARAH, 60 clips

Source: `analysis/svarah_validation/review_sheet.csv`. Bootstrap B=10000, seed 42.
Wilcoxon signed-rank, two-sided, normal approximation with tie correction.

- Mean WER against the original reference: 56.7%
- Mean WER against the corrected reference: 48.8%
- Mean drop: 7.8 pp (95% bootstrap CI 4.0 to 12.2 pp)
- Wilcoxon p: 4.49e-04
- Clips that improve: 17 of 60

| Model | Mean drop (pp) | Wilcoxon p | p (Holm) |
|---|:---:|:---:|:---:|
| large | 4.8 | 4.9e-02 | 4.9e-02 |
| parakeet | 8.8 | 9.7e-04 | 3.9e-03 |
| parakeet_ctc | 8.6 | 7.3e-04 | 3.7e-03 |
| qwen3 | 9.7 | 1.0e-03 | 3.9e-03 |
| medium | 7.3 | 6.5e-03 | 1.3e-02 |

| Reviewer verdict | Clips |
|---|:---:|
| Genuine model error | 48 |
| Reference error | 12 |

| Error label | Clips |
|---|:---:|
| Number formatting | 29 |
| Reference error | 14 |
| Short utterance | 13 |
| Indian-language named entity | 13 |
| Acronym or code | 13 |
| Disfluency | 6 |
| Brand or product name | 6 |
| Truncated audio | 3 |
| Accent / pronunciation | 2 |
| Hindi named entity | 2 |
| English name or rare word | 1 |
