# Human review statistics: AESRC, 28 clips

Source: `analysis/aesrc_validation/review_sheet.csv`. Bootstrap B=10000, seed 42.
Wilcoxon signed-rank, two-sided, normal approximation with tie correction.

- Mean WER against the original reference: 51.5%
- Mean WER against the corrected reference: 48.4%
- Mean drop: 3.1 pp (95% bootstrap CI 0.3 to 6.9 pp)
- Wilcoxon p: 6.79e-02
- Clips that improve: 4 of 28

| Model | Mean drop (pp) | Wilcoxon p | p (Holm) |
|---|:---:|:---:|:---:|
| large | 5.7 | 6.8e-02 | 3.4e-01 |
| parakeet | 0.9 | 7.2e-01 | 7.2e-01 |
| parakeet_ctc | 2.3 | 1.4e-01 | 4.3e-01 |
| qwen3 | 3.1 | 1.4e-01 | 4.3e-01 |
| medium | 3.4 | 6.8e-02 | 3.4e-01 |

| Reviewer verdict | Clips |
|---|:---:|
| Genuine model error | 22 |
| Reference error | 3 |
| Not a real error | 2 |
| Unsure | 1 |

| Error label | Clips |
|---|:---:|
| Hindi named entity | 13 |
| Accent / pronunciation | 7 |
| Short utterance | 5 |
| Foreign named entity | 5 |
| Reference error | 4 |
| Indian-language named entity | 3 |
| Spelling variant | 3 |
| English name or rare word | 2 |
| Number formatting | 1 |
| Acronym or code | 1 |
| Brand or product name | 1 |
