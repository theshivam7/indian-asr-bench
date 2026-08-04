# TIE speaker-disjoint control: does repairing the split remove the gain?

Recovered from the original job logs (`ft_disjoint.log`, `score.log`) and committed here so the
result is checkable. The per-clip transcripts for these six runs were not retained, so unlike
every other table in this repository these numbers cannot be recomputed from
`stage1_raw_transcripts/`. They are the scoring output as the job printed it.

## The question

TIE's official split places 100% of test speakers inside train
([`speaker_overlap.md`](speaker_overlap.md)). On that split, fine-tuning Whisper Medium moves WER
by +0.20 pp, a null. Two readings are possible: the corpus cannot support a gain at this
capacity, or the split is hiding one. Repairing the split answers it, but repairing it costs
training data (567 of 7,200 clips survive removing every overlapping speaker), so a smaller
training set is confounded with the split change.

The control below separates the two. Three seeds train on the speaker-disjoint subset, and three
more train on a **size-matched** speaker-overlapping subset at the same clip budget. If the
regression comes from disjointness it appears only in the first group; if it comes from having
less data it appears in both.

## Corpus WER (%), TIE test split, all five modes

| Model | T-raw | T-clean | HF-raw | HF-clean | W-norm |
|---|---:|---:|---:|---:|---:|
| `medium_hf` (pretrained baseline) | 14.75 | 14.42 | 17.72 | 15.51 | 14.23 |
| `medium_ft` (official split) | 14.71 | 14.61 | 17.70 | 15.70 | 14.31 |
| `medium_ft_disjoint` (seed 42) | 16.53 | 16.17 | 19.39 | 17.10 | 15.95 |
| `medium_ft_disjoint_s43` | 15.14 | 14.80 | 18.01 | 15.73 | 14.53 |
| `medium_ft_disjoint_s44` | 15.58 | 15.20 | 18.52 | 16.16 | 15.01 |
| `medium_ft_sizematch_s42` | 14.70 | 14.33 | 17.60 | 15.28 | 14.16 |
| `medium_ft_sizematch_s43` | 14.82 | 14.40 | 17.57 | 15.36 | 14.44 |
| `medium_ft_sizematch_s44` | 14.80 | 14.40 | 17.71 | 15.49 | 14.31 |

## Paired significance vs. the same pretrained baseline

Speaker-clustered bootstrap over 280 speakers, 985 scored clips, `transcript_clean`,
Holm-corrected within each family of three.

| Run | WER | Δ vs. baseline | 95% CI | p | p (Holm) |
|---|---:|---:|:---:|---:|---:|
| disjoint, seed 42 | 16.17% | **+1.75 pp** | [+0.13, +4.17] | 0.016 | **0.048** |
| disjoint, seed 43 | 14.80% | +0.38 pp | [-0.01, +0.74] | 0.058 | 0.116 |
| disjoint, seed 44 | 15.20% | +0.79 pp | [-0.18, +2.25] | 0.163 | 0.163 |
| size-matched, seed 42 | 14.33% | -0.09 pp | [-0.45, +0.23] | 0.581 | 1.000 |
| size-matched, seed 43 | 14.40% | -0.02 pp | [-1.78, +1.53] | 0.997 | 1.000 |
| size-matched, seed 44 | 14.40% | -0.02 pp | [-1.85, +1.90] | 0.985 | 1.000 |

## Reading

All three speaker-disjoint runs move in the same direction, away from the baseline, and one
clears Holm correction at +1.75 pp. All three size-matched controls land flat, none approaching
significance, with the two directionally negative by a hundredth of a point. The training budget
is the same in both groups, so the smaller training set does not explain the disjoint runs'
behaviour; what differs is whether test speakers were seen during training.

So the official split's +0.20 pp null is not a neutral result. Once the speaker overlap is
removed the fine-tune is worse than doing nothing, which means the official-split number was
being propped up by adaptation to test speakers rather than reporting a genuine failure to learn.
That is the concrete cost of a speaker-entangled split: not just an inflated gain, but a null
that conceals a regression.

This is what motivated moving the capacity study to AESRC2020, whose test split is natively
speaker-disjoint, where the same protocol does yield an attributable gain at every size
([Fine-tuning and split design](../../../SUMMARY.md#fine-tuning-and-split-design)).

## Caveats

- **Not recomputable from this repository.** Per-clip transcripts for these six runs were not
  retained and the cluster account is no longer accessible. The tables above are the scoring
  output verbatim. Every other result in this repository recomputes from committed transcripts;
  this one does not, and should be weighted accordingly.
- **Three seeds, one size.** Enough to show the direction and to rule out the training-budget
  explanation, not enough for a seed-level significance claim.
- **567 training clips** is small in absolute terms. The size-matched control makes the
  comparison fair, but both arms are data-starved relative to the 7,197-clip official-split run.
