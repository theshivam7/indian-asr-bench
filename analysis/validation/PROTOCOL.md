# Human validation protocol for the artifact classifier

## Purpose

The benchmark's central diagnostic classifies clips as dataset artifacts
(`clip_over_run`, `content_mismatch`) vs. unflagged, from cross-model consensus
recall/length-ratio statistics (`analysis/error_analysis.py`). Clips with
references shorter than 4 words are a third outcome, `short_ref`, the
recall/ratio signals are quantized to uselessness there, so the classifier
declares them **unclassifiable** rather than flagging them. This protocol
measures the classifier against blind human judgment, so the paper can report
validated precision/recall instead of an unvalidated heuristic. `short_ref`
clips are excluded from every stratum (they are outside the instrument's
domain, so they belong in neither the precision nor the recall estimate);
`build_validation_sample.py` enforces this.

## Sampling design (stratified; built by `build_validation_sample.py`)

Consensus flags are rare (~1% of the corpus), so "N per predicted category" is
impossible. Instead, four strata over the full corpus:

| Stratum | Definition | Sample |
|---|---|---|
| A | ALL consensus-flagged clips (`clip_over_run` + `content_mismatch`) | census (cap 60) |
| B | Borderline: flagged by ≥1 individual model, but not by consensus | census (cap 30) |
| C | High-WER unflagged: top consensus-mean-WER clips never flagged by any model | 40 |
| D | Random unflagged (everything else) | 20 |

A yields classifier **precision**. B/C/D yield the false-negative rates needed
for a stratum-weighted **recall** estimate (D's weight is large and its sample
small, so the recall CI is honest-but-wide; the paper reports it as such).

Sampling is seeded (default 42) and reproducible; regenerate after the 9-model
TIE results are synced so consensus uses all models.

## Blinding

The annotator receives ONLY: an audio file (randomized item number, not the
clip ID) and the reference transcript. Never: any model output, the predicted
category, WER, or which stratum an item came from. The question is a property
of the *data*, so model outputs must not anchor the judgment.

## Annotation task (one question per item)

Listen to the full clip, reading the reference alongside. Choose ONE label:

- **A. Faithful**: the reference covers the spoken content of the clip.
  Small deviations (a misspelled word, a missed filler, punctuation) still
  count as faithful.
- **B. Audio exceeds reference**: you clearly hear complete additional speech
  (a sentence/clause or more) that the reference does not cover, the
  transcript stops or starts short of the audio.
- **C. Mismatch**: the reference substantially does not correspond to what is
  spoken (wrong segment, misaligned transcript, different content).
- **D. Unusable audio**: no intelligible speech (silence, noise, music).
- **E. Unsure**: cannot decide after two listens. (Use sparingly; E items are
  reported but excluded from precision/recall.)

Ground-truth mapping: **artifact = B, C, or D**; **non-artifact = A**.
Decision rule of thumb: A vs B hinges on *complete extra utterances*, not a
word or two; A vs C hinges on whether you could follow the audio using the
reference at all.

Fill the `label` column of `annotation_sheet.csv` with A/B/C/D/E. Optional
`notes` column for anything odd. Expected pace ≈ 45–60 s/item → ~2 h for ~130
items.

## Second annotator (recommended, optional)

A second person labels the `overlap = yes` subset (~60 items, same sheet,
`label_2` column). `score_validation.py` then reports Cohen's κ on the binary
artifact decision. If κ ≥ 0.6 (substantial), the construct is reliable.

## Metrics reported (by `score_validation.py`)

- Precision of the consensus artifact flag, with Wilson 95% CI (stratum A).
- Per-category precision (`clip_over_run` → B; `content_mismatch` → C/D).
- Stratum-weighted recall estimate of the consensus flag, with a bootstrap CI
  over the sampled strata (explicitly labeled an estimate).
- Cohen's κ (binary artifact) if a second annotator labeled the overlap set.

## In the paper

Methods: one paragraph describing this protocol (stratified blind audit,
census of flags, seeded sampling). Results: one table, precision, recall
estimate, κ, and n per stratum. All claims about artifact prevalence then cite
the *validated* classifier.
