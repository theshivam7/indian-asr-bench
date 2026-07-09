# Phase 1 review & Phase 2 readiness

Phase 1 built the benchmark (nine pretrained ASR systems × two datasets), ran a
fine-tuning capacity study across three Whisper sizes, and instrumented a full-corpus
error-analysis pipeline. This is a retrospective on what that process found — what was
wrong and got fixed, what's a disclosed limitation rather than a bug, and what Phase 2
should do differently.

## What Phase 1 delivered

- **Full model×dataset coverage**: all nine registry models (Whisper Tiny/Base/Small/
  Medium/Large-v3/large-v3-turbo, Parakeet-TDT/CTC, Qwen3-ASR) now score on both
  TIE_shorts and Svarah. Four gaps (Tiny+Small on Svarah, large-v3-turbo+Parakeet-CTC on
  TIE) were identified via a direct registry audit and closed with real NSCC runs, not
  backfilled or estimated.
- **A fine-tuning capacity study**: Whisper Tiny/Small/Medium each fine-tuned once on
  TIE's official split and compared against their own pretrained baseline through an
  identical decoding pipeline, testing whether Medium's earlier null fine-tuning result
  was a capacity ceiling or a dataset limitation. Result: a capacity gradient consistent
  with the ceiling hypothesis (gain shrinks monotonically, Tiny −2.96pp → Small −1.17pp →
  Medium +0.20pp) but not statistically confirmed at 985 test clips after Holm
  correction — reported as suggestive, not conclusive.
- **A validated error-analysis instrument**: a full-corpus, multi-model consensus
  classifier for reference artifacts, cross-checked with two independent lines of
  evidence (clip-over-run agreement across architecturally unrelated models; inter-
  hypothesis vs. hypothesis-to-reference distance), rather than a hand-reviewed sample.
  It was then applied to a second, structurally different dataset (Svarah) and caught
  its own failure mode there (a naive 4.4%-vs-real-0.8% artifact-share error on Svarah's
  isolated-word items) — the classifier auditing itself is a substantive result, not
  just a clean pipeline run.
- **Statistical rigor throughout**: every headline comparison (36 pairs per dataset, 3
  more for the fine-tuning family) runs through a speaker- or recording-clustered paired
  bootstrap with Holm–Bonferroni correction, not raw point-estimate deltas.

## Weak points found, and how they were resolved

| Issue | Found by | Resolution |
|---|---|---|
| `whisper/` as a driver folder name would silently shadow the `openai-whisper` pip package on import, breaking every pretrained Whisper run | Proactive empirical test before committing (`sys.path` + `import whisper` resolved to the empty local folder) | Renamed to `whisper_asr/` before any commit landed |
| Whisper "Large" was displayed unversioned and linked to the `large-v2` HF repo, while the actual loaded checkpoint is `large-v3` | Direct check against the `openai-whisper` package's `_MODELS` alias table | Registry display fixed to "Whisper Large-v3", `model_id` set explicitly to `"large-v3"`, HF link corrected — no rerun needed, same weights either way |
| `wer_whisper_medium_ft.py` hand-rolled its own checkpoint/resume/manifest loop, duplicating `utils/inference_loop.py`'s shared logic | Code-review pass comparing all four engine drivers | Unified onto `run_transcription()`, extended with 1-arg/2-arg callback auto-detection (arity-based) so the shared loop now also serves the HF chunked-pipeline case; verified via a synthetic mock test covering the highest-risk case (non-contiguous checkpoint resume) |
| A speaker-disjoint/size-matched fine-tuning sub-study (real completed work) added narrative complexity without adding to the paper's core claim | Direct user judgment call | Removed entirely — code, 72 result files, 4 PBS job scripts, and all README references — rather than left half-integrated |
| `docs/DECODE_CONFIG.md` (a real, previously-committed provenance doc, later intentionally untracked to keep it off GitHub) went missing from disk mid-session | Broken-link check during the README rewrite | Recovered byte-for-byte from git history (`git show <commit>:docs/DECODE_CONFIG.md`), restored to the same untracked path |
| One NSCC-side commit landed unsigned (GPG key not configured on the interactive shell used for that push) | User noticed GitHub's "unverified" badge | Re-signed via a scoped `git rebase --exec "commit --amend -S"`, force-pushed with `--lease` |
| A local dataset-stats check attempted to load TIE's validation split without pointing `HF_DATASETS_CACHE` at scratch, hitting NSCC's quota-limited `$HOME` cache | User caught the wrong instruction mid-run | Corrected to set `HF_DATASETS_CACHE` explicitly before the load; partial `$HOME` cache cleaned up (home quota itself was never actually at risk — 53GB of a 200TB quota) |

None of these were caught by CI or a test suite — they surfaced through direct empirical
checks (import tests, package source inspection, broken-link scans) run *before* trusting
an assumption. That pattern held up better here than "review the code and hope" would
have.

## What's a documented limitation, not a bug

These are disclosed in the README's Limitations section and are re-stated here because
they should shape Phase 2 planning, not just sit as a caveat:

- **Fine-tuning is single-seed per size.** No multi-seed replication means the capacity
  gradient (Tiny > Small > Medium) is a suggestive point-estimate trend, not a
  statistically confirmed effect — none of the three deltas survives Holm correction at
  985 test clips.
- **TIE's fine-tuning comparison is speaker-matched, not speaker-disjoint.** 100% of test
  speakers appear in training. There is no clip-level leakage, but any measured gain
  partly reflects speaker adaptation. (The disjoint/size-matched sub-study that measured
  this directly was removed from the repo as a distraction from the main claim, but the
  underlying finding — TIE's official split has no disjoint option beyond 3.8h of usable
  data — is exactly what motivates the Phase 2 dataset recommendation below.)
- **Svarah has no public speaker IDs.** Statistical clustering falls back to a
  recording-tag proxy (3,232 clusters vs. the paper's 117 true speakers), which likely
  understates within-speaker correlation and gives anti-conservative confidence
  intervals.
- **The artifact classifier is unvalidated against human judgment.** It's backed by
  architecture-independent agreement evidence (models that share no decoder or training
  objective converging on the same "extra" words), which is strong indirect evidence, but
  the blind annotation protocol in `analysis/validation/` hasn't been run yet.

## Phase 2 recommendations

1. **Prefer AESRC2020's Indian-accent subset over Svarah for the next fine-tuning study,
   if licensing clears.** A local suitability analysis (kept out of the public repo,
   pending license confirmation) found it solves TIE's structural fine-tuning problem
   directly: its 481 test speakers have **zero** overlap with its 38 train speakers
   (natively disjoint, vs. TIE's 100% overlap), and its 17.5h of train audio is *entirely*
   usable disjoint data — over 4× the 3.8h TIE could offer after excluding overlapping
   speakers. This removes the single biggest caveat on Phase 1's fine-tuning result
   without needing a second confound-control study to prove it. The blocking item is a
   license sign-off; treat that as a Phase 2 prerequisite, not a nice-to-have.
2. **If licensing doesn't clear in time, run a true speaker-disjoint re-split of TIE
   before drawing further fine-tuning conclusions from it** — 3.8h is a real constraint,
   but it's a cleaner comparison than the current speaker-matched one, and the
   infrastructure for this (the disjoint-split logic) already existed once and can be
   rebuilt from the git history that removed it this phase, if needed.
3. **Run the blind human-annotation pass** (`analysis/validation/PROTOCOL.md`) before
   publishing the artifact-classifier numbers as a headline claim. The indirect evidence
   is strong, but it's still a heuristic standing in for ground truth.
4. **Multi-seed the next fine-tuning study** (3+ seeds per size minimum) so a capacity
   gradient claim can actually clear Holm correction rather than being reported as
   suggestive-only.
5. **Consider a three-way transfer matrix** (TIE ↔ AESRC-Indian ↔ Svarah) if the AESRC2020
   integration goes ahead — the suitability analysis flagged this as a natural, reviewer-
   visible extension once a second in-domain dataset is wired into the registry, and the
   registry's dataset-agnostic Stage 2/3 pipeline needs no changes to support it beyond
   one new `DatasetSpec`.
