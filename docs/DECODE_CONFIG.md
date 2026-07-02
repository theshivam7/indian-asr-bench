# Decoding configuration & reproducibility notes

## What each engine uses (report these in the paper)

| Engine | Explicit settings | Everything else |
|---|---|---|
| openai-whisper (base/medium/large/large-v3-turbo) | `language="en"` (+ `fp16=False` on CPU) | library defaults: greedy decoding with temperature fallback (0.0 → 1.0 in 0.2 steps on quality-gate failure), `condition_on_previous_text=True`, default no-speech/compression thresholds |
| hf_whisper (medium_hf / medium_ft*) | chunked `transformers` pipeline (see `utils/transcribe_hf.py`) | library defaults |
| NeMo (parakeet / parakeet_ctc) | batch transcription, `batch_size=16` | library defaults |
| qwen3 | `language="English"`, `max_new_tokens=512` | library defaults |

Per-run values are recorded in `results/<dataset>/stage1_raw_transcripts/wer_<model>_manifest.json`
(model + dataset revisions, package versions, git commit, decode kwargs, host, timestamp).

## Known nondeterminism (disclosed, not hidden)

openai-whisper's **temperature fallback is stochastic**: clips that fail the
compression-ratio/log-prob gates at temperature 0 are re-decoded at sampled
temperatures, so re-running Stage 1 from scratch can produce slightly different
transcripts for those clips. `condition_on_previous_text=True` additionally
couples 30-s windows in clips longer than 30 s.

**The committed Stage-1 raw CSVs are therefore the reproducibility anchor**:
every number in the paper (Stage 2 scoring, Stage 3 statistics/analyses,
figures) regenerates deterministically from them with no GPU. Decode settings
were left at community defaults deliberately — they are what practitioners run,
and changing them mid-project would break comparability with completed runs.

## Version pinning

- HF dataset revisions are pinned in `utils/registry.py` (`hf_revision`) and
  passed to `load_dataset` — an upstream dataset update cannot silently change
  the benchmark.
- Python package pins live in `environments/*.yaml` + per-task
  `requirements.txt`; the manifest records the versions *actually* installed at
  run time, which is what counts when environments drift.
- The `whisper_norm` evaluation mode uses `whisper_normalizer==0.1.0`, verified
  byte-identical to `openai/whisper`'s reference `EnglishTextNormalizer` on all
  7,391 distinct reference/hypothesis strings in the TIE corpus (2026-07-03).
